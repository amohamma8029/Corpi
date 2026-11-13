"""Expired PDF actions keep minimal, non-resumable replay receipts."""

import asyncio
import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient
from test_import_expiry import EXPIRED, HEADERS, TOKEN, make_history
from test_source_expiry import assert_source_purged, finish_source_cleanup

from core.api.app import create_app
from core.ingest.files import ManagedPdfFiles
from core.ingest.service import PdfIngestService
from core.jobs.sweeper import TrashSweeper
from core.persistence.errors import DeletedPdfAcquisitionError, PersistenceError
from core.persistence.models import encode_json, new_id


def test_expired_actions_reject_exact_changed_and_binary_replays(store, monkeypatch):
    history = make_history(store, monkeypatch, shared=True)
    actions = [
        store.load_pdf_acquisition(record.item.acquisition_id)
        for record in (history.old, history.current)
    ]
    neighbour = store.load_pdf_acquisition(history.neighbour.item.acquisition_id)
    store.trash_source(history.old.item.source_id, EXPIRED)
    restarted = finish_source_cleanup(store, history.old.item.source_id, "startup")
    assert_source_purged(restarted, history, shared=True)
    ingest = PdfIngestService(restarted, "receipt-replay")
    with TestClient(
        create_app(TOKEN, store=restarted, pdf_ingest=ingest), base_url="http://127.0.0.1"
    ) as client:
        for action in actions:
            path = f"/notebooks/{action.notebook_id}/pdf-acquisitions"
            payload = {
                "retry_key": action.retry_key,
                "filename": action.display_filename,
                "intent": action.intent,
                "target_source_id": action.target_source_id,
            }
            response = client.post(path, headers=HEADERS, json=payload)
            assert (
                response.status_code == 409
                and response.json()["detail"]["code"] == "source_deleted"
            )
            for changed in (
                {"filename": "different.pdf"},
                {"intent": "add", "target_source_id": None}
                if action.intent == "replace"
                else {"intent": "replace", "target_source_id": history.neighbour.item.source_id},
                {"target_source_id": history.neighbour.item.source_id, "intent": "replace"},
            ):
                response = client.post(path, headers=HEADERS, json={**payload, **changed})
                assert response.status_code == 409
                assert response.json()["detail"]["code"] == "invalid_acquisition"
            # Another notebook must not claim an expired global action key.
            other = client.post(
                f"/notebooks/{neighbour.notebook_id}/pdf-acquisitions",
                headers=HEADERS,
                json={**payload, "intent": "add", "target_source_id": None},
            )
            assert other.status_code == 409
            for response in (
                client.get(f"/pdf-acquisitions/{action.id}", headers=HEADERS),
                client.post(f"/pdf-acquisitions/{action.id}/retry", headers=HEADERS),
                client.post(f"/pdf-acquisitions/{action.id}/cancel", headers=HEADERS),
                client.put(
                    f"/pdf-acquisitions/{action.id}/content",
                    headers={**HEADERS, "Content-Type": "application/pdf"},
                    content=history.old.content,
                ),
            ):
                assert (
                    response.status_code == 409
                    and response.json()["detail"]["code"] == "source_deleted"
                )
            assert client.get(f"/pdf-acquisitions/{action.id}").status_code == 401
            assert (
                client.post(
                    path, headers={**HEADERS, "Origin": "https://hostile.example"}, json=payload
                ).status_code
                == 403
            )
            assert (
                client.post(
                    path, headers={**HEADERS, "Host": "hostile.example"}, json=payload
                ).status_code
                == 403
            )
            with pytest.raises(DeletedPdfAcquisitionError):
                restarted.load_pdf_acquisition(action.id)
            for mutate in (
                lambda action=action: restarted.begin_pdf_upload(action.id, "late"),
                lambda action=action: restarted.claim_pdf_finalization(
                    action.id, "late", recovery=True
                ),
                lambda action=action: restarted.resume_failed_pdf_acquisition(action.id, "late"),
            ):
                with pytest.raises(PersistenceError):
                    mutate()
            with restarted.database.connect() as connection:
                receipt = dict(
                    connection.execute(
                        "SELECT * FROM deleted_pdf_acquisitions WHERE id=?", (action.id,)
                    ).fetchone()
                )
                assert set(receipt) == {
                    "id",
                    "retry_key",
                    "notebook_id",
                    "request_sha256",
                    "created_at",
                    "deleted_at",
                }
                expected = hashlib.sha256(
                    encode_json(
                        [
                            action.notebook_id,
                            action.intent,
                            action.target_source_id,
                            action.display_filename,
                        ]
                    ).encode("utf-8")
                ).hexdigest()
                assert receipt["request_sha256"] == expected
        assert restarted.load_pdf_acquisition(neighbour.id) == neighbour
        assert restarted.list_unfinished_pdf_acquisitions(actions[0].notebook_id) == ()
        assert restarted.list_recoverable_pdf_acquisitions() == ()
    assert_source_purged(restarted, history, shared=True)


def test_deleted_action_keys_and_ids_cannot_be_reinserted_or_reassigned(store, monkeypatch):
    history = make_history(store, monkeypatch)
    action = store.load_pdf_acquisition(history.current.item.acquisition_id)
    neighbour = store.load_pdf_acquisition(history.neighbour.item.acquisition_id)
    store.trash_source(history.old.item.source_id, EXPIRED)
    restarted = finish_source_cleanup(store, history.old.item.source_id, "maintenance")
    with restarted.database.connect() as connection:
        # Bypass the store intentionally: database guards cover both identities,
        # even when a caller copies an otherwise-valid surviving acquisition.
        for field, value in (("id", action.id), ("retry_key", action.retry_key)):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f"UPDATE pdf_acquisitions SET {field}=? WHERE id=?", (value, neighbour.id)
                )
            fields = [row[1] for row in connection.execute("PRAGMA table_info(pdf_acquisitions)")]
            original = dict(
                connection.execute(
                    "SELECT * FROM pdf_acquisitions WHERE id=?", (neighbour.id,)
                ).fetchone()
            )
            copied = {
                **original,
                "id": new_id(),
                "retry_key": new_id(),
                "staging_relative_path": f"files/staging/{new_id()}.part",
                field: value,
            }
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f"INSERT INTO pdf_acquisitions({','.join(fields)}) "
                    f"VALUES({','.join('?' for _ in fields)})",
                    tuple(copied[key] for key in fields),
                )
        # The original replacement constraint must remain enforced as well.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE pdf_acquisitions SET intent='replace',target_source_id=NULL WHERE id=?",
                (neighbour.id,),
            )
    assert restarted.load_pdf_acquisition(neighbour.id) == neighbour


def test_expiry_removes_complete_failed_and_unjournalled_completed_staging(store, monkeypatch):
    history = make_history(store, monkeypatch)
    action = store.load_pdf_acquisition(history.current.item.acquisition_id)
    files = ManagedPdfFiles(store.database.config.data_dir)
    # Model a completed import's crash between durable finalization and cleanup
    # journalling. Its receipt must not discard the only remaining staging owner.
    with files.open_staging(action.staging_relative_path) as stage:
        stage.write(history.current.content)
    with store.database.connect() as connection:
        connection.execute(
            "DELETE FROM acquisition_file_cleanup WHERE acquisition_id=?", (action.id,)
        )
    failed = store.create_pdf_acquisition(
        new_id(),
        action.notebook_id,
        "replace",
        "failed.pdf",
        "failed-owner",
        history.old.item.source_id,
    )
    uploading = store.begin_pdf_upload(failed.id, "failed-owner")
    with files.open_staging(failed.staging_relative_path) as stage:
        stage.write(history.current.content)
    staged = store.record_pdf_upload(
        failed.id,
        "failed-owner",
        uploading.finalization_generation,
        hashlib.sha256(history.current.content).hexdigest(),
        len(history.current.content),
    )
    failed = store.fail_pdf_acquisition(
        staged.id, "failed-owner", staged.finalization_generation, "controlled_failure"
    )
    store.trash_source(history.old.item.source_id, EXPIRED)
    restarted = finish_source_cleanup(store, history.old.item.source_id, "startup")
    assert_source_purged(restarted, history, shared=False)
    assert [
        (store.database.config.data_dir / deleted.staging_relative_path).exists()
        for deleted in (action, failed)
    ] == [False, False]
    for deleted in (action, failed):
        with pytest.raises(DeletedPdfAcquisitionError):
            restarted.create_pdf_acquisition(
                deleted.retry_key,
                deleted.notebook_id,
                deleted.intent,
                deleted.display_filename,
                "replay",
                deleted.target_source_id,
            )


def test_cleanup_failure_preserves_action_ownership_until_receipts_can_commit(store, monkeypatch):
    history = make_history(store, monkeypatch)
    source_id = history.old.item.source_id
    store.trash_source(source_id, EXPIRED)
    ingest = PdfIngestService(store, "cleanup-failure")
    original_remove = ingest.files.remove_recorded

    def fail_staging(path, *args, **kwargs):
        if store.database.config.data_dir / path == history.staging:
            raise OSError("controlled staging unlink failure")
        return original_remove(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(ingest.files, "remove_recorded", fail_staging)
        try:
            ingest.maintenance()
            assert TrashSweeper(store).sweep_once() == ()
        finally:
            asyncio.run(ingest.stop())
    assert history.staging.exists() and store.load_trash(source_id) is not None
    with store.database.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM deleted_pdf_acquisitions").fetchone()[0] == 0
        )
    for record in (history.old, history.current):
        assert store.load_pdf_acquisition(record.item.acquisition_id).state == "completed"
    assert_source_purged(finish_source_cleanup(store, source_id, "startup"), history, shared=False)
