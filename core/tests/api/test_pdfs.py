from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from core.api.app import create_app
from core.ingest.config import PdfIngestLimits
from core.ingest.files import ManagedPdfFiles
from core.ingest.locks import FingerprintLockCoordinator
from core.ingest.service import PdfIngestService
from core.jobs.sweeper import TrashSweeper
from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import PersistenceError
from core.persistence.models import (
    AttemptGuard,
    ChunkPassage,
    ExtractionOutcome,
    ExtractionVersion,
    IndexVersion,
    ManagedFile,
    Passage,
    RetrievalChunk,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore

TOKEN = "test-pdf-token"
ORIGIN = "http://tauri.localhost"


def minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{number} 0 obj\n".encode("ascii"))
        body.extend(obj)
        body.extend(b"\nendobj\n")
    xref_offset = len(body)
    body.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    body.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(body)


@pytest.fixture()
def app_client(tmp_path: Path):
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    ingest = PdfIngestService(store, "core-test")
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=ingest)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, store, database


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN}


def acquire(client: TestClient, notebook_id: str, retry_key: str, content: bytes) -> dict:
    created = client.post(
        f"/notebooks/{notebook_id}/pdf-acquisitions",
        headers=auth_headers(),
        json={"retry_key": retry_key, "filename": f"{retry_key}.pdf"},
    )
    assert created.status_code == 201, created.text
    uploaded = client.put(
        f"/pdf-acquisitions/{created.json()['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=content,
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()


def test_unfinished_pdf_attempts_are_authenticated_scoped_and_truthful(app_client) -> None:
    client, store, database = app_client
    notebook_a = store.create_notebook(new_id(), "Interrupted imports")
    notebook_b = store.create_notebook(new_id(), "Other notebook")
    path = f"/notebooks/{notebook_a.id}/pdf-acquisitions"
    partial = store.create_pdf_acquisition(
        "partial", notebook_a.id, "add", "partial.pdf", "core-test"
    )
    receiving = store.begin_pdf_upload(partial.id, "core-test")
    with ManagedPdfFiles(database.config.data_dir).open_staging(
        receiving.staging_relative_path
    ) as stage:
        stage.write(b"%PDF-partial")
    client.app.state.pdf_ingest.reconcile_startup()
    store.create_pdf_acquisition("other", notebook_b.id, "add", "other.pdf", "core-test")
    cancelled = store.create_pdf_acquisition(
        "cancelled", notebook_a.id, "add", "cancelled.pdf", "core-test"
    )
    client.post(f"/pdf-acquisitions/{cancelled.id}/cancel", headers=auth_headers())
    acquire(client, notebook_a.id, "completed", minimal_pdf("completed"))

    assert client.get(path).status_code == 401
    assert (
        client.get(f"/notebooks/{new_id()}/pdf-acquisitions", headers=auth_headers()).status_code
        == 404
    )
    response = client.get(path, headers=auth_headers())
    assert response.status_code == 200
    assert len(response.json()) == 1
    attempt = response.json()[0]
    assert attempt["id"] == partial.id
    assert attempt["state"] == "failed"
    assert attempt["failure_code"] == "upload_interrupted"
    assert attempt["requires_upload"] is True
    assert attempt["can_retry"] is False
    assert attempt["can_cancel"] is True
    assert attempt["input_sha256"] is None
    assert "staging_relative_path" not in attempt
    # File reselection resumes the same intent instead of creating a second attempt.
    content = minimal_pdf("reselected")
    resumed = client.put(
        f"/pdf-acquisitions/{partial.id}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=content,
    )
    assert resumed.status_code == 200
    assert resumed.json()["id"] == partial.id
    assert resumed.json()["state"] == "completed"
    assert client.get(path, headers=auth_headers()).json() == []


def publish_pdf_import(store: PersistenceStore, source_id: str, import_id: str) -> None:
    item = store.load_pdf_import(import_id)
    timestamp = utc_timestamp()
    extraction_id = new_id()
    extraction = ExtractionVersion(
        extraction_id,
        source_id,
        item.source_version_id,
        "test-pdf",
        "1",
        "test-normalization",
        1,
        {},
        timestamp,
    )
    passage = Passage(
        new_id(),
        source_id,
        item.source_version_id,
        extraction_id,
        0,
        "Synthetic checked PDF evidence.",
        1,
        {"page": 1},
        timestamp,
    )
    with store.database.connect() as connection:
        extract_job_id = connection.execute(
            "SELECT id FROM jobs WHERE kind='pdf-extract' AND source_version_id=?",
            (item.source_version_id,),
        ).fetchone()[0]
    store.claim_job(extract_job_id, "core-test")
    extract_attempt = store.start_step_attempt(extract_job_id, "extract-text")
    extract_guard = AttemptGuard("core-test", "extract-text", extract_attempt.attempt_number)
    store.commit_extraction_step(
        extract_job_id,
        "extract-text",
        extraction,
        (passage,),
        1,
        {"pages": 1},
        guard=extract_guard,
        outcome=ExtractionOutcome(
            extraction_id, source_id, item.source_version_id, "complete", timestamp
        ),
    )
    store.finish_step_attempt(extract_job_id, "extract-text", extract_attempt.attempt_number)
    store.finish_job(extract_job_id, "completed")

    index_id = new_id()
    chunk = RetrievalChunk(
        new_id(), index_id, source_id, item.source_version_id, extraction_id, 0, passage.text
    )
    mapping = ChunkPassage(
        chunk.id,
        passage.id,
        index_id,
        source_id,
        item.source_version_id,
        extraction_id,
        0,
        0,
        len(passage.text),
    )
    index = IndexVersion(
        index_id,
        source_id,
        item.source_version_id,
        extraction_id,
        "keyword",
        "test-v1",
        1,
        {},
        None,
        None,
        timestamp,
    )
    index_job = store.create_job(
        f"test-index:{item.source_version_id}",
        "test-index",
        source_id,
        source_version_id=item.source_version_id,
        extraction_version_id=extraction_id,
    )
    store.create_job_status(index_job.id)
    store.claim_job(index_job.id, "core-test")
    index_attempt = store.start_step_attempt(index_job.id, "build-index")
    index_guard = AttemptGuard("core-test", "build-index", index_attempt.attempt_number)
    store.commit_index_step(
        index_job.id,
        "build-index",
        index,
        (chunk,),
        (mapping,),
        1,
        {"chunks": 1},
        guard=index_guard,
    )
    store.finish_step_attempt(index_job.id, "build-index", index_attempt.attempt_number)
    publish_attempt = store.start_step_attempt(index_job.id, "publish-source")
    store.publish_readiness(
        index_job.id,
        source_id,
        item.source_version_id,
        [index_id],
        guard=AttemptGuard("core-test", "publish-source", publish_attempt.attempt_number),
    )
    store.finish_step_attempt(index_job.id, "publish-source", publish_attempt.attempt_number)
    store.finish_job(index_job.id, "completed")


def test_pdf_upload_persists_verified_original_and_duplicate_add_is_noop(app_client) -> None:
    client, store, database = app_client
    created = client.post("/notebooks", headers=auth_headers(), json={"title": "Research"})
    assert created.status_code == 201
    notebook_id = created.json()["id"]
    original = minimal_pdf("first import")

    saved = acquire(client, notebook_id, "first-action", original)
    repeated = acquire(client, notebook_id, "duplicate-action", original)

    assert saved["state"] == "completed"
    assert repeated["outcome_kind"] == "duplicate"
    assert repeated["source_id"] == saved["source_id"]
    sources = client.get(f"/notebooks/{notebook_id}/sources", headers=auth_headers())
    assert len(sources.json()) == 1
    assert store.load_pdf_acquisition(saved["id"]).outcome_import_id == saved["import_id"]
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        managed = connection.execute("SELECT * FROM managed_files").fetchone()
        assert managed["relative_path"].startswith("files/objects/")
    cancelled = client.post(
        f"/sources/{saved['source_id']}/imports/{saved['import_id']}/cancel",
        headers=auth_headers(),
    )
    assert cancelled.status_code == 200, cancelled.text
    assert store.load_pdf_import(saved["import_id"]).state == "failed"
    retried = client.post(
        f"/sources/{saved['source_id']}/imports/{saved['import_id']}/retry",
        headers=auth_headers(),
    )
    assert retried.status_code == 200, retried.text
    assert retried.json()["status"] == "queued"
    assert store.load_pdf_import(saved["import_id"]).state == "pending"
    opened = client.get(
        f"/sources/{saved['source_id']}/imports/{saved['import_id']}/pdf",
        headers=auth_headers(),
    )
    assert opened.status_code == 200
    assert opened.headers["content-type"] == "application/pdf"
    assert opened.content == original
    storage = client.get("/storage/originals", headers=auth_headers())
    assert storage.status_code == 200
    assert storage.json()["max_upload_bytes"] == 25_000_000
    assert storage.json()["original_file_count"] == 1
    replay = client.put(
        f"/pdf-acquisitions/{saved['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=original,
    )
    assert replay.status_code == 200
    assert replay.json()["import_id"] == saved["import_id"]
    conflict = client.put(
        f"/pdf-acquisitions/{saved['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=minimal_pdf("different bytes"),
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "retry_key_conflict"
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2


def test_saved_pdf_remains_openable_after_the_users_file_is_moved_and_processing_fails(
    tmp_path: Path,
) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    ingest = PdfIngestService(store, "core-original-check")
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=ingest)
    user_file = tmp_path / "user-selected.pdf"
    content = minimal_pdf("stored copy")
    user_file.write_bytes(content)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        notebook = client.post(
            "/notebooks", headers=auth_headers(), json={"title": "Moved original"}
        ).json()
        saved = acquire(client, notebook["id"], "moved-original", user_file.read_bytes())
        moved_file = user_file.with_name("user-selected-moved.pdf")
        user_file.rename(moved_file)
        store.cancel_pdf_import_processing(saved["import_id"])

        opened = client.get(
            f"/sources/{saved['source_id']}/imports/{saved['import_id']}/pdf",
            headers=auth_headers(),
        )

    assert moved_file.is_file()
    assert opened.status_code == 200
    assert opened.content == content
    assert store.load_pdf_import(saved["import_id"]).state == "failed"


def test_two_notebooks_share_one_physical_original_without_sharing_sources(app_client) -> None:
    client, store, database = app_client
    first_notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Notebook one"}
    ).json()
    second_notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Notebook two"}
    ).json()
    content = minimal_pdf("shared physical bytes")

    first = acquire(client, first_notebook["id"], "share-one", content)
    second = acquire(client, second_notebook["id"], "share-two", content)

    assert first["source_id"] != second["source_id"]
    assert first["import_id"] != second["import_id"]
    with database.connect() as connection:
        managed_ids = {
            row[0]
            for row in connection.execute(
                "SELECT managed_file_id FROM source_versions ORDER BY source_id"
            )
        }
        assert len(managed_ids) == 1
        assert connection.execute("SELECT count(*) FROM managed_files").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM original_registry").fetchone()[0] == 1
    for result in (first, second):
        saved = store.load_import_managed_file(result["import_id"])
        assert (
            ManagedPdfFiles(database.config.data_dir).read_verified(
                saved.relative_path, saved.sha256, saved.size_bytes
            )
            == content
        )
    report = client.get("/storage/originals", headers=auth_headers()).json()
    assert report["original_file_count"] == 1
    assert report["original_bytes"] == len(content)


def test_real_pdf_near_the_upload_limit_is_saved_and_opened_completely(app_client) -> None:
    client, store, database = app_client
    notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Near PDF limit"}
    ).json()
    base = minimal_pdf("near the size boundary")
    content = base + b" " * (25_000_000 - len(base))
    saved = acquire(client, notebook["id"], "near-limit", content)
    managed = store.load_import_managed_file(saved["import_id"])
    opened = client.get(
        f"/sources/{saved['source_id']}/imports/{saved['import_id']}/pdf",
        headers=auth_headers(),
    )

    assert managed.size_bytes == 25_000_000
    assert opened.status_code == 200
    assert opened.content == content


def test_filename_does_not_merge_different_bytes_and_unchanged_replace_is_noop(app_client) -> None:
    client, store, database = app_client
    notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Names aren't identity"}
    ).json()
    first_bytes = minimal_pdf("first content")
    second_bytes = minimal_pdf("different content")

    first_action = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={"retry_key": "same-name-first", "filename": "same.pdf"},
    ).json()
    first = client.put(
        f"/pdf-acquisitions/{first_action['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=first_bytes,
    ).json()
    second_action = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={"retry_key": "same-name-second", "filename": "same.pdf"},
    ).json()
    second = client.put(
        f"/pdf-acquisitions/{second_action['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=second_bytes,
    ).json()
    unchanged_action = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "unchanged-replace",
            "filename": "renamed.pdf",
            "intent": "replace",
            "target_source_id": first["source_id"],
        },
    ).json()
    unchanged = client.put(
        f"/pdf-acquisitions/{unchanged_action['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=first_bytes,
    )

    assert first["source_id"] != second["source_id"]
    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json()["outcome_kind"] == "unchanged"
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM source_versions").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2


def test_simultaneous_staged_imports_create_one_canonical_original(app_client) -> None:
    _, store, database = app_client
    content = minimal_pdf("simultaneous original")
    checksum = hashlib.sha256(content).hexdigest()
    files = ManagedPdfFiles(database.config.data_dir)
    notebooks = [
        store.create_notebook(new_id(), "Concurrent A"),
        store.create_notebook(new_id(), "Concurrent B"),
    ]
    acquisitions = []
    for index, notebook in enumerate(notebooks):
        acquisition = store.create_pdf_acquisition(
            f"concurrent-{index}",
            notebook.id,
            "add",
            f"same-{index}.pdf",
            "core-concurrent",
        )
        receiving = store.begin_pdf_upload(acquisition.id, "core-concurrent")
        with files.open_staging(receiving.staging_relative_path) as stage:
            stage.write(content)
            stage.flush()
            os.fsync(stage.fileno())
        store.record_pdf_upload(
            acquisition.id,
            "core-concurrent",
            receiving.finalization_generation,
            checksum,
            len(content),
        )
        acquisitions.append(acquisition)
    ingest = PdfIngestService(store, "core-concurrent")

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda item: ingest.retry(item.id), acquisitions))

    assert [item.state for item in outcomes] == ["completed", "completed"]
    with database.connect() as connection:
        managed_ids = {
            row[0]
            for row in connection.execute(
                "SELECT managed_file_id FROM source_versions ORDER BY source_id"
            )
        }
        assert len(managed_ids) == 1
        assert connection.execute("SELECT count(*) FROM managed_files").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM original_registry").fetchone()[0] == 1


def test_import_racing_whole_source_expiry_keeps_a_shared_original(app_client) -> None:
    client, store, database = app_client
    content = minimal_pdf("source expiry race")
    notebook_a = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Expiring owner"}
    ).json()
    notebook_b = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Surviving owner"}
    ).json()
    source_a = acquire(client, notebook_a["id"], "expiry-owner", content)
    source_b = acquire(client, notebook_b["id"], "survivor", content)
    managed = store.load_import_managed_file(source_a["import_id"])
    expiry = (
        (datetime.now(UTC) - timedelta(days=31))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    store.trash_source(source_a["source_id"], expiry)

    new_notebook = store.create_notebook(new_id(), "Import during expiry")
    ingest: PdfIngestService = client.app.state.pdf_ingest
    acquisition = ingest.create_acquisition(
        "expiry-race-import", new_notebook.id, "add", "race.pdf"
    )
    receiving = store.begin_pdf_upload(acquisition.id, ingest.owner_instance_id)
    files = ManagedPdfFiles(database.config.data_dir)
    with files.open_staging(receiving.staging_relative_path) as stage:
        stage.write(content)
        stage.flush()
        os.fsync(stage.fileno())
    store.record_pdf_upload(
        acquisition.id,
        ingest.owner_instance_id,
        receiving.finalization_generation,
        hashlib.sha256(content).hexdigest(),
        len(content),
    )
    barrier = threading.Barrier(3)
    sweeper = TrashSweeper(store, fingerprint_locks=ingest.fingerprint_locks)

    def purge() -> tuple[str, ...]:
        barrier.wait()
        return sweeper.sweep_once()

    def import_pdf():
        barrier.wait()
        return ingest.retry(acquisition.id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        purge_future = executor.submit(purge)
        import_future = executor.submit(import_pdf)
        barrier.wait()
        purged = purge_future.result(timeout=10)
        imported = import_future.result(timeout=10)

    assert source_a["source_id"] in purged
    assert imported.state == "completed"
    assert imported.outcome_import_id is not None
    reused = store.load_import_managed_file(imported.outcome_import_id)
    assert reused.id == managed.id
    assert store.load_import_managed_file(source_b["import_id"]).id == managed.id
    assert (
        ManagedPdfFiles(database.config.data_dir).read_verified(
            reused.relative_path, reused.sha256, reused.size_bytes
        )
        == content
    )


def test_shared_pdf_survives_first_source_purge_and_retries_failed_last_unlink(
    app_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, store, database = app_client
    content = minimal_pdf("shared retention")
    notebook_a = client.post("/notebooks", headers=auth_headers(), json={"title": "Owner A"}).json()
    notebook_b = client.post("/notebooks", headers=auth_headers(), json={"title": "Owner B"}).json()
    import_a = acquire(client, notebook_a["id"], "owner-a", content)
    import_b = acquire(client, notebook_b["id"], "owner-b", content)
    managed = store.load_import_managed_file(import_a["import_id"])
    saved_path = database.config.data_dir / managed.relative_path
    locks: FingerprintLockCoordinator = client.app.state.pdf_ingest.fingerprint_locks
    expired = (
        (datetime.now(UTC) - timedelta(days=31))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )

    store.trash_source(import_a["source_id"], expired)
    assert TrashSweeper(store, fingerprint_locks=locks).sweep_once() == (import_a["source_id"],)
    assert saved_path.is_file()
    reopened_database = Database(DatabaseConfig(database.config.data_dir))
    reopened_database.initialize()
    store = PersistenceStore(reopened_database)
    reopened_ingest = PdfIngestService(store, "core-after-shared-file-restart")
    reopened_app = create_app(
        TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=reopened_ingest
    )
    with TestClient(reopened_app, base_url="http://127.0.0.1") as reopened_client:
        assert store.load_import_managed_file(import_b["import_id"]).id == managed.id
        still_open = reopened_client.get(
            f"/sources/{import_b['source_id']}/imports/{import_b['import_id']}/pdf",
            headers=auth_headers(),
        )
        assert still_open.status_code == 200
        assert still_open.content == content

    store.trash_source(import_b["source_id"], expired)
    original_unlink = Path.unlink
    failed_once = False

    def fail_original_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal failed_once
        if path == saved_path and not failed_once:
            failed_once = True
            raise PermissionError("injected PDF removal failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(type(saved_path), "unlink", fail_original_unlink)
    first_sweep = TrashSweeper(
        store, fingerprint_locks=reopened_ingest.fingerprint_locks
    ).sweep_once()
    assert first_sweep == ()
    assert saved_path.is_file()
    removal = store.load_managed_file_removal(managed.id)
    assert removal is not None and removal.removed_at is None and removal.attempts == 1

    monkeypatch.setattr(type(saved_path), "unlink", original_unlink)
    second_sweep = TrashSweeper(
        store, fingerprint_locks=reopened_ingest.fingerprint_locks
    ).sweep_once()
    assert second_sweep == (import_b["source_id"],)
    assert not saved_path.exists()


def test_corrupt_saved_original_is_not_reused_repaired_or_served(app_client) -> None:
    client, store, database = app_client
    notebook = client.post("/notebooks", headers=auth_headers(), json={"title": "Integrity"}).json()
    content = minimal_pdf("verify me")
    saved = acquire(client, notebook["id"], "integrity-first", content)
    managed = store.load_import_managed_file(saved["import_id"])
    path = database.config.data_dir / managed.relative_path
    path.write_bytes(b"damaged original")

    repeated = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={"retry_key": "integrity-repeat", "filename": "again.pdf"},
    ).json()
    reuse = client.put(
        f"/pdf-acquisitions/{repeated['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=content,
    )
    opened = client.get(
        f"/sources/{saved['source_id']}/imports/{saved['import_id']}/pdf",
        headers=auth_headers(),
    )

    assert reuse.status_code == 409
    assert reuse.json()["detail"]["code"] == "integrity_failure"
    assert opened.status_code == 409
    assert opened.json()["detail"]["code"] == "integrity_failure"
    assert path.read_bytes() == b"damaged original"
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM managed_files").fetchone()[0] == 1


def test_legacy_registration_hold_wins_against_expired_history_purge(
    app_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, store, database = app_client
    notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Old PDF history"}
    ).json()
    old_content = minimal_pdf("legacy original")
    earlier = acquire(client, notebook["id"], "legacy-earlier", old_content)
    replacement = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "legacy-newer",
            "filename": "newer.pdf",
            "intent": "replace",
            "target_source_id": earlier["source_id"],
        },
    ).json()
    newer = client.put(
        f"/pdf-acquisitions/{replacement['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=minimal_pdf("new current"),
    ).json()
    publish_pdf_import(store, earlier["source_id"], newer["import_id"])
    store.cancel_pdf_import_processing(earlier["import_id"])
    managed = store.load_import_managed_file(earlier["import_id"])
    future = (
        (datetime.now(UTC) + timedelta(days=30))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    store.trash_pdf_import(earlier["import_id"], future)
    with database.connect() as connection:
        connection.execute("DELETE FROM original_registry WHERE managed_file_id=?", (managed.id,))
    other_notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "New notebook"}
    ).json()
    ingest: PdfIngestService = client.app.state.pdf_ingest
    real_read = ingest.files.read_verified
    reservation_results: list[object] = []

    def expire_and_try_reservation(path: str, checksum: str, size_bytes: int) -> bytes:
        if path == managed.relative_path and not reservation_results:
            with database.connect() as connection:
                connection.execute(
                    "UPDATE pdf_import_trash SET purge_deadline_at=? WHERE import_id=?",
                    ("2000-01-01T00:00:00.000000Z", earlier["import_id"]),
                )
            reservation_results.append(
                store.reserve_pdf_import_file_removal(
                    managed.id, managed.relative_path, earlier["import_id"]
                )
            )
        return real_read(path, checksum, size_bytes)

    monkeypatch.setattr(ingest.files, "read_verified", expire_and_try_reservation)
    newly_added = acquire(client, other_notebook["id"], "legacy-register", old_content)
    assert reservation_results == [None]
    assert newly_added["outcome_kind"] == "created"
    assert store.load_import_managed_file(newly_added["import_id"]).id == managed.id

    ingest.purge_expired_pdf_history()

    with pytest.raises(PersistenceError):
        store.load_pdf_import(earlier["import_id"])
    reused = store.load_import_managed_file(newly_added["import_id"])
    assert (
        ManagedPdfFiles(database.config.data_dir).read_verified(
            reused.relative_path, reused.sha256, reused.size_bytes
        )
        == old_content
    )


def test_legacy_registration_tries_older_files_until_one_verifies(app_client) -> None:
    client, store, database = app_client
    notebook = store.create_notebook(new_id(), "Legacy files")
    source_id = new_id()
    store.create_source(source_id, notebook.id, "legacy-source")
    content = minimal_pdf("canonical legacy content")
    checksum = hashlib.sha256(content).hexdigest()
    older = ManagedFile(
        new_id(),
        "files/objects/legacy-older.pdf",
        len(content),
        checksum,
        "2020-01-01T00:00:00.000000Z",
    )
    newer = ManagedFile(
        new_id(),
        "files/objects/legacy-newer.pdf",
        len(content),
        checksum,
        "2021-01-01T00:00:00.000000Z",
    )
    store.save_managed_file(older)
    store.save_managed_file(newer)
    store.create_source_version(source_id, checksum, older.id)
    store.create_source_version(source_id, checksum, newer.id)
    files = ManagedPdfFiles(database.config.data_dir)
    bad_path = database.config.data_dir / older.relative_path
    good_path = database.config.data_dir / newer.relative_path
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"x" * len(content))
    good_path.write_bytes(content)

    imported = acquire(client, notebook.id, "register-legacy", content)
    linked = store.load_import_managed_file(imported["import_id"])

    assert linked.id == newer.id
    assert bad_path.read_bytes() == b"x" * len(content)
    assert files.read_verified(newer.relative_path, checksum, len(content)) == content
    with database.connect() as connection:
        canonical = connection.execute("SELECT managed_file_id FROM original_registry").fetchone()
        assert canonical[0] == newer.id


def test_changed_replace_stays_pending_until_processing_publication(app_client) -> None:
    client, store, _ = app_client
    notebook = client.post("/notebooks", headers=auth_headers(), json={"title": "Versions"}).json()
    first = acquire(client, notebook["id"], "version-one", minimal_pdf("first"))
    publish_pdf_import(store, first["source_id"], first["import_id"])
    original_readiness = store.load_readiness(first["source_id"])
    assert original_readiness is not None
    replacement = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "version-two",
            "filename": "updated.pdf",
            "intent": "replace",
            "target_source_id": first["source_id"],
        },
    ).json()
    second = client.put(
        f"/pdf-acquisitions/{replacement['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=minimal_pdf("second"),
    )

    assert second.status_code == 200, second.text
    state = store.load_source_pdf_state(first["source_id"])
    assert state is not None
    assert state.current_import_id == first["import_id"]
    assert state.pending_import_id == second.json()["import_id"]
    assert store.load_readiness(first["source_id"]) == original_readiness
    opened = client.get(
        f"/sources/{first['source_id']}/imports/{first['import_id']}/pdf",
        headers=auth_headers(),
    )
    assert opened.content == minimal_pdf("first")

    publish_pdf_import(store, first["source_id"], second.json()["import_id"])
    state = store.load_source_pdf_state(first["source_id"])
    assert state is not None
    assert state.current_import_id == second.json()["import_id"]
    assert state.pending_import_id is None
    new_readiness = store.load_readiness(first["source_id"])
    assert new_readiness is not None
    assert (
        new_readiness.source_version_id
        == store.load_pdf_import(second.json()["import_id"]).source_version_id
    )


def test_add_reports_earlier_history_and_explicit_replace_reuses_that_import(app_client) -> None:
    client, store, database = app_client
    notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Earlier versions"}
    ).json()
    earlier_bytes = minimal_pdf("keep this older version")
    first = acquire(client, notebook["id"], "history-v1", earlier_bytes)
    current_bytes = minimal_pdf("current version")
    replacement = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "history-v2",
            "filename": "current.pdf",
            "intent": "replace",
            "target_source_id": first["source_id"],
        },
    ).json()
    current = client.put(
        f"/pdf-acquisitions/{replacement['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=current_bytes,
    ).json()
    publish_pdf_import(store, first["source_id"], current["import_id"])
    earlier_file = store.load_import_managed_file(first["import_id"])

    duplicate_action = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={"retry_key": "history-add", "filename": "again.pdf"},
    ).json()
    duplicate = client.put(
        f"/pdf-acquisitions/{duplicate_action['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=earlier_bytes,
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "pdf_in_history"

    future = (
        (datetime.now(UTC) + timedelta(days=30))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    store.cancel_pdf_import_processing(first["import_id"])
    store.trash_pdf_import(first["import_id"], future)
    trashed_action = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={"retry_key": "history-add-trash", "filename": "again.pdf"},
    ).json()
    trashed = client.put(
        f"/pdf-acquisitions/{trashed_action['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=earlier_bytes,
    )
    assert trashed.status_code == 409
    assert trashed.json()["detail"]["code"] == "pdf_in_history"
    assert "trash" in trashed.json()["detail"]["message"]

    explicit_action = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "explicit-history-replace",
            "filename": "restore-as-new-version.pdf",
            "intent": "replace",
            "target_source_id": first["source_id"],
        },
    ).json()
    explicit = client.put(
        f"/pdf-acquisitions/{explicit_action['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=earlier_bytes,
    )
    assert explicit.status_code == 200, explicit.text
    explicit_import = explicit.json()["import_id"]
    assert store.load_import_managed_file(explicit_import).id == earlier_file.id
    state = store.load_source_pdf_state(first["source_id"])
    assert state is not None and state.current_import_id == current["import_id"]
    assert state.pending_import_id == explicit_import
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 1


def test_cancelled_and_superseded_replacements_keep_the_previous_ready_pdf(app_client) -> None:
    client, store, _ = app_client
    notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Replacement races"}
    ).json()
    original = acquire(client, notebook["id"], "ready-first", minimal_pdf("ready original"))
    publish_pdf_import(store, original["source_id"], original["import_id"])
    original_readiness = store.load_readiness(original["source_id"])
    assert original_readiness is not None

    first_replacement = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "replace-a",
            "filename": "a.pdf",
            "intent": "replace",
            "target_source_id": original["source_id"],
        },
    ).json()
    first_response = client.put(
        f"/pdf-acquisitions/{first_replacement['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=minimal_pdf("candidate A"),
    )
    assert first_response.status_code == 200, first_response.text
    candidate_a = first_response.json()
    cancelled = client.post(
        f"/sources/{original['source_id']}/imports/{candidate_a['import_id']}/cancel",
        headers=auth_headers(),
    )
    assert cancelled.status_code == 200, cancelled.text
    assert store.load_readiness(original["source_id"]) == original_readiness
    state = store.load_source_pdf_state(original["source_id"])
    assert state is not None and state.current_import_id == original["import_id"]
    assert state.pending_import_id is None

    second_replacement = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "replace-b",
            "filename": "b.pdf",
            "intent": "replace",
            "target_source_id": original["source_id"],
        },
    ).json()
    second_response = client.put(
        f"/pdf-acquisitions/{second_replacement['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=minimal_pdf("candidate B"),
    )
    assert second_response.status_code == 200, second_response.text
    candidate_b = second_response.json()
    with store.database.connect() as connection:
        second_job_id = connection.execute(
            """SELECT id FROM jobs WHERE source_version_id=(
                   SELECT source_version_id FROM pdf_imports WHERE id=?
               )""",
            (candidate_b["import_id"],),
        ).fetchone()[0]
    third_replacement = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "replace-c",
            "filename": "c.pdf",
            "intent": "replace",
            "target_source_id": original["source_id"],
        },
    ).json()
    third_response = client.put(
        f"/pdf-acquisitions/{third_replacement['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=minimal_pdf("candidate C"),
    )
    assert third_response.status_code == 200, third_response.text
    candidate_c = third_response.json()
    assert store.load_pdf_import(candidate_b["import_id"]).state == "failed"
    second_job_status = store.load_job_status(second_job_id)
    assert second_job_status is not None
    assert second_job_status.status == "cancelled"
    assert second_job_status.failure_reason == "superseded by newer PDF replacement"
    state = store.load_source_pdf_state(original["source_id"])
    assert state is not None and state.current_import_id == original["import_id"]
    assert state.pending_import_id == candidate_c["import_id"]
    assert store.load_readiness(original["source_id"]) == original_readiness


def test_import_blocked_by_failed_removal_retries_saved_input_after_cleanup(
    app_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, store, database = app_client
    old_notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Expiring PDF"}
    ).json()
    new_notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Retry after cleanup"}
    ).json()
    content = minimal_pdf("same PDF")
    first = acquire(client, old_notebook["id"], "cleanup-first", content)
    managed = store.load_import_managed_file(first["import_id"])
    saved_path = database.config.data_dir / managed.relative_path
    expired = (
        (datetime.now(UTC) - timedelta(days=31))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    store.trash_source(first["source_id"], expired)
    original_unlink = Path.unlink
    failed_once = False

    def fail_once(path: Path, missing_ok: bool = False) -> None:
        nonlocal failed_once
        if path == saved_path and not failed_once:
            failed_once = True
            raise PermissionError("injected removal failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(type(saved_path), "unlink", fail_once)
    ingest: PdfIngestService = client.app.state.pdf_ingest
    sweeper = TrashSweeper(store, fingerprint_locks=ingest.fingerprint_locks)
    assert sweeper.sweep_once() == ()
    removal = store.load_managed_file_removal(managed.id)
    assert removal is not None and removal.removed_at is None and removal.attempts == 1

    acquisition = client.post(
        f"/notebooks/{new_notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={"retry_key": "cleanup-second", "filename": "copy.pdf"},
    ).json()
    lock = ingest.fingerprint_locks.lock_for(managed.sha256, managed.size_bytes)
    assert lock.acquire(timeout=1)
    started = time.monotonic()
    try:
        blocked = client.put(
            f"/pdf-acquisitions/{acquisition['id']}/content",
            headers={**auth_headers(), "Content-Type": "application/pdf"},
            content=content,
        )
    finally:
        lock.release()
    elapsed = time.monotonic() - started

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "file_cleanup_pending"
    assert elapsed < 2
    current = store.load_pdf_acquisition(acquisition["id"])
    assert current.state == "waiting_cleanup"
    assert current.input_complete_at is not None
    assert (database.config.data_dir / current.staging_relative_path).is_file()
    report = client.get("/storage/originals", headers=auth_headers()).json()
    assert report["bytes_pending_removal"] == managed.size_bytes
    assert report["retained_staging_bytes"] == len(content)
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1

    ingest.reconcile_startup()
    assert (database.config.data_dir / current.staging_relative_path).is_file()
    reopened = create_app(
        TOKEN,
        allowed_origins=(ORIGIN,),
        store=store,
        pdf_ingest=PdfIngestService(store, "core-reopened"),
    )
    with TestClient(reopened, base_url="http://127.0.0.1") as reopened_client:
        restored = reopened_client.get(
            f"/notebooks/{new_notebook['id']}/pdf-acquisitions", headers=auth_headers()
        ).json()
        assert len(restored) == 1
        assert restored[0]["id"] == current.id
        assert restored[0]["state"] == "waiting_cleanup"
        assert restored[0]["can_retry"] is True
        assert restored[0]["can_cancel"] is True
        assert restored[0]["requires_upload"] is False
    monkeypatch.setattr(type(saved_path), "unlink", original_unlink)
    assert sweeper.sweep_once() == (first["source_id"],)

    retried = ingest.retry(current.id)
    assert retried.state == "completed"
    assert retried.outcome_source_id is not None
    assert retried.outcome_import_id is not None
    replacement_file = store.load_import_managed_file(retried.outcome_import_id)
    assert replacement_file.id != managed.id
    assert (
        ManagedPdfFiles(database.config.data_dir).read_verified(
            replacement_file.relative_path, replacement_file.sha256, replacement_file.size_bytes
        )
        == content
    )


def test_stream_enforces_actual_upload_bytes_and_keeps_partial_input_unsaved(
    tmp_path: Path,
) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    ingest = PdfIngestService(
        store,
        "core-test",
        limits=PdfIngestLimits(max_upload_bytes=32),
    )
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=ingest)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        notebook = client.post(
            "/notebooks", headers=auth_headers(), json={"title": "Limits"}
        ).json()
        acquisition = client.post(
            f"/notebooks/{notebook['id']}/pdf-acquisitions",
            headers=auth_headers(),
            json={"retry_key": "too-large", "filename": "large.pdf"},
        ).json()
        response = client.put(
            f"/pdf-acquisitions/{acquisition['id']}/content",
            headers={**auth_headers(), "Content-Type": "application/pdf"},
            content=minimal_pdf("beyond limit"),
        )
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "pdf_too_large"
        current = store.load_pdf_acquisition(acquisition["id"])
        assert current.state == "failed"
        assert current.outcome_source_id is None
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 0


def test_pdf_at_the_configured_size_limit_is_saved_with_its_exact_bytes(app_client) -> None:
    client, store, database = app_client
    notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "Near upload limit"}
    ).json()
    base = minimal_pdf("near limit")
    content = base + b" " * (25_000_000 - len(base))
    saved = acquire(client, notebook["id"], "near-limit", content)
    managed = store.load_import_managed_file(saved["import_id"])
    assert managed.size_bytes == 25_000_000
    assert (
        ManagedPdfFiles(database.config.data_dir).read_verified(
            managed.relative_path, managed.sha256, managed.size_bytes
        )
        == content
    )


def test_pdf_disk_write_failure_is_reported_without_creating_a_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    ingest = PdfIngestService(store, "core-disk-failure")

    def fail_open(_relative_path: str):
        raise PermissionError("injected write failure")

    monkeypatch.setattr(ingest.files, "open_staging", fail_open)
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=ingest)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        notebook = client.post(
            "/notebooks", headers=auth_headers(), json={"title": "Disk failure"}
        ).json()
        acquisition = client.post(
            f"/notebooks/{notebook['id']}/pdf-acquisitions",
            headers=auth_headers(),
            json={"retry_key": "disk-failure", "filename": "disk.pdf"},
        ).json()
        response = client.put(
            f"/pdf-acquisitions/{acquisition['id']}/content",
            headers={**auth_headers(), "Content-Type": "application/pdf"},
            content=minimal_pdf("write failure"),
        )
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "pdf_storage_failed"
        assert store.load_pdf_acquisition(acquisition["id"]).state == "failed"
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 0
    assert "PermissionError" in caplog.text
    assert "injected write failure" not in caplog.text
    assert TOKEN not in caplog.text
    assert str(tmp_path) not in caplog.text


@pytest.mark.anyio
async def test_total_upload_deadline_busy_slot_and_health_responsiveness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from core.ingest import service as ingest_module

    # Advance only the upload clock. Real SQLite/file scheduling must not decide
    # whether the test's 80 ms idle or 250 ms total deadline expired first.
    clock = SimpleNamespace(now=0.0)
    stream_timeouts = []

    class UploadScheduler:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        async def wait_for(self, awaitable, timeout):
            started = clock.now
            is_stream = not asyncio.isfuture(awaitable)
            if is_stream:
                stream_timeouts.append(timeout)
            result = await awaitable
            if clock.now - started >= timeout:
                clock.now = started + timeout
                raise TimeoutError
            return result

    monkeypatch.setattr(ingest_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(ingest_module, "asyncio", UploadScheduler())
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    ingest = PdfIngestService(
        store,
        "core-test",
        limits=PdfIngestLimits(upload_total_seconds=0.25, upload_idle_seconds=0.08),
    )
    notebook = store.create_notebook(new_id(), "Slow uploads")
    first = ingest.create_acquisition("slow-first", notebook.id, "add", "slow.pdf")
    second = ingest.create_acquisition("slow-second", notebook.id, "add", "second.pdf")
    idle = ingest.create_acquisition("idle-action", notebook.id, "add", "idle.pdf")
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=ingest)
    transport = httpx.ASGITransport(app=app)

    body_started = asyncio.Event()
    release_body = asyncio.Event()

    async def slow_body():
        body_started.set()
        yield b"%PDF-"
        await release_body.wait()
        for _ in range(12):
            clock.now += 0.03
            yield b"x" * 12

    async def idle_body():
        yield b"%PDF-"
        clock.now += 0.12
        yield b"remaining bytes"

    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        first_request = asyncio.create_task(
            client.put(
                f"/pdf-acquisitions/{first.id}/content",
                headers={**auth_headers(), "Content-Type": "application/pdf"},
                content=slow_body(),
            )
        )
        await body_started.wait()
        health = await client.get("/health", headers=auth_headers())
        busy = await client.put(
            f"/pdf-acquisitions/{second.id}/content",
            headers={**auth_headers(), "Content-Type": "application/pdf"},
            content=minimal_pdf("second"),
        )
        release_body.set()
        timed_out = await first_request
        idle_timeout = await client.put(
            f"/pdf-acquisitions/{idle.id}/content",
            headers={**auth_headers(), "Content-Type": "application/pdf"},
            content=idle_body(),
        )

    await ingest.stop()
    assert health.status_code == 200
    assert busy.status_code == 409
    assert busy.json()["detail"]["code"] == "import_busy"
    assert timed_out.status_code == 408
    assert timed_out.json()["detail"]["code"] == "upload_timeout"
    assert idle_timeout.status_code == 408
    assert idle_timeout.json()["detail"]["code"] == "upload_idle_timeout"
    assert 0 < min(stream_timeouts) < ingest.limits.upload_idle_seconds
    assert max(stream_timeouts) == ingest.limits.upload_idle_seconds
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 0


def test_finalization_timeout_invalidates_the_late_writer_and_keeps_recovery_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    ingest = PdfIngestService(
        store,
        "core-timeout",
        limits=PdfIngestLimits(upload_total_seconds=0.35, upload_idle_seconds=0.2),
    )
    notebook = store.create_notebook(new_id(), "Finalization timeout")
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=ingest)
    entered = threading.Event()
    release = threading.Event()
    worker_returned = threading.Event()
    original_finish = store.finish_pdf_acquisition

    def pause_finalization(*args, **kwargs):
        entered.set()
        release.wait(5)
        try:
            return original_finish(*args, **kwargs)
        finally:
            worker_returned.set()

    monkeypatch.setattr(store, "finish_pdf_acquisition", pause_finalization)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        acquisition = client.post(
            f"/notebooks/{notebook.id}/pdf-acquisitions",
            headers=auth_headers(),
            json={"retry_key": "final-timeout", "filename": "timeout.pdf"},
        ).json()
        response = client.put(
            f"/pdf-acquisitions/{acquisition['id']}/content",
            headers={**auth_headers(), "Content-Type": "application/pdf"},
            content=minimal_pdf("late finalizer"),
        )
        assert entered.is_set()
        assert response.status_code == 408
        assert response.json()["detail"]["code"] == "upload_timeout"
        staged = store.load_pdf_acquisition(acquisition["id"])
        assert staged.state == "staged"
        assert staged.input_complete_at is not None
        assert staged.outcome_source_id is None
        release.set()
        assert worker_returned.wait(5)

        monkeypatch.setattr(store, "finish_pdf_acquisition", original_finish)
        recovered = ingest.retry(acquisition["id"])
        assert recovered.state == "completed"
        assert recovered.outcome_source_id is not None
        assert recovered.outcome_import_id is not None
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 1


def test_cancelling_a_placed_prejob_acquisition_removes_only_its_journaled_files(
    tmp_path: Path,
) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    service = PdfIngestService(store, "core-cancel-placement")
    notebook = store.create_notebook(new_id(), "Cancelled placement")
    acquisition = service.create_acquisition(
        "cancel-placement", notebook.id, "add", "cancelled.pdf"
    )
    receiving = store.begin_pdf_upload(acquisition.id, "core-cancel-placement")
    content = minimal_pdf("cancel after placement")
    checksum = hashlib.sha256(content).hexdigest()
    files = ManagedPdfFiles(database.config.data_dir)
    with files.open_staging(receiving.staging_relative_path) as stage:
        stage.write(content)
        stage.flush()
        os.fsync(stage.fileno())
    store.record_pdf_upload(
        acquisition.id,
        "core-cancel-placement",
        receiving.finalization_generation,
        checksum,
        len(content),
    )
    finalizing = store.claim_pdf_finalization(acquisition.id, "core-cancel-placement")
    managed_id = new_id()
    relative_path = f"files/objects/{managed_id}.pdf"
    store.set_pdf_placement_plan(
        acquisition.id,
        "core-cancel-placement",
        finalizing.finalization_generation,
        managed_id,
        relative_path,
        checksum,
        len(content),
    )
    files.place_staging(receiving.staging_relative_path, relative_path)
    store.mark_pdf_placement_complete(
        acquisition.id, "core-cancel-placement", finalizing.finalization_generation
    )

    cancelled = service.cancel(acquisition.id)

    assert cancelled.state == "cancelled"
    assert not (database.config.data_dir / relative_path).exists()
    assert not (database.config.data_dir / receiving.staging_relative_path).exists()
    assert store.list_pending_acquisition_file_cleanup() == ()
    with database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM managed_files").fetchone()[0] == 0


def test_pdf_routes_require_existing_bearer_and_origin_checks(app_client) -> None:
    client, _, _ = app_client
    assert client.get("/notebooks").status_code == 401
    response = client.options(
        "/notebooks",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert "POST" in response.headers["access-control-allow-methods"]


def test_startup_recovers_a_complete_staged_upload_without_rewriting_legacy_files(
    tmp_path: Path,
) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    notebook = store.create_notebook("0f5098ac-4a1b-47a5-9c20-c2ab1dbfcb3c", "Crash recovery")
    first_core = PdfIngestService(store, "core-before-crash")
    acquisition = first_core.create_acquisition("recover-action", notebook.id, "add", "recover.pdf")
    receiving = store.begin_pdf_upload(acquisition.id, "core-before-crash")
    content = minimal_pdf("recover this saved input")
    checksum = hashlib.sha256(content).hexdigest()
    stage = ManagedPdfFiles(database.config.data_dir).open_staging(receiving.staging_relative_path)
    stage.write(content)
    stage.flush()
    os.fsync(stage.fileno())
    stage.close()
    store.record_pdf_upload(
        acquisition.id,
        "core-before-crash",
        receiving.finalization_generation,
        checksum,
        len(content),
    )

    recovered_core = PdfIngestService(store, "core-after-restart")
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=recovered_core)
    with TestClient(app, base_url="http://127.0.0.1"):
        recovered = store.load_pdf_acquisition(acquisition.id)
        assert recovered.state == "completed"
        assert recovered.outcome_source_id is not None
        assert recovered.outcome_import_id is not None
        managed = store.load_import_managed_file(recovered.outcome_import_id)
        assert (
            ManagedPdfFiles(database.config.data_dir).read_verified(
                managed.relative_path, managed.sha256, managed.size_bytes
            )
            == content
        )
        assert not (database.config.data_dir / receiving.staging_relative_path).exists()

    reopened_database = Database(DatabaseConfig(database.config.data_dir))
    reopened_database.initialize()
    reopened_store = PersistenceStore(reopened_database)
    reopened_service = PdfIngestService(reopened_store, "core-after-app-reopen")
    reopened_app = create_app(
        TOKEN,
        allowed_origins=(ORIGIN,),
        store=reopened_store,
        pdf_ingest=reopened_service,
    )
    with TestClient(reopened_app, base_url="http://127.0.0.1") as reopened_client:
        notebooks = reopened_client.get("/notebooks", headers=auth_headers())
        assert [item["id"] for item in notebooks.json()] == [notebook.id]
        sources = reopened_client.get(f"/notebooks/{notebook.id}/sources", headers=auth_headers())
        assert len(sources.json()) == 1
        reopened_pdf = reopened_client.get(
            f"/sources/{recovered.outcome_source_id}/imports/{recovered.outcome_import_id}/pdf",
            headers=auth_headers(),
        )
        assert reopened_pdf.status_code == 200
        assert reopened_pdf.content == content


def test_startup_discards_only_the_incomplete_upload_after_a_core_crash(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Interrupted transfer")
    first_core = PdfIngestService(store, "core-before-partial-crash")
    acquisition = first_core.create_acquisition("partial-crash", notebook.id, "add", "partial.pdf")
    receiving = store.begin_pdf_upload(acquisition.id, "core-before-partial-crash")
    stage = ManagedPdfFiles(database.config.data_dir).open_staging(receiving.staging_relative_path)
    stage.write(b"%PDF-1.4\npartial only")
    stage.close()

    restarted = PdfIngestService(store, "core-after-partial-crash")
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=restarted)
    with TestClient(app, base_url="http://127.0.0.1"):
        current = store.load_pdf_acquisition(acquisition.id)
        assert current.state == "failed"
        assert current.failure_code == "upload_interrupted"
        assert current.input_complete_at is None
        assert not (database.config.data_dir / receiving.staging_relative_path).exists()
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 0


def test_startup_recovers_a_pdf_already_placed_at_its_durable_plan(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Placed recovery")
    first_core = PdfIngestService(store, "core-before-placement-crash")
    acquisition = first_core.create_acquisition("placed-recovery", notebook.id, "add", "placed.pdf")
    receiving = store.begin_pdf_upload(acquisition.id, "core-before-placement-crash")
    content = minimal_pdf("already placed")
    checksum = hashlib.sha256(content).hexdigest()
    files = ManagedPdfFiles(database.config.data_dir)
    with files.open_staging(receiving.staging_relative_path) as stage:
        stage.write(content)
        stage.flush()
        os.fsync(stage.fileno())
    store.record_pdf_upload(
        acquisition.id,
        "core-before-placement-crash",
        receiving.finalization_generation,
        checksum,
        len(content),
    )
    finalizing = store.claim_pdf_finalization(acquisition.id, "core-before-placement-crash")
    managed_file_id = new_id()
    relative_path = f"files/objects/{managed_file_id}.pdf"
    store.set_pdf_placement_plan(
        acquisition.id,
        "core-before-placement-crash",
        finalizing.finalization_generation,
        managed_file_id,
        relative_path,
        checksum,
        len(content),
    )
    files.place_staging(receiving.staging_relative_path, relative_path)
    store.mark_pdf_placement_complete(
        acquisition.id, "core-before-placement-crash", finalizing.finalization_generation
    )

    recovered = PdfIngestService(store, "core-after-placement-crash")
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=recovered)
    with TestClient(app, base_url="http://127.0.0.1"):
        outcome = store.load_pdf_acquisition(acquisition.id)
        assert outcome.state == "completed"
        assert outcome.planned_managed_file_id == managed_file_id
        assert outcome.outcome_import_id is not None
        managed = store.load_import_managed_file(outcome.outcome_import_id)
        assert managed.id == managed_file_id
        assert files.read_verified(relative_path, checksum, len(content)) == content
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM managed_files").fetchone()[0] == 1
        assert not (database.config.data_dir / receiving.staging_relative_path).exists()


def test_startup_reverifies_a_reuse_hold_and_releases_it_with_the_import(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    original_bytes = minimal_pdf("existing legacy original")
    checksum = hashlib.sha256(original_bytes).hexdigest()
    owner_notebook = store.create_notebook(new_id(), "Existing owner")
    owner_source = new_id()
    store.create_source(owner_source, owner_notebook.id, "legacy.pdf")
    managed_id = new_id()
    relative_path = f"files/objects/{managed_id}.pdf"
    managed = ManagedFile(managed_id, relative_path, len(original_bytes), checksum, utc_timestamp())
    store.save_managed_file(managed)
    original_path = database.config.data_dir / relative_path
    original_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.write_bytes(original_bytes)
    store.create_source_version(owner_source, checksum, managed_id)
    new_notebook = store.create_notebook(new_id(), "New owner")

    first_core = PdfIngestService(store, "core-before-reuse-crash")
    acquisition = first_core.create_acquisition("reuse-crash", new_notebook.id, "add", "legacy.pdf")
    receiving = store.begin_pdf_upload(acquisition.id, "core-before-reuse-crash")
    stage = ManagedPdfFiles(database.config.data_dir).open_staging(receiving.staging_relative_path)
    stage.write(original_bytes)
    stage.flush()
    os.fsync(stage.fileno())
    stage.close()
    store.record_pdf_upload(
        acquisition.id,
        "core-before-reuse-crash",
        receiving.finalization_generation,
        checksum,
        len(original_bytes),
    )
    finalizing = store.claim_pdf_finalization(acquisition.id, "core-before-reuse-crash")
    store.set_pdf_reuse_hold(
        acquisition.id, "core-before-reuse-crash", finalizing.finalization_generation, managed_id
    )

    restarted_store = PersistenceStore(database)
    assert restarted_store.has_live_pdf_reuse_hold(managed_id)
    assert (
        restarted_store.reserve_managed_file_removal(managed_id, relative_path, owner_source)
        is None
    )
    restarted_service = PdfIngestService(restarted_store, "core-after-reuse-crash")
    app = create_app(
        TOKEN, allowed_origins=(ORIGIN,), store=restarted_store, pdf_ingest=restarted_service
    )
    with TestClient(app, base_url="http://127.0.0.1"):
        result = restarted_store.load_pdf_acquisition(acquisition.id)
        assert result.state == "completed"
        assert result.outcome_source_id is not None
        assert result.outcome_import_id is not None
        assert result.planned_reuse_managed_file_id is None
        assert not restarted_store.has_live_pdf_reuse_hold(managed_id)
        reused = restarted_store.load_import_managed_file(result.outcome_import_id)
        assert reused.id == managed_id
        assert (
            ManagedPdfFiles(database.config.data_dir).read_verified(
                reused.relative_path, reused.sha256, reused.size_bytes
            )
            == original_bytes
        )


def test_conflicting_planned_destination_is_preserved_for_review(tmp_path: Path) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Conflicting placement")
    service = PdfIngestService(store, "core-before-placement-check")
    acquisition = service.create_acquisition("conflict-plan", notebook.id, "add", "conflict.pdf")
    receiving = store.begin_pdf_upload(acquisition.id, "core-before-placement-check")
    expected = minimal_pdf("expected bytes")
    checksum = hashlib.sha256(expected).hexdigest()
    files = ManagedPdfFiles(database.config.data_dir)
    with files.open_staging(receiving.staging_relative_path) as stage:
        stage.write(expected)
        stage.flush()
        os.fsync(stage.fileno())
    store.record_pdf_upload(
        acquisition.id,
        "core-before-placement-check",
        receiving.finalization_generation,
        checksum,
        len(expected),
    )
    finalizing = store.claim_pdf_finalization(acquisition.id, "core-before-placement-check")
    managed_file_id = new_id()
    relative_path = f"files/objects/{managed_file_id}.pdf"
    store.set_pdf_placement_plan(
        acquisition.id,
        "core-before-placement-check",
        finalizing.finalization_generation,
        managed_file_id,
        relative_path,
        checksum,
        len(expected),
    )
    conflict = b"unrelated bytes at the planned destination"
    path = database.config.data_dir / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(conflict)

    recovered = PdfIngestService(store, "core-after-placement-check")
    app = create_app(TOKEN, allowed_origins=(ORIGIN,), store=store, pdf_ingest=recovered)
    with TestClient(app, base_url="http://127.0.0.1"):
        assert path.read_bytes() == conflict
        assert store.load_pdf_acquisition(acquisition.id).state == "failed"
        with database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 0
            assert connection.execute("SELECT count(*) FROM managed_files").fetchone()[0] == 0


def test_old_unowned_staging_is_reconciled_but_recent_or_recoverable_input_is_kept(
    tmp_path: Path,
) -> None:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    store = PersistenceStore(database)
    notebook = store.create_notebook(new_id(), "Staging cleanup")
    service = PdfIngestService(store, "core-test")

    old = service.create_acquisition("old-partial", notebook.id, "add", "partial.pdf")
    old_upload = store.begin_pdf_upload(old.id, "core-test")
    old_path = ManagedPdfFiles(database.config.data_dir).open_staging(old.staging_relative_path)
    old_path.write(b"partial bytes")
    old_path.close()
    store.fail_pdf_acquisition(
        old.id, "core-test", old_upload.finalization_generation, "upload_interrupted"
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE pdf_acquisitions SET last_write_at=? WHERE id=?",
            ("2000-01-01T00:00:00.000000Z", old.id),
        )

    recoverable = service.create_acquisition("complete-input", notebook.id, "add", "complete.pdf")
    receiving = store.begin_pdf_upload(recoverable.id, "core-test")
    complete_content = minimal_pdf("keep me for retry")
    digest = hashlib.sha256(complete_content).hexdigest()
    complete_path = ManagedPdfFiles(database.config.data_dir).open_staging(
        recoverable.staging_relative_path
    )
    complete_path.write(complete_content)
    complete_path.flush()
    os.fsync(complete_path.fileno())
    complete_path.close()
    store.record_pdf_upload(
        recoverable.id,
        "core-test",
        receiving.finalization_generation,
        digest,
        len(complete_content),
    )

    service.maintenance()

    assert not (database.config.data_dir / old.staging_relative_path).exists()
    completed = store.load_pdf_acquisition(recoverable.id)
    assert completed.state == "completed"
    assert completed.outcome_import_id is not None
    managed = store.load_import_managed_file(completed.outcome_import_id)
    assert (
        ManagedPdfFiles(database.config.data_dir).read_verified(
            managed.relative_path, managed.sha256, managed.size_bytes
        )
        == complete_content
    )


def test_expired_earlier_import_is_purged_without_removing_current_pdf(app_client) -> None:
    client, store, database = app_client
    notebook = client.post(
        "/notebooks", headers=auth_headers(), json={"title": "History purge"}
    ).json()
    original = acquire(client, notebook["id"], "history-first", minimal_pdf("earlier"))
    replacement_action = client.post(
        f"/notebooks/{notebook['id']}/pdf-acquisitions",
        headers=auth_headers(),
        json={
            "retry_key": "history-second",
            "filename": "newer.pdf",
            "intent": "replace",
            "target_source_id": original["source_id"],
        },
    ).json()
    replacement_response = client.put(
        f"/pdf-acquisitions/{replacement_action['id']}/content",
        headers={**auth_headers(), "Content-Type": "application/pdf"},
        content=minimal_pdf("current"),
    )
    assert replacement_response.status_code == 200, replacement_response.text
    current = replacement_response.json()
    store.cancel_pdf_import_processing(original["import_id"])
    with database.connect() as connection:
        connection.execute(
            """UPDATE source_pdf_state SET current_import_id=?,pending_import_id=NULL
               WHERE source_id=?""",
            (current["import_id"], original["source_id"]),
        )
        connection.execute(
            "UPDATE pdf_imports SET state='ready' WHERE id=?", (current["import_id"],)
        )
    store.trash_pdf_import(original["import_id"], "2000-01-01T00:00:00.000000Z")

    PdfIngestService(store, "history-cleaner").purge_expired_pdf_history()

    with pytest.raises(PersistenceError):
        store.load_pdf_import(original["import_id"])
    still_open = client.get(
        f"/sources/{current['source_id']}/imports/{current['import_id']}/pdf",
        headers=auth_headers(),
    )
    assert still_open.status_code == 200
    assert still_open.content == minimal_pdf("current")
