"""Earlier-import expiry with real PDF files, native search and on-disk recovery.

Vectors are controlled unit vectors: these are lifecycle checks, not model-quality tests.
"""

import asyncio
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import native_text_pdf
from test_search_expiry import reopen

from core.api.app import create_app
from core.ingest.service import PdfIngestService
from core.jobs.runner import StepCommitContext, StepContext
from core.jobs.steps import build_pdf_extract_steps
from core.jobs.sweeper import TrashSweeper
from core.persistence.errors import PersistenceError
from core.persistence.models import AttemptGuard, PdfImport, new_id, utc_timestamp
from core.persistence.store import PersistenceStore
from core.search.chunking import make_chunks
from core.search.install import ModelSetupService
from core.search.profiles import installation_path, profiles, verify_installation
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage
from core.search.supervisor import stop_owned_process_tree, worker_environment

TOKEN = "history-expiry-test-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
EXPIRED = "2000-01-01T00:00:00.000000Z"
FUTURE = "9999-01-01T00:00:00.000000Z"
VECTOR = vector_bytes([1.0] + [0.0] * 383, 384)


@dataclass(frozen=True)
class IndexedImport:
    item: PdfImport
    index: str
    extraction: str
    row_id: int
    original: Path
    content: bytes


@dataclass(frozen=True)
class History:
    old: IndexedImport
    current: IndexedImport
    neighbour: IndexedImport
    staging: Path
    snapshot: Path


def publish_import(store: PersistenceStore, import_id: str, content: bytes) -> IndexedImport:
    """Run real extraction/commit, then publish a real index with controlled vectors."""
    item = store.load_pdf_import(import_id)
    with store.database.connect() as connection:
        extract_job = connection.execute(
            "SELECT id FROM jobs WHERE source_version_id=? AND kind='pdf-extract'",
            (item.source_version_id,),
        ).fetchone()[0]
    step = build_pdf_extract_steps(store)[0]
    store.claim_job(extract_job, "history-test")
    attempt = store.start_step_attempt(extract_job, step.step_key)
    guard = AttemptGuard("history-test", step.step_key, attempt.attempt_number)
    context = StepContext(
        extract_job, step.step_key, attempt.attempt_number, lambda: False, lambda: False, guard
    )
    assert step.inputs is not None
    fingerprint = hashlib.sha256(
        json.dumps(step.inputs(context), sort_keys=True).encode()
    ).hexdigest()
    step.commit(
        store, StepCommitContext(extract_job, step.step_key, guard, fingerprint, step.run(context))
    )
    store.finish_step_attempt(extract_job, step.step_key, attempt.attempt_number, None)
    store.finish_job(extract_job, "completed")
    extraction = store.load_job_steps(extract_job)[0].output_extraction_version_id
    assert extraction is not None
    report = store.load_extraction_report(extraction)
    assert report.outcome is not None and report.outcome.outcome == "complete"
    with store.database.connect() as connection:
        job = connection.execute(
            "SELECT job_id FROM search_handoffs WHERE extraction_version_id=?", (extraction,)
        ).fetchone()[0]
        settings = json.loads(
            connection.execute(
                "SELECT settings_json FROM search_job_inputs WHERE job_id=?", (job,)
            ).fetchone()[0]
        )
    store.claim_job(job, "history-test")
    attempt = store.start_step_attempt(job, "prepare")
    guard = AttemptGuard("history-test", "prepare", attempt.attempt_number)
    profile = profiles()["minilm-en"]
    chunks = make_chunks(report.passages, profile, lambda _: 1, lambda text: text)
    storage = SearchStorage(store)
    index = storage.create_generation(job, guard, profile, chunks, settings)
    graph = store.load_index(index)
    assert len(graph.chunks) == 1
    storage.commit_batch(job, guard, index, 0, [graph.chunks[0].id], [VECTOR])
    storage.finish_components(job, guard, index)
    store.commit_index_step(
        job, "prepare", graph.index, graph.chunks, graph.mappings, 1, {}, guard=guard
    )
    store.publish_readiness(job, item.source_id, item.source_version_id, [index], guard=guard)
    store.finish_step_attempt(job, "prepare", attempt.attempt_number, None)
    store.finish_job(job, "completed")
    with store.database.connect() as connection:
        row_id = connection.execute(
            "SELECT row_id FROM search_rows WHERE index_version_id=?", (index,)
        ).fetchone()[0]
    managed = store.load_import_managed_file(import_id)
    return IndexedImport(
        item,
        index,
        extraction,
        row_id,
        store.database.config.data_dir / managed.relative_path,
        content,
    )


def make_history(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, *, shared: bool = False
) -> History:
    notebook = store.create_notebook(new_id(), "History owner")
    other = store.create_notebook(new_id(), "Neighbour")
    # These legacy lifecycle controls construct MiniLM vectors, not the new default.
    with store._transaction() as connection:
        for item in (notebook, other):
            connection.execute(
                "INSERT INTO notebook_profiles(notebook_id,active_manifest_id,updated_at)"
                " VALUES(?,?,?)",
                (item.id, profiles()["minilm-en"].identity, utc_timestamp()),
            )
    ingest = PdfIngestService(store, "history-upload")
    original_remove = ingest.files.remove_recorded
    retained_staging = set()

    def fail_owned_staging(path, *args, **kwargs):
        if path in retained_staging:
            raise OSError("controlled upload cleanup failure")
        return original_remove(path, *args, **kwargs)

    def upload(client, notebook_id, content, *, target=None, retain=False):
        response = client.post(
            f"/notebooks/{notebook_id}/pdf-acquisitions",
            headers=HEADERS,
            json={
                "retry_key": new_id(),
                "filename": "history.pdf",
                "intent": "replace" if target else "add",
                "target_source_id": target,
            },
        )
        assert response.status_code == 201, response.text
        acquisition = store.load_pdf_acquisition(response.json()["id"])
        if retain:
            retained_staging.add(acquisition.staging_relative_path)
        response = client.put(
            f"/pdf-acquisitions/{acquisition.id}/content",
            headers={**HEADERS, "Content-Type": "application/pdf"},
            content=content,
        )
        assert response.status_code == 200, response.text
        return publish_import(store, response.json()["import_id"], content)

    old_bytes = native_text_pdf("Historical rainfall measurements remain immutable until expiry.")
    app = create_app(TOKEN, store=store, pdf_ingest=ingest)
    with monkeypatch.context() as scoped:
        scoped.setattr(ingest.files, "remove_recorded", fail_owned_staging)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            old = upload(client, notebook.id, old_bytes, retain=True)
            current = upload(
                client,
                notebook.id,
                native_text_pdf("Current rainfall measurements must survive history cleanup."),
                target=old.item.source_id,
            )
            neighbour = upload(
                client,
                other.id,
                old_bytes
                if shared
                else native_text_pdf(
                    "Neighbour rainfall measurements must survive history cleanup."
                ),
            )
    staging = store.database.config.data_dir / retained_staging.pop()
    assert staging.read_bytes() == old_bytes
    assert store.list_pending_acquisition_file_cleanup()
    # A real database snapshot contains the history that expiry must remove.
    snapshot = store.database.config.snapshot_dir / "snapshot-r12-history.sqlite3"
    snapshot.parent.mkdir(exist_ok=True)
    with store.database.connect() as connection, closing(sqlite3.connect(snapshot)) as target:
        connection.backup(target)
    return History(old, current, neighbour, staging, snapshot)


def assert_native_present(store: PersistenceStore, record: IndexedImport) -> None:
    with store.database.connect() as connection:
        for table in ("search_words", "search_vectors"):
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE rowid=?", (record.row_id,)
                ).fetchone()[0]
                == 1
            )
        assert (
            connection.execute(
                "SELECT count(*) FROM search_batches WHERE index_version_id=? AND complete=1",
                (record.index,),
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM job_steps WHERE output_index_version_id=?", (record.index,)
            ).fetchone()[0]
            == 1
        )


def assert_survivors(store: PersistenceStore, history: History) -> None:
    for record in (history.current, history.neighbour):
        assert store.load_pdf_import(record.item.id).state == "ready"
        assert record.original.read_bytes() == record.content
        assert_native_present(store, record)
        notebook = store.load_source(record.item.source_id).notebook_id
        storage = SearchStorage(store)
        assert storage.search(notebook, [record.item.source_version_id], query="rainfall")
        assert storage.search(
            notebook,
            [record.item.source_version_id],
            vector=VECTOR,
            vector_manifest_id=profiles()["minilm-en"].identity,
        )
    state = store.load_source_pdf_state(history.current.item.source_id)
    assert state is not None
    assert state.current_import_id == history.current.item.id and state.pending_import_id is None


def assert_purged(store: PersistenceStore, history: History) -> None:
    old = history.old
    with pytest.raises(PersistenceError):
        store.load_pdf_import(old.item.id)
    with store.database.connect() as connection:
        for table in ("search_words", "search_vectors"):
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE rowid=?", (old.row_id,)
                ).fetchone()[0]
                == 0
            )
        for table in (
            "search_generations",
            "search_batches",
            "search_components",
            "search_rows",
            "chunk_passages",
            "retrieval_chunks",
        ):
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE index_version_id=?", (old.index,)
                ).fetchone()[0]
                == 0
            )
        for table in ("index_versions", "passages", "extraction_versions", "jobs"):
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE source_version_id=?",
                    (old.item.source_version_id,),
                ).fetchone()[0]
                == 0
            )
        assert (
            connection.execute(
                "SELECT count(*) FROM search_handoffs WHERE extraction_version_id=?",
                (old.extraction,),
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT count(*) FROM search_words").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM search_vectors").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert not history.staging.exists() and not history.snapshot.exists()
    assert store.list_pending_acquisition_file_cleanup() == ()
    assert_survivors(store, history)


def clean_after_reopen(store: PersistenceStore, history: History, entry: str) -> PersistenceStore:
    restarted = reopen(store)
    ingest = PdfIngestService(restarted, "history-restarted")
    if entry == "maintenance":
        try:
            ingest.maintenance()
        finally:
            asyncio.run(ingest.stop())
    else:
        sweeper = TrashSweeper(restarted, maintenance=ingest.maintenance)
        app = create_app(TOKEN, store=restarted, pdf_ingest=ingest, sweeper=sweeper)
        with TestClient(app, base_url="http://127.0.0.1"):
            deadline = time.monotonic() + 10
            while any(
                row.import_id == history.old.item.id
                for row in restarted.list_expired_pdf_import_trash(FUTURE)
            ):
                assert time.monotonic() < deadline, "startup history purge did not finish"
                time.sleep(0.01)
    return restarted


def test_unexpired_history_retains_native_evidence_but_is_not_searchable(store, monkeypatch):
    history = make_history(store, monkeypatch)
    store.trash_pdf_import(history.old.item.id, FUTURE)
    ingest = PdfIngestService(store, "unexpired-maintenance")
    try:
        ingest.maintenance()
    finally:
        asyncio.run(ingest.stop())
    restarted = reopen(store)
    assert_native_present(restarted, history.old)
    assert history.old.original.read_bytes() == history.old.content
    assert history.snapshot.exists()
    notebook = store.load_source(history.old.item.source_id).notebook_id
    storage = SearchStorage(restarted)
    assert storage.search(notebook, [history.old.item.source_version_id], query="rainfall") == ()
    assert (
        storage.search(
            notebook,
            [history.old.item.source_version_id],
            vector=VECTOR,
            vector_manifest_id=profiles()["minilm-en"].identity,
        )
        == ()
    )
    assert_survivors(restarted, history)


@pytest.mark.parametrize("entry", ["maintenance", "startup"])
@pytest.mark.parametrize("shared", [False, True])
def test_expired_history_purges_native_rows_and_preserves_current_and_shared_original(
    store, monkeypatch, entry, shared
):
    history = make_history(store, monkeypatch, shared=shared)
    assert (history.old.original == history.neighbour.original) is shared
    store.trash_pdf_import(history.old.item.id, EXPIRED)
    restarted = clean_after_reopen(store, history, entry)
    assert_purged(restarted, history)
    assert history.old.original.exists() is shared
    # Completed cleanup is safe to replay, including its shared-file decision.
    assert_purged(clean_after_reopen(restarted, history, entry), history)


@pytest.mark.parametrize("entry", ["maintenance", "startup"])
@pytest.mark.parametrize("phase", ["file", "snapshot", "records"])
def test_history_cleanup_failures_keep_native_evidence_and_resume_after_restart(
    store, monkeypatch, entry, phase
):
    history = make_history(store, monkeypatch)
    store.trash_pdf_import(history.old.item.id, EXPIRED)
    ingest = PdfIngestService(store, "interrupted-maintenance")
    with monkeypatch.context() as scoped:
        if phase == "file":
            original_remove = ingest.files.remove_recorded

            def fail_file(path, *args, **kwargs):
                if store.database.config.data_dir / path == history.old.original:
                    raise OSError("controlled original unlink failure")
                return original_remove(path, *args, **kwargs)

            scoped.setattr(ingest.files, "remove_recorded", fail_file)
        elif phase == "snapshot":
            original_unlink = Path.unlink

            def fail_snapshot(path, *args, **kwargs):
                if path == history.snapshot:
                    raise OSError("controlled snapshot unlink failure")
                return original_unlink(path, *args, **kwargs)

            scoped.setattr(Path, "unlink", fail_snapshot)
        else:
            from core.search import storage

            original_purge = storage.purge_search

            def fail_after_native_deletion(connection, **kwargs):
                original_purge(connection, **kwargs)
                assert connection.in_transaction
                assert (
                    connection.execute(
                        "SELECT count(*) FROM search_vectors WHERE rowid=?", (history.old.row_id,)
                    ).fetchone()[0]
                    == 0
                )
                raise PersistenceError("controlled failure inside native purge transaction")

            scoped.setattr(storage, "purge_search", fail_after_native_deletion)
        try:
            ingest.maintenance()
        finally:
            asyncio.run(ingest.stop())
    # The independent connection must see rollback, not partial native deletion.
    assert_native_present(reopen(store), history.old)
    pending = store.list_expired_pdf_import_trash(FUTURE)
    assert len(pending) == 1 and pending[0].import_id == history.old.item.id
    assert (pending[0].file_removed_at is None) is (phase == "file")
    assert (pending[0].snapshots_removed_at is not None) is (phase == "records")
    assert_survivors(store, history)
    assert_purged(clean_after_reopen(store, history, entry), history)


@pytest.mark.parametrize("entry", ["maintenance", "startup"])
def test_process_exit_inside_native_purge_rolls_back_and_restart_finishes(
    store, monkeypatch, entry
):
    history = make_history(store, monkeypatch)
    store.trash_pdf_import(history.old.item.id, EXPIRED)
    ingest = PdfIngestService(store, "before-crash")

    def pause_before_record_transaction(_import_id):
        raise PersistenceError("controlled handoff to crashing process")

    with monkeypatch.context() as scoped:
        scoped.setattr(store, "remove_trashed_pdf_import_records", pause_before_record_transaction)
        try:
            ingest.maintenance()
        finally:
            asyncio.run(ingest.stop())
    pending = store.list_expired_pdf_import_trash(FUTURE)[0]
    assert pending.file_removed_at is not None and pending.snapshots_removed_at is not None
    assert_native_present(store, history.old)
    script = """
import os
import sys
from pathlib import Path
from core.persistence.database import Database, DatabaseConfig
from core.persistence.store import PersistenceStore
from core.search import storage

database = Database(DatabaseConfig(Path(sys.argv[1])))
database.initialize()
original = storage.purge_search
def crash_after_native_deletion(connection, **kwargs):
    original(connection, **kwargs)
    assert connection.in_transaction
    for table in ('search_words', 'search_vectors'):
        query = f'SELECT count(*) FROM {table} WHERE rowid=?'
        assert connection.execute(query, (int(sys.argv[3]),)).fetchone()[0] == 0
    print('native deletion reached inside uncommitted transaction', flush=True)
    os._exit(73)
storage.purge_search = crash_after_native_deletion
PersistenceStore(database).remove_trashed_pdf_import_records(sys.argv[2])
raise AssertionError('crash seam was not reached')
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(store.database.config.data_dir),
            history.old.item.id,
            str(history.old.row_id),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=worker_environment(),
    )
    try:
        stdout, stderr = process.communicate(timeout=20)
    finally:
        stop_owned_process_tree(process)
    assert process.returncode == 73, (stdout, stderr)
    assert stdout.strip() == b"native deletion reached inside uncommitted transaction"
    assert stderr == b""
    # A fresh runtime must recover both native virtual tables and all ownership
    # before retrying. This is an actual process exit, not an exception rollback.
    restarted = reopen(store)
    assert_native_present(restarted, history.old)
    assert_survivors(restarted, history)
    assert_purged(clean_after_reopen(restarted, history, entry), history)


def test_installed_shared_model_survives_earlier_import_expiry(store, monkeypatch, request):
    directory = request.config.getoption("--models-directory", default=None)
    if directory is None:
        pytest.skip("explicit installed-model directory required; no implicit downloads")
    profile = profiles()["minilm-en"]
    source = Path(directory) / profile.identity
    verify_installation(source, profile)
    installed = installation_path(store.database.config.data_dir, profile)
    shutil.copytree(source, installed)
    setup = ModelSetupService(store)
    setup.recover()
    assert setup.state(profile.id)["state"] == "installed"
    history = make_history(store, monkeypatch)
    store.trash_pdf_import(history.old.item.id, EXPIRED)
    restarted = clean_after_reopen(store, history, "startup")
    assert_purged(restarted, history)
    verify_installation(installed, profile)
    recovered = ModelSetupService(restarted)
    recovered.recover()
    assert recovered.state(profile.id)["state"] == "installed"
    with restarted.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM notebook_profiles WHERE active_manifest_id=?",
                (profile.identity,),
            ).fetchone()[0]
            == 2
        )
