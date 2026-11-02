"""Whole-source process-exit recovery with real PDF imports and native indexes.

Reuses controlled vectors from history fixtures, not a model-quality/speed check.
"""

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_import_expiry import (
    EXPIRED,
    FUTURE,
    TOKEN,
    VECTOR,
    History,
    assert_native_present,
    make_history,
)
from test_search_expiry import reopen

from core.api.app import create_app
from core.ingest.service import PdfIngestService
from core.jobs.sweeper import TrashSweeper
from core.persistence.errors import PersistenceError
from core.persistence.store import PersistenceStore
from core.search.install import ModelSetupService
from core.search.profiles import installation_path, profiles, verify_installation
from core.search.storage import SearchStorage
from core.search.supervisor import stop_owned_process_tree, worker_environment


def assert_neighbour_survives(store: PersistenceStore, history: History) -> None:
    record = history.neighbour
    assert store.load_pdf_import(record.item.id).state == "ready"
    assert record.original.read_bytes() == record.content
    assert_native_present(store, record)
    source = store.load_source(record.item.source_id)
    state = store.load_source_pdf_state(source.id)
    assert state is not None and state.current_import_id == record.item.id
    storage = SearchStorage(store)
    assert storage.search(source.notebook_id, [record.item.source_version_id], query="rainfall")
    assert storage.search(
        source.notebook_id,
        [record.item.source_version_id],
        vector=VECTOR,
        vector_manifest_id=profiles()["minilm-en"].identity,
    )


def assert_source_purged(store: PersistenceStore, history: History, *, shared: bool) -> None:
    source_id = history.old.item.source_id
    assert store.load_trash(source_id) is None
    with pytest.raises(PersistenceError):
        store.load_source(source_id)
    for record in (history.old, history.current):
        with pytest.raises(PersistenceError):
            store.load_pdf_import(record.item.id)
        with store.database.connect() as connection:
            for table in ("search_words", "search_vectors"):
                assert (
                    connection.execute(
                        f"SELECT count(*) FROM {table} WHERE rowid=?", (record.row_id,)
                    ).fetchone()[0]
                    == 0
                )
            for table in (
                "search_generations",
                "search_batches",
                "search_components",
                "search_rows",
                "profile_search_publications",
                "chunk_passages",
                "retrieval_chunks",
            ):
                assert (
                    connection.execute(
                        f"SELECT count(*) FROM {table} WHERE index_version_id=?", (record.index,)
                    ).fetchone()[0]
                    == 0
                )
            assert (
                connection.execute(
                    "SELECT count(*) FROM search_handoffs WHERE extraction_version_id=?",
                    (record.extraction,),
                ).fetchone()[0]
                == 0
            )
    with store.database.connect() as connection:
        for record in (history.old, history.current):
            assert record.item.acquisition_id is not None
            assert (
                connection.execute(
                    "SELECT count(*) FROM pdf_acquisitions WHERE id=?",
                    (record.item.acquisition_id,),
                ).fetchone()[0]
                == 0
            )
            assert (
                connection.execute(
                    "SELECT count(*) FROM deleted_pdf_acquisitions WHERE id=?",
                    (record.item.acquisition_id,),
                ).fetchone()[0]
                == 1
            )
        for table in (
            "source_versions",
            "pdf_imports",
            "source_pdf_state",
            "passages",
            "extraction_versions",
            "index_versions",
            "jobs",
            "source_readiness",
        ):
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE source_id=?", (source_id,)
                ).fetchone()[0]
                == 0
            )
        assert connection.execute("SELECT count(*) FROM search_words").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM search_vectors").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert not history.current.original.exists()
    assert history.old.original.exists() is shared
    assert not history.staging.exists() and not history.snapshot.exists()
    assert store.list_pending_acquisition_file_cleanup() == ()
    assert_neighbour_survives(store, history)


def finish_source_cleanup(store: PersistenceStore, source_id: str, entry: str) -> PersistenceStore:
    restarted = reopen(store)
    ingest = PdfIngestService(restarted, "source-restarted")
    sweeper = TrashSweeper(restarted, maintenance=ingest.maintenance)
    if entry == "maintenance":
        try:
            ingest.maintenance()
            assert sweeper.sweep_once() == (source_id,)
            assert sweeper.sweep_once() == ()
        finally:
            asyncio.run(ingest.stop())
    else:
        app = create_app(TOKEN, store=restarted, pdf_ingest=ingest, sweeper=sweeper)
        with TestClient(app, base_url="http://127.0.0.1"):
            deadline = time.monotonic() + 10
            while restarted.load_trash(source_id) is not None:
                assert time.monotonic() < deadline, "startup source purge did not finish"
                time.sleep(0.01)
            # Leave the application lifespan before replaying cleanup.
        assert TrashSweeper(restarted).sweep_once() == ()
    return restarted


def prepare_source_record_phase(store, monkeypatch, history):
    source_id = history.old.item.source_id
    store.trash_source(source_id, EXPIRED)
    ingest = PdfIngestService(store, "source-before-crash")
    try:
        ingest.maintenance()
    finally:
        asyncio.run(ingest.stop())

    def pause_records(_source_id):
        raise PersistenceError("controlled handoff to crashing process")

    with monkeypatch.context() as scoped:
        scoped.setattr(store, "remove_trashed_records", pause_records)
        assert TrashSweeper(store).sweep_once() == ()
    pending = store.load_trash(source_id)
    assert pending is not None
    assert pending.files_removed_at is not None and pending.snapshots_removed_at is not None
    assert not history.staging.exists() and not history.snapshot.exists()
    for record in (history.old, history.current):
        assert_native_present(store, record)


@pytest.mark.parametrize("entry", ["maintenance", "startup"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("crash_point", ["native", "receipts"])
def test_source_process_exit_inside_native_purge_rolls_back_and_restart_finishes(
    store, monkeypatch, entry, shared, crash_point
):
    history = make_history(store, monkeypatch, shared=shared)
    assert (history.old.original == history.neighbour.original) is shared
    source_id = history.old.item.source_id
    prepare_source_record_phase(store, monkeypatch, history)
    script = """
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from core.persistence.database import Database, DatabaseConfig
from core.persistence.store import PersistenceStore
from core.search import storage

database = Database(DatabaseConfig(Path(sys.argv[1])))
database.initialize()
original = storage.purge_search
def verify_deleted(connection):
    assert connection.in_transaction
    for row_id in map(int, sys.argv[3:5]):
        for table in ('search_words', 'search_vectors'):
            query = f'SELECT count(*) FROM {table} WHERE rowid=?'
            assert connection.execute(query, (row_id,)).fetchone()[0] == 0
    for table in ('search_words', 'search_vectors'):
        query = f'SELECT count(*) FROM {table} WHERE rowid=?'
        assert connection.execute(query, (int(sys.argv[5]),)).fetchone()[0] == 1
    assert connection.execute(
        'SELECT count(*) FROM search_generations WHERE index_version_id IN (?,?)',
        sys.argv[6:8],
    ).fetchone()[0] == 0
def crash(connection):
    verify_deleted(connection)
    assert connection.execute('SELECT count(*) FROM deleted_pdf_acquisitions').fetchone()[0] == (
        0 if sys.argv[8] == 'native' else 2
    )
    print('uncommitted deletion reached: ' + sys.argv[8], flush=True)
    os._exit(73)
store = PersistenceStore(database)
if sys.argv[8] == 'native':
    def crash_after_native_deletion(connection, **kwargs):
        assert kwargs == {'source_id': sys.argv[2]}
        original(connection, **kwargs)
        crash(connection)
    storage.purge_search = crash_after_native_deletion
else:
    transaction = store._transaction
    @contextmanager
    def crash_after_receipts():
        with transaction() as connection:
            connection.create_function('crash_after_receipts', 0, lambda: crash(connection))
            connection.execute('''CREATE TEMP TRIGGER crash_after_receipts
                BEFORE DELETE ON main.pdf_acquisitions
                BEGIN SELECT crash_after_receipts(); END''')
            yield connection
    store._transaction = crash_after_receipts
store.remove_trashed_records(sys.argv[2])
raise AssertionError('crash seam was not reached')
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(store.database.config.data_dir),
            source_id,
            str(history.old.row_id),
            str(history.current.row_id),
            str(history.neighbour.row_id),
            history.old.index,
            history.current.index,
            crash_point,
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
    assert stdout.strip() == f"uncommitted deletion reached: {crash_point}".encode()
    assert stderr == b""
    # Fresh runtime recovery must restore both virtual tables and checkpoints;
    # a successful retry alone could hide a partial or skipped rollback.
    restarted = reopen(store)
    pending = restarted.load_trash(source_id)
    assert pending is not None and pending.files_removed_at and pending.snapshots_removed_at
    with restarted.database.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM deleted_pdf_acquisitions").fetchone()[0] == 0
        )
    for record in (history.old, history.current):
        assert_native_present(restarted, record)
        assert restarted.load_pdf_import(record.item.id).source_id == source_id
    assert_neighbour_survives(restarted, history)
    assert_source_purged(finish_source_cleanup(restarted, source_id, entry), history, shared=shared)


def test_unexpired_whole_source_keeps_real_originals_and_native_rows(store, monkeypatch):
    history = make_history(store, monkeypatch, shared=True)
    source_id = history.old.item.source_id
    notebook_id = store.load_source(source_id).notebook_id
    store.trash_source(source_id, FUTURE)
    restarted = reopen(store)
    assert TrashSweeper(restarted).sweep_once() == ()
    for record in (history.old, history.current):
        assert_native_present(restarted, record)
        assert record.original.read_bytes() == record.content
        storage = SearchStorage(restarted)
        assert storage.search(notebook_id, [record.item.source_version_id], query="rainfall") == ()
        assert (
            storage.search(
                notebook_id,
                [record.item.source_version_id],
                vector=VECTOR,
                vector_manifest_id=profiles()["minilm-en"].identity,
            )
            == ()
        )
    assert history.staging.exists() and history.snapshot.exists()
    assert_neighbour_survives(restarted, history)


def test_shared_installed_model_survives_whole_source_expiry(store, monkeypatch, request):
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
    history = make_history(store, monkeypatch, shared=True)
    store.trash_source(history.old.item.source_id, EXPIRED)
    restarted = finish_source_cleanup(store, history.old.item.source_id, "startup")
    assert_source_purged(restarted, history, shared=True)
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
