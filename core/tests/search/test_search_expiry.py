"""Real on-disk native search rows follow source-expiry cleanup across restart."""

import asyncio
import time

import pytest
from test_storage import seed

from core.jobs.sweeper import TrashSweeper
from core.persistence.database import Database
from core.persistence.store import PersistenceStore
from core.search.profiles import profiles
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage


def reopen(store: PersistenceStore) -> PersistenceStore:
    database = Database(store.database.config)
    database.initialize()
    return PersistenceStore(database)


def test_unexpired_trash_is_hidden_but_retains_native_rows(store: PersistenceStore) -> None:
    notebook, source, version, _, _, _ = seed(store, "rainfall", [1.0] + [0.0] * 383)
    store.trash_source(source, "9999-01-01T00:00:00.000000Z")
    restarted = reopen(store)
    assert TrashSweeper(restarted).sweep_once() == ()
    assert restarted.load_trash(source) is not None
    assert SearchStorage(restarted).search(notebook, [version], query="rainfall") == ()
    with restarted.database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM search_words").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM search_vectors").fetchone()[0] == 1


@pytest.mark.parametrize("entry", ["maintenance", "startup"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_expired_source_native_search_purge_survives_restart_and_preserves_neighbour(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, entry: str, interrupted: bool
) -> None:
    vector = [1.0] + [0.0] * 383
    notebook, source, version, index, _, _ = seed(store, "expired rainfall", vector)
    _, neighbour, neighbour_version, _, _, _ = seed(
        store, "protected rainfall", vector, notebook_id=notebook
    )
    shared_asset = store.database.config.data_dir / "models" / "shared-asset-control"
    shared_asset.parent.mkdir()
    shared_asset.write_bytes(b"public sentinel, not model weights")
    store.trash_source(source, "2000-01-01T00:00:00.000000Z")
    assert SearchStorage(store).search(notebook, [version], query="rainfall") == ()

    if interrupted:

        def fail_final_phase(_source: str) -> None:
            raise OSError("controlled final-record cleanup interruption")

        with monkeypatch.context() as scoped:
            scoped.setattr(store, "remove_trashed_records", fail_final_phase)
            assert TrashSweeper(store).sweep_once() == ()
        pending = store.load_trash(source)
        assert pending is not None
        assert pending.files_removed_at is not None and pending.snapshots_removed_at is not None
        with store.database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM search_vectors").fetchone()[0] == 2

    restarted = reopen(store)
    sweeper = TrashSweeper(restarted)
    if entry == "maintenance":
        assert sweeper.sweep_once() == (source,)
    else:

        async def startup() -> None:
            await sweeper.start()
            try:
                deadline = time.monotonic() + 10
                while restarted.load_trash(source) is not None:
                    assert time.monotonic() < deadline, "startup purge did not finish"
                    await asyncio.sleep(0.01)
            finally:
                await sweeper.stop()

        asyncio.run(startup())
    assert restarted.load_trash(source) is None
    assert restarted.load_source(neighbour).id == neighbour
    assert shared_asset.read_bytes() == b"public sentinel, not model weights"
    storage = SearchStorage(restarted)
    assert storage.search(notebook, [neighbour_version], query="rainfall")
    assert storage.search(
        notebook,
        [neighbour_version],
        vector=vector_bytes(vector, 384),
        vector_manifest_id=profiles()["bge-small-en"].identity,
    )
    with restarted.database.connect() as connection:
        for table in ("search_generations", "search_batches", "search_components", "search_rows"):
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE index_version_id=?", (index,)
                ).fetchone()[0]
                == 0
            )
        assert connection.execute("SELECT count(*) FROM search_words").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM search_vectors").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
