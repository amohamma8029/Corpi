"""Trash sweeper: startup and timer expiry with resumable phases."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.jobs.sweeper import TrashSweeper
from core.persistence.errors import PersistenceError
from core.persistence.models import ManagedFile, new_id, utc_timestamp
from core.persistence.store import PersistenceStore


def past_deadline() -> str:
    return (
        (datetime.now(UTC) - timedelta(minutes=1))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def future_deadline() -> str:
    return (
        (datetime.now(UTC) + timedelta(days=30))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def seed_trashed_source(store: PersistenceStore, data_dir: Path, deadline: str) -> str:
    notebook_id = new_id()
    store.create_notebook(notebook_id, "Trash")
    source_id = new_id()
    store.create_source(source_id, notebook_id, "victim.pdf")
    managed = ManagedFile(new_id(), "originals/victim.pdf", 17, "a" * 64, utc_timestamp())
    store.save_managed_file(managed)
    original = data_dir / "originals" / "victim.pdf"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"the original bytes")
    store.create_source_version(source_id, "a" * 64, managed.id)
    store.trash_source(source_id, deadline)
    return source_id


def test_expired_trash_is_purged_with_files_and_snapshots(
    store: PersistenceStore, data_dir: Path
) -> None:
    snapshot_dir = data_dir / "migration-snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / "snapshot-r0001.sqlite3"
    snapshot.write_bytes(b"full database copy")

    source_id = seed_trashed_source(store, data_dir, past_deadline())

    purged = TrashSweeper(store).sweep_once()
    assert purged == (source_id,)

    assert not (data_dir / "originals" / "victim.pdf").exists()
    assert not snapshot.exists()
    with pytest.raises(PersistenceError):
        store.load_source(source_id)
    assert store.load_trash(source_id) is None


def test_unexpired_trash_is_untouched(store: PersistenceStore, data_dir: Path) -> None:
    source_id = seed_trashed_source(store, data_dir, future_deadline())

    assert TrashSweeper(store).sweep_once() == ()
    assert store.load_source(source_id).id == source_id
    assert (data_dir / "originals" / "victim.pdf").is_file()


def test_completed_phases_are_not_repeated(store: PersistenceStore, data_dir: Path) -> None:
    source_id = seed_trashed_source(store, data_dir, past_deadline())
    original = data_dir / "originals" / "victim.pdf"

    # Simulate a crash after the files phase: the mark exists, the file does too.
    store.mark_trash_files_removed(source_id)
    TrashSweeper(store).sweep_once()

    # The marked phase is not repeated (the file is left as-is), while the
    # remaining phases complete and the tracker leaves with the source row.
    assert original.is_file()
    assert store.load_trash(source_id) is None
    with pytest.raises(PersistenceError):
        store.load_source(source_id)


def test_timer_sweeps_a_session_that_never_goes_idle(
    store: PersistenceStore, data_dir: Path
) -> None:
    async def scenario() -> bool:
        sweeper = TrashSweeper(store, interval_seconds=0.05)
        await sweeper.start()
        try:
            # Created after startup: only the wall-clock timer can catch it.
            source_id = seed_trashed_source(store, data_dir, past_deadline())
            for _ in range(250):
                if store.load_trash(source_id) is None:
                    return True
                await asyncio.sleep(0.02)
            return False
        finally:
            await sweeper.stop()

    assert asyncio.run(scenario())
