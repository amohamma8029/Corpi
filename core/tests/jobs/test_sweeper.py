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


def past_deadline(minutes_ago: float = 1.0) -> str:
    return (
        (datetime.now(UTC) - timedelta(minutes=minutes_ago))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def future_deadline() -> str:
    return (
        (datetime.now(UTC) + timedelta(days=30))
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def seed_trashed_source(
    store: PersistenceStore,
    data_dir: Path,
    deadline: str,
    filename: str = "victim.pdf",
) -> str:
    notebook_id = new_id()
    store.create_notebook(notebook_id, "Trash")
    source_id = new_id()
    store.create_source(source_id, notebook_id, filename)
    managed = ManagedFile(new_id(), f"originals/{filename}", 17, "a" * 64, utc_timestamp())
    store.save_managed_file(managed)
    original = data_dir / "originals" / filename
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


def seed_shared_file_source(
    store: PersistenceStore, data_dir: Path, file_id: str, name: str
) -> str:
    notebook_id = new_id()
    store.create_notebook(notebook_id, f"Shared {name}")
    source_id = new_id()
    store.create_source(source_id, notebook_id, name)
    store.create_source_version(source_id, "a" * 64, file_id)
    return source_id


def test_shared_managed_file_survives_until_its_last_source_is_purged(
    store: PersistenceStore, data_dir: Path
) -> None:
    shared = ManagedFile(new_id(), "originals/shared.pdf", 17, "a" * 64, utc_timestamp())
    store.save_managed_file(shared)
    original = data_dir / "originals" / "shared.pdf"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"the original bytes")

    first_source = seed_shared_file_source(store, data_dir, shared.id, "first.pdf")
    second_source = seed_shared_file_source(store, data_dir, shared.id, "second.pdf")

    store.trash_source(first_source, past_deadline())
    assert TrashSweeper(store).sweep_once() == (first_source,)

    # The bytes survive for the second source, and no removal was requested.
    assert original.is_file()
    assert store.load_managed_file_removal(shared.id) is None
    assert store.load_source(second_source).id == second_source

    store.trash_source(second_source, past_deadline())
    assert TrashSweeper(store).sweep_once() == (second_source,)
    assert not original.exists()


def test_failed_file_removal_stays_pending_and_retries_after_restart(
    store: PersistenceStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_id = seed_trashed_source(store, data_dir, past_deadline())
    original = data_dir / "originals" / "victim.pdf"
    assert original.is_file()

    original_unlink = Path.unlink
    calls = {"count": 0}

    def flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "victim.pdf" and calls["count"] == 0:
            calls["count"] += 1
            raise PermissionError("injected failure")
        return original_unlink(self, missing_ok=missing_ok)

    # Patch the concrete class (WindowsPath on Windows, PosixPath elsewhere):
    # the platform class can carry its own unlink implementation.
    monkeypatch.setattr(type(original), "unlink", flaky_unlink)

    assert TrashSweeper(store).sweep_once() == ()  # phase incomplete, source pending
    assert original.is_file()
    trash = store.load_trash(source_id)
    assert trash is not None
    assert trash.files_removed_at is None

    removal = store.load_managed_file_removal(store.list_source_managed_files(source_id)[0].id)
    assert removal is not None
    assert removal.removed_at is None
    assert removal.attempts == 1
    assert removal.last_error == "PermissionError"

    # "Restart": a fresh sweeper retries the pending removal and completes.
    monkeypatch.setattr(type(original), "unlink", original_unlink)
    assert TrashSweeper(store).sweep_once() == (source_id,)
    assert not original.exists()
    assert store.load_trash(source_id) is None


def test_missing_file_counts_as_removed(store: PersistenceStore, data_dir: Path) -> None:
    source_id = seed_trashed_source(store, data_dir, past_deadline())
    (data_dir / "originals" / "victim.pdf").unlink()

    assert TrashSweeper(store).sweep_once() == (source_id,)
    assert store.load_trash(source_id) is None


def test_one_failing_source_does_not_stop_the_sweep(
    store: PersistenceStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    earlier = past_deadline(minutes_ago=60)
    later = past_deadline(minutes_ago=30)
    failing_source = seed_trashed_source(store, data_dir, earlier, "victim-a.pdf")
    healthy_source = seed_trashed_source(store, data_dir, later, "victim-b.pdf")

    original = store.remove_trashed_records

    def flaky(source_id: str) -> None:
        if source_id == failing_source:
            raise PersistenceError("injected purge failure")
        return original(source_id)

    monkeypatch.setattr(store, "remove_trashed_records", flaky)

    assert TrashSweeper(store).sweep_once() == (healthy_source,)
    assert store.load_trash(failing_source) is not None  # stays pending

    monkeypatch.setattr(store, "remove_trashed_records", original)
    assert TrashSweeper(store).sweep_once() == (failing_source,)
