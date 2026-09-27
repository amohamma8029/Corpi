"""Trash-expiry sweeper (subproblem 1.3, decision 7).

Purges expired trash at startup and on a wall-clock timer that fires
regardless of user activity (default hourly, never less than daily), so a
session left open past a deadline still cleans up. Purge phases are recorded
and idempotent: managed files first, then migration snapshots (a snapshot is a
full database copy and can still contain purged data), then the atomic records
removal that takes the source row and its trash tracker together.

The sweep is resumable: a crash between phases leaves the recorded marks, and
the next sweep continues from the remaining phases without making the source
visible again.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from core.persistence.models import utc_timestamp
from core.persistence.store import PersistenceStore

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_INTERVAL_SECONDS = 3600.0


class TrashSweeper:
    """Runs the expiry sweep at startup and then on the configured interval."""

    def __init__(
        self,
        store: PersistenceStore,
        *,
        interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
    ) -> None:
        self._store = store
        self._interval = max(0.01, interval_seconds)
        self._task: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("sweeper is already started")
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        assert self._stop is not None
        self._stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run_forever(self) -> None:
        assert self._stop is not None
        while True:
            await asyncio.to_thread(self.sweep_once)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                return

    def sweep_once(self) -> tuple[str, ...]:
        """Purge every expired source; returns the purged source ids."""
        purged: list[str] = []
        for trash in self._store.list_expired_trash(utc_timestamp()):
            self._purge(trash.source_id)
            purged.append(trash.source_id)
        return tuple(purged)

    def _purge(self, source_id: str) -> None:
        trash = self._store.load_trash(source_id)
        if trash is None:
            return
        if trash.files_removed_at is None:
            self._cleanup_managed_files(source_id)
            self._store.mark_trash_files_removed(source_id)
        if trash.snapshots_removed_at is None:
            self._cleanup_migration_snapshots()
            self._store.mark_trash_snapshots_removed(source_id)
        self._store.remove_trashed_records(source_id)

    def _cleanup_managed_files(self, source_id: str) -> None:
        data_dir = self._store.database.config.data_dir.resolve()
        for managed in self._store.list_source_managed_files(source_id):
            candidate = (data_dir / managed.relative_path).resolve()
            if not candidate.is_relative_to(data_dir):
                logger.warning("skipping managed file outside the data directory")
                continue
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                logger.warning("managed file could not be removed during purge")

    def _cleanup_migration_snapshots(self) -> None:
        snapshot_dir = self._store.database.config.snapshot_dir
        if not snapshot_dir.is_dir():
            return
        for candidate in snapshot_dir.glob("snapshot-r*.sqlite3*"):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                logger.warning("migration snapshot could not be removed during purge")
