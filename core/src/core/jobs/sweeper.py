"""Trash-expiry sweeper (subproblem 1.3, decision 7).

Purges expired trash at startup and on a wall-clock timer that fires
regardless of user activity (default hourly, never less than daily), so a
session left open past a deadline still cleans up. Purge phases are recorded
and idempotent: managed files first, then migration snapshots (a snapshot is a
full database copy and can still contain purged data), then the atomic records
removal that takes the source row and its trash tracker together.

Cleanup failures stay pending: a file that survives an I/O error keeps its
durable removal marker (which also blocks new references to it) and the phase
stays unmarked, so the next sweep — including one after a restart — retries
it. An already-missing file counts as removed; an I/O failure does not. A
failure in one source's purge never stops the sweep of the others, and a
failed sweep never stops the sweeper.
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
            try:
                await asyncio.to_thread(self.sweep_once)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a failed sweep must not stop later sweeps
                logger.exception("trash sweep failed; expired trash stays pending for a retry")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                return

    def sweep_once(self) -> tuple[str, ...]:
        """Purge every fully-expired source; returns the purged source ids.

        A source whose purge is incomplete (a failed removal it will retry) or
        errors stays pending and is not reported as purged; the rest continue.
        """
        purged: list[str] = []
        for trash in self._store.list_expired_trash(utc_timestamp()):
            try:
                if self._purge(trash.source_id):
                    purged.append(trash.source_id)
            except Exception:  # noqa: BLE001 - keep other sources moving
                logger.exception("purge for %s failed; it stays pending", trash.source_id)
        return tuple(purged)

    def _purge(self, source_id: str) -> bool:
        """Advance one source's purge; True only when its records are removed."""
        trash = self._store.load_trash(source_id)
        if trash is None:
            return False
        if trash.files_removed_at is None:
            if not self._cleanup_managed_files(source_id):
                return False  # failures keep the phase unmarked for the next sweep
            self._store.mark_trash_files_removed(source_id)
        if trash.snapshots_removed_at is None:
            if not self._cleanup_migration_snapshots():
                return False
            self._store.mark_trash_snapshots_removed(source_id)
        self._store.remove_trashed_records(source_id)
        return True

    def _cleanup_managed_files(self, source_id: str) -> bool:
        """Unlink the source's managed files that no other source references.

        Returns True when the phase may be marked: every file is either still
        referenced elsewhere, already gone, or removed now. The reference
        check and the removal reservation happen in one transaction, so a
        reference cannot slip in between them; files that fail are recorded on
        their removal marker and keep the phase pending.
        """
        data_dir = self._store.database.config.data_dir.resolve()
        all_resolved = True
        for managed in self._store.list_source_managed_files(source_id):
            removal = self._store.reserve_managed_file_removal(
                managed.id, managed.relative_path, source_id
            )
            if removal is None:
                continue  # another source still needs these bytes
            if removal.removed_at is not None:
                continue
            candidate = (data_dir / managed.relative_path).resolve()
            if not candidate.is_relative_to(data_dir):
                logger.warning("skipping managed file outside the data directory")
                self._store.record_managed_file_removal_failure(
                    managed.id, "path escapes the data directory"
                )
                all_resolved = False
                continue
            try:
                candidate.unlink(missing_ok=True)
            except OSError as error:
                self._store.record_managed_file_removal_failure(managed.id, type(error).__name__)
                all_resolved = False
                continue
            self._store.mark_managed_file_removed(managed.id)
        return all_resolved

    def _cleanup_migration_snapshots(self) -> bool:
        snapshot_dir = self._store.database.config.snapshot_dir
        if not snapshot_dir.is_dir():
            return True
        all_removed = True
        for candidate in snapshot_dir.glob("snapshot-r*.sqlite3*"):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                logger.warning("migration snapshot could not be removed during purge")
                all_removed = False
        return all_removed
