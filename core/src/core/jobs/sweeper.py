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
import stat
from collections.abc import Callable
from pathlib import Path

from core.ingest.files import ManagedPdfFiles
from core.ingest.locks import FingerprintLockCoordinator
from core.persistence.errors import PersistenceError
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
        fingerprint_locks: FingerprintLockCoordinator | None = None,
        maintenance: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._interval = max(0.01, interval_seconds)
        self._fingerprint_locks = fingerprint_locks
        self._maintenance = maintenance
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
                await asyncio.to_thread(self._run_maintenance)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a failed sweep must not stop later sweeps
                logger.exception("trash sweep failed; expired trash stays pending for a retry")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                return

    def _run_maintenance(self) -> None:
        self.sweep_once()
        if self._maintenance is not None:
            self._maintenance()

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
            lock = (
                None
                if self._fingerprint_locks is None
                else self._fingerprint_locks.lock_for(managed.sha256, managed.size_bytes)
            )
            if lock is not None and not lock.acquire(timeout=30.0):
                all_resolved = False
                continue
            try:
                removal = self._store.reserve_managed_file_removal(
                    managed.id, managed.relative_path, source_id
                )
                if removal is None:
                    if self._store.has_live_pdf_reuse_hold(managed.id):
                        all_resolved = False
                    continue  # another source or a live acquisition still needs these bytes
                if removal.removed_at is not None:
                    continue
                try:
                    if managed.relative_path.startswith("files/"):
                        ManagedPdfFiles(data_dir).remove_recorded(managed.relative_path)
                    else:
                        candidate = (data_dir / managed.relative_path).resolve()
                        if not candidate.is_relative_to(data_dir) or _has_redirected_component(
                            data_dir, managed.relative_path
                        ):
                            raise OSError("managed file path is outside or redirected")
                        candidate.unlink(missing_ok=True)
                except (OSError, PersistenceError) as error:
                    self._store.record_managed_file_removal_failure(
                        managed.id, type(error).__name__
                    )
                    all_resolved = False
                    continue
                self._store.mark_managed_file_removed(managed.id)
            finally:
                if lock is not None:
                    lock.release()
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


def _has_redirected_component(root: Path, relative_path: str) -> bool:
    current = root
    parts = Path(relative_path).parts
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            return True
    return False
