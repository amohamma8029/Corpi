"""Cross-process guard: one core per data directory.

SQLite serializes individual writes, not the multi-step sequence that makes up
an import, so two cores sharing one data directory could both claim and run the
same job. The lock is acquired before database initialization — covering
startup migrations — and held until the process exits or the core stops.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import IO

from core.persistence.errors import CoreAlreadyRunningError, InvalidDataDirectoryError

LOCK_FILENAME = "core.lock"


class DataDirectoryLock:
    """An advisory OS lock on ``<data_dir>/core.lock`` held for the core's life."""

    def __init__(self, data_dir: Path, handle: IO[bytes]) -> None:
        self._data_dir = data_dir
        self._handle = handle

    @classmethod
    def acquire(cls, data_dir: Path) -> DataDirectoryLock:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise InvalidDataDirectoryError("local data directory is unavailable") from None

        lock_path = data_dir / LOCK_FILENAME
        try:
            handle = open(lock_path, "a+b")
        except OSError:
            raise InvalidDataDirectoryError("local data directory is unavailable") from None

        try:
            _ensure_lockable_byte(handle)
            _lock(handle)
        except OSError:
            handle.close()
            raise CoreAlreadyRunningError(
                "another Corpi core already holds this data directory"
            ) from None
        _record_owner(handle)
        return cls(data_dir, handle)

    def release(self) -> None:
        if self._handle.closed:
            return
        try:
            _unlock(self._handle)
        finally:
            self._handle.close()

    def __enter__(self) -> DataDirectoryLock:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def _ensure_lockable_byte(handle: IO[bytes]) -> None:
    """Byte-range locks need a byte to exist; keep one even when recording owner info."""
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()


def _record_owner(handle: IO[bytes]) -> None:
    """Diagnostic breadcrumb; the lock itself is the enforcement."""
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n".encode("ascii"))
        handle.flush()
    except OSError:
        pass


def _lock(handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
