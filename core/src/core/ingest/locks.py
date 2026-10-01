"""Shared per-fingerprint coordination for PDF reuse and physical removal."""

from __future__ import annotations

import threading
from contextlib import contextmanager


class FingerprintLockCoordinator:
    """Maps fingerprints to bounded lock stripes shared by import and purge."""

    _STRIPE_COUNT = 64

    def __init__(self) -> None:
        self._locks = tuple(threading.Lock() for _ in range(self._STRIPE_COUNT))

    def _lock_for(self, checksum: str, size_bytes: int) -> threading.Lock:
        stripe = hash((checksum, size_bytes)) % self._STRIPE_COUNT
        return self._locks[stripe]

    @contextmanager
    def hold(self, checksum: str, size_bytes: int, timeout: float):
        lock = self._lock_for(checksum, size_bytes)
        if not lock.acquire(timeout=max(0.0, timeout)):
            raise TimeoutError("PDF content is being finalized or removed")
        try:
            yield
        finally:
            lock.release()

    def lock_for(self, checksum: str, size_bytes: int) -> threading.Lock:
        """Return the lock to a synchronous purge step using the same lock map."""
        return self._lock_for(checksum, size_bytes)
