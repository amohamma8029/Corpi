"""Stable provider failures shared by the core service and API boundary."""

from __future__ import annotations


class ProviderFailure(Exception):
    """A sanitized, actionable provider or credential failure."""

    def __init__(self, code: str, message: str, *, status: int = 502) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class ProviderCancelled(ProviderFailure):
    """An interactive call was stopped before a usable result existed."""

    def __init__(self, message: str = "The provider call was stopped.") -> None:
        super().__init__("cancelled", message, status=499)
