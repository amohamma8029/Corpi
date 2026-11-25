"""Credential storage boundary for generation providers.

Only ProviderService calls this module. The desktop may submit or replace a
secret through the authenticated local API, but it never reads one back.
"""

from __future__ import annotations

from typing import Protocol

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from core.providers.errors import ProviderFailure

SERVICE_NAME = "app.corpi.desktop"


class CredentialStore(Protocol):
    def get(self, provider_id: str) -> str | None: ...

    def set(self, provider_id: str, secret: str) -> None: ...

    def delete(self, provider_id: str) -> None: ...


def _account(provider_id: str) -> str:
    return f"provider/{provider_id}"


class KeyringCredentialStore:
    """OS-native credential store: Credential Locker, Keychain, or Secret Service."""

    def get(self, provider_id: str) -> str | None:
        try:
            return keyring.get_password(SERVICE_NAME, _account(provider_id))
        except KeyringError as error:
            raise ProviderFailure(
                "credential_store_unavailable",
                "The operating-system credential store is unavailable.",
                status=503,
            ) from error

    def set(self, provider_id: str, secret: str) -> None:
        if not secret:
            raise ValueError("credential must not be empty")
        try:
            keyring.set_password(SERVICE_NAME, _account(provider_id), secret)
        except KeyringError as error:
            raise ProviderFailure(
                "credential_store_unavailable",
                "The operating-system credential store could not save the credential.",
                status=503,
            ) from error

    def delete(self, provider_id: str) -> None:
        account = _account(provider_id)
        try:
            if keyring.get_password(SERVICE_NAME, account) is None:
                return
            try:
                keyring.delete_password(SERVICE_NAME, account)
            except PasswordDeleteError:
                # Some backends report an error even though the item was removed.
                pass
            # Verify the result so the API never claims success while the key
            # remains in the OS store, regardless of how the backend reported it.
            if keyring.get_password(SERVICE_NAME, account) is not None:
                raise ProviderFailure(
                    "credential_store_unavailable",
                    "The operating-system credential store did not remove the credential.",
                    status=503,
                )
        except ProviderFailure:
            raise
        except KeyringError as error:
            raise ProviderFailure(
                "credential_store_unavailable",
                "The operating-system credential store could not remove the credential.",
                status=503,
            ) from error


class InMemoryCredentialStore:
    """Test-only store; never selected by the application entry point."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, provider_id: str) -> str | None:
        return self._values.get(provider_id)

    def set(self, provider_id: str, secret: str) -> None:
        if not secret:
            raise ValueError("credential must not be empty")
        self._values[provider_id] = secret

    def delete(self, provider_id: str) -> None:
        self._values.pop(provider_id, None)
