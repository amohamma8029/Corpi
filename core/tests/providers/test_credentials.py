import pytest
from keyring.errors import KeyringError, PasswordDeleteError

from core.providers.credentials import KeyringCredentialStore
from core.providers.errors import ProviderFailure


def test_keyring_failure_is_actionable_and_sanitized(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise KeyringError("backend unavailable")

    monkeypatch.setattr("core.providers.credentials.keyring.get_password", fail)

    with pytest.raises(ProviderFailure) as failure:
        KeyringCredentialStore().get("google-gemini")

    assert failure.value.code == "credential_store_unavailable"
    assert "backend unavailable" not in str(failure.value)


def test_delete_failure_rechecks_and_reports_a_remaining_key(monkeypatch) -> None:
    def fail_delete(service, account):
        raise PasswordDeleteError("denied")

    monkeypatch.setattr(
        "core.providers.credentials.keyring.get_password",
        lambda service, account: "still-there",
    )
    monkeypatch.setattr(
        "core.providers.credentials.keyring.delete_password",
        fail_delete,
    )

    with pytest.raises(ProviderFailure) as failure:
        KeyringCredentialStore().delete("google-gemini")

    assert failure.value.code == "credential_store_unavailable"


def test_delete_verifies_removal_before_success(monkeypatch) -> None:
    state: dict[str, str | None] = {"value": "secret"}

    def get_password(service, account):
        return state["value"]

    def delete_password(service, account):
        state["value"] = None

    monkeypatch.setattr("core.providers.credentials.keyring.get_password", get_password)
    monkeypatch.setattr("core.providers.credentials.keyring.delete_password", delete_password)

    KeyringCredentialStore().delete("google-gemini")

    assert state["value"] is None


def test_delete_is_a_no_op_when_no_credential_exists(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        "core.providers.credentials.keyring.get_password",
        lambda service, account: None,
    )
    monkeypatch.setattr(
        "core.providers.credentials.keyring.delete_password",
        lambda service, account: calls.append(account),
    )

    KeyringCredentialStore().delete("google-gemini")

    assert calls == []
