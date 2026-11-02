import asyncio
import hashlib
import sys
import time
from dataclasses import replace

import pytest

from core.persistence.corelock import DataDirectoryLock
from core.persistence.errors import ConflictingWriteError
from core.persistence.models import new_id, utc_timestamp
from core.persistence.store import PersistenceStore
from core.search.install import ModelSetupService
from core.search.profiles import ModelFile, installation_path, profiles, verify_installation


class SmallSetup(ModelSetupService):
    """Asset-protocol control, not an inference/model-quality substitute."""

    def profile(self, profile_id):
        return replace(
            profiles()[profile_id],
            files=(
                ModelFile("asset.bin", 4, hashlib.sha256(b"good").hexdigest(), "unused-test-url"),
            ),
        )


def worker(*, delay=0.0, content=b"good"):
    code = (
        "import sys,json,time; from pathlib import Path; "
        "p=Path(sys.argv[2]); (p/'assets').mkdir(parents=True,exist_ok=True); "
        f"time.sleep({delay}); (p/'assets'/'asset.bin').write_bytes({content!r}); "
        "print(json.dumps({'type':'done','bytes':4}),flush=True); time.sleep(10)"
    )
    return [sys.executable, "-c", code]


def wait(service, profile="minilm-en", timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = service.state(profile)
        if state["state"] != "downloading" and (
            service._thread is None or not service._thread.is_alive()
        ):
            return state
        time.sleep(0.02)
    raise AssertionError("model setup did not settle")


def test_setup_verifies_and_promotes_once_with_idempotent_request(store: PersistenceStore):
    service = SmallSetup(store, worker_command=worker())
    service.begin("minilm-en", "same-request", 0)
    service.begin("minilm-en", "same-request", 0)
    state = wait(service)
    assert state["state"] == "installed"
    profile = service.profile("minilm-en")
    verify_installation(installation_path(service.data_dir, profile), profile)
    with store.database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM model_setup_attempts").fetchone()[0] == 1
        assert (
            connection.execute("SELECT cleanup_pending FROM model_setup_attempts").fetchone()[0]
            == 0
        )
    assert service.state("minilm-en")["generation"] == 1


def test_cancel_invalidates_writer_before_promotion_and_other_transfer_is_rejected(
    store: PersistenceStore,
):
    service = SmallSetup(store, worker_command=worker(delay=5))
    service.begin("minilm-en", "cancel-me", 0)
    with pytest.raises(ConflictingWriteError, match="active"):
        service.begin("multilingual-e5", "other", 0)
    cancelled = service.cancel("minilm-en", 1)
    assert cancelled["state"] == "cancelled" and cancelled["generation"] == 2
    assert wait(service)["state"] == "cancelled"
    assert not installation_path(service.data_dir, service.profile("minilm-en")).exists()
    with store.database.connect() as connection:
        assert connection.execute(
            "SELECT state,cleanup_pending FROM model_setup_attempts"
        ).fetchone()[:] == ("cancelled", 0)


def test_corrupt_assets_never_install_and_retry_accounting_is_cumulative(store: PersistenceStore):
    service = SmallSetup(store, worker_command=worker(content=b"evil"), max_bytes=8)
    service.begin("minilm-en", "corrupt", 0)
    state = wait(service)
    assert state["state"] == "failed"
    assert state["received_bytes"] == 8 and state["transfer_attempts"] == 2
    assert not installation_path(service.data_dir, service.profile("minilm-en")).exists()


def test_restart_does_not_claim_partial_staging_as_installed_and_cleanup_waits_for_lock(
    store: PersistenceStore,
):
    service = SmallSetup(store)
    profile = service.profile("minilm-en")
    service._ensure(profile)
    identity = new_id()
    relative = "models/.setup/" + identity
    path = service._path(relative)
    path.mkdir(parents=True)
    (path / "partial.bin").write_bytes(b"partial")
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO model_setup_attempts(id,request_key,manifest_id,generation,state,"
            "staging_path,created_at,updated_at) VALUES(?,?,?,1,'running',?,?,?)",
            (identity, "interrupted", profile.identity, relative, utc_timestamp(), utc_timestamp()),
        )
        connection.execute(
            "UPDATE model_installations SET state='downloading',generation=1,staging_path=?"
            " WHERE manifest_id=?",
            (relative, profile.identity),
        )
    with DataDirectoryLock.acquire(path):
        service.recover()
        assert path.exists()
        assert service.state("minilm-en")["state"] == "failed"
    service.cleanup()
    assert not path.exists()


def test_shutdown_stops_transfer_and_leaves_retryable_state(store: PersistenceStore):
    service = SmallSetup(store, worker_command=worker(delay=30))
    service.begin("minilm-en", "shutdown", 0)
    asyncio.run(service.stop())
    assert service.state("minilm-en")["state"] == "failed"
    assert service.state("minilm-en")["failure_code"] == "model_setup_interrupted"


def test_verification_time_is_inside_setup_deadline(store: PersistenceStore, monkeypatch):
    def slow_verification(path, profile):
        time.sleep(0.6)
        verify_installation(path, profile)

    monkeypatch.setattr("core.search.install.verify_installation", slow_verification)
    service = SmallSetup(store, worker_command=worker(), timeout_seconds=0.5)
    service.begin("minilm-en", "verification-deadline", 0)
    state = wait(service)
    assert state["state"] == "failed"
    assert state["failure_code"] == "model_download_limit"
    assert not installation_path(service.data_dir, service.profile("minilm-en")).exists()
