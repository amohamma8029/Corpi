"""Setup actions wait for the actual writer and durable cleanup ownership."""

import asyncio
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient
from test_model_setup import SmallSetup, wait, worker

from core.api.app import create_app
from core.persistence.corelock import DataDirectoryLock
from core.persistence.errors import DatabaseBusyError, IntegrityViolationError, PersistenceError
from core.persistence.models import new_id, utc_timestamp
from core.search import install
from core.search.profiles import installation_path, verify_installation

HEADERS = {"Authorization": "Bearer setup-availability-test"}


def listed(client, profile_id="minilm-en"):
    response = client.get("/models", headers=HEADERS)
    assert response.status_code == 200
    return next(model for model in response.json() if model["profile_id"] == profile_id)


@pytest.fixture
def stopping_writer(store, monkeypatch):
    stopping, release, progressed = (threading.Event() for _ in range(3))
    stopped = []
    original_stop = install.stop_owned_process_tree

    def pause_stop(process):
        stopping.set()
        try:
            assert release.wait(20), "test did not release the owned transfer"
        finally:
            original_stop(process)
            assert process.poll() is not None
            stopped.append(process.pid)

    monkeypatch.setattr(install, "stop_owned_process_tree", pause_stop)
    code = (
        "import sys,json,time; from pathlib import Path; "
        "from core.persistence.corelock import DataDirectoryLock; "
        "p=Path(sys.argv[2]); lock=DataDirectoryLock.acquire(p); "
        "(p/'assets').mkdir(); (p/'assets'/'asset.bin').write_bytes(b'good'); "
        "print(json.dumps({'type':'progress','bytes':4}),flush=True); time.sleep(30)"
    )
    service = SmallSetup(store, worker_command=[sys.executable, "-c", code])
    original_progress = service._progress

    def progress(*args):
        original_progress(*args)
        progressed.set()

    monkeypatch.setattr(service, "_progress", progress)
    try:
        yield service, stopping, release, progressed, stopped
    finally:
        release.set()
        asyncio.run(service.stop())


def test_cancel_response_hides_retry_until_the_owned_writer_stops(store, stopping_writer):
    service, stopping, release, progressed, stopped = stopping_writer
    app = create_app("setup-availability-test", store=store, model_setup=service)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        try:
            started = client.post(
                "/models/minilm-en/setup",
                headers=HEADERS,
                json={"request_key": "cancel-owned-writer", "generation": 0},
            )
            assert started.status_code == 200 and started.json()["can_cancel"]
            assert progressed.wait(10), "owned worker did not report progress"
            cancelled = client.post(
                "/models/minilm-en/setup/cancel", headers=HEADERS, json={"generation": 1}
            )
            assert cancelled.status_code == 200
            assert stopping.wait(10), "owned transfer did not enter stop"
            assert cancelled.json()["state"] == "cancelled"
            assert cancelled.json()["generation"] == 2
            denied = client.post(
                "/models/minilm-en/setup",
                headers=HEADERS,
                json={"request_key": "too-early", "generation": 2},
            )
            assert denied.status_code == 409 and "active" in denied.json()["detail"]
            with store.database.connect() as connection:
                assert connection.execute(
                    "SELECT generation,state,received_bytes,cleanup_pending"
                    " FROM model_setup_attempts"
                ).fetchall()[0][:] == (1, "cancelled", 4, 1)
                assert (
                    connection.execute("SELECT count(*) FROM model_setup_attempts").fetchone()[0]
                    == 1
                )
            assert cancelled.json()["can_retry"] is False
            assert listed(client)["can_retry"] is False
        finally:
            release.set()
            wait(service)
        assert len(stopped) == 1
        assert listed(client)["can_retry"] is True
        service.worker_command = worker()
        retried = client.post(
            "/models/minilm-en/setup",
            headers=HEADERS,
            json={"request_key": "after-owned-stop", "generation": 2},
        )
        assert retried.status_code == 200 and retried.json()["generation"] == 3
        assert wait(service)["state"] == "installed"
        profile = service.profile("minilm-en")
        verify_installation(installation_path(service.data_dir, profile), profile)
        with store.database.connect() as connection:
            attempts = connection.execute(
                "SELECT generation,state,cleanup_pending,spent_seconds FROM model_setup_attempts"
                " ORDER BY rowid"
            ).fetchall()
        assert [tuple(row[:3]) for row in attempts] == [(1, "cancelled", 0), (3, "installed", 0)]
        assert all(row[3] > 0 for row in attempts) and len(stopped) == 2


def test_other_model_download_waits_for_the_single_active_transfer(store, stopping_writer):
    service, _, release, progressed, _ = stopping_writer
    app = create_app("setup-availability-test", store=store, model_setup=service)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        try:
            service.begin("minilm-en", "one-active-transfer", 0)
            assert progressed.wait(10)
            denied = client.post(
                "/models/multilingual-e5/setup",
                headers=HEADERS,
                json={"request_key": "other-active-transfer", "generation": 0},
            )
            assert denied.status_code == 409
            assert listed(client, "multilingual-e5")["state"] == "missing"
            assert listed(client, "multilingual-e5")["can_retry"] is False
        finally:
            service.cancel("minilm-en", 1)
            release.set()
            wait(service)
        assert listed(client, "multilingual-e5")["can_retry"] is True


def test_retry_waits_for_recorded_cleanup_even_without_a_live_service_thread(store):
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
            (
                identity,
                "surviving-owner",
                profile.identity,
                relative,
                utc_timestamp(),
                utc_timestamp(),
            ),
        )
        connection.execute(
            "UPDATE model_installations SET state='downloading',generation=1,staging_path=?"
            " WHERE manifest_id=?",
            (relative, profile.identity),
        )
    app = create_app("setup-availability-test", store=store, model_setup=service)
    with (
        DataDirectoryLock.acquire(path) as surviving_owner,
        TestClient(app, base_url="http://127.0.0.1") as client,
    ):
        assert listed(client)["state"] == "failed"
        denied = client.post(
            "/models/minilm-en/setup",
            headers=HEADERS,
            json={"request_key": "before-cleanup", "generation": 1},
        )
        assert denied.status_code == 409 and "cleanup" in denied.json()["detail"]
        assert listed(client)["can_retry"] is False
        assert (path / "partial.bin").read_bytes() == b"partial"
        surviving_owner.release()
        deadline = time.monotonic() + 10
        while not listed(client)["can_retry"] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert listed(client)["can_retry"] is True and not path.exists()
        with store.database.connect() as connection:
            assert connection.execute(
                "SELECT generation,state,cleanup_pending FROM model_setup_attempts"
            ).fetchone()[:] == (1, "interrupted", 0)
    assert service._cleanup_task is not None and service._cleanup_task.done()


def test_shutdown_waits_for_lifecycle_cleanup_before_returning(store, monkeypatch):
    service = SmallSetup(store)
    entered, release, finished = (threading.Event() for _ in range(3))
    cleanup = service.cleanup

    def paused_cleanup():
        entered.set()
        assert release.wait(10), "test did not release lifecycle cleanup"
        cleanup()
        finished.set()

    async def exercise():
        await service.start()
        monkeypatch.setattr(service, "cleanup", paused_cleanup)
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            stopping = asyncio.create_task(service.stop())
            await asyncio.sleep(0)
            assert not stopping.done() and not finished.is_set()
            release.set()
            await asyncio.wait_for(stopping, 5)
            assert finished.is_set()
            assert service._cleanup_task is not None and service._cleanup_task.done()
        finally:
            release.set()
            await service.stop()

    asyncio.run(exercise())


def test_lifecycle_cleanup_retries_a_transient_database_busy_error(store, monkeypatch):
    service = SmallSetup(store)
    finished = threading.Event()
    cleanup = service.cleanup
    calls = 0

    def contended_cleanup():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DatabaseBusyError("controlled transient writer contention")
        cleanup()
        finished.set()

    async def exercise():
        await service.start()
        monkeypatch.setattr(service, "cleanup", contended_cleanup)
        try:
            assert await asyncio.to_thread(finished.wait, 5)
            assert calls >= 2
        finally:
            await service.stop()
        assert service._cleanup_task is not None and service._cleanup_task.done()

    asyncio.run(exercise())


@pytest.mark.parametrize("error_type", [IntegrityViolationError, PersistenceError])
def test_lifecycle_cleanup_does_not_retry_non_busy_persistence_failures(
    store,
    monkeypatch,
    error_type,
):
    service = SmallSetup(store)
    calls = 0

    def failed_cleanup():
        nonlocal calls
        calls += 1
        raise error_type("controlled non-transient persistence failure")

    async def exercise():
        await service.start()
        cleanup_task = service._cleanup_task
        assert cleanup_task is not None
        monkeypatch.setattr(service, "cleanup", failed_cleanup)
        try:
            with pytest.raises(error_type, match="non-transient"):
                await asyncio.wait_for(asyncio.shield(cleanup_task), 5)
            assert calls == 1
        finally:
            with pytest.raises(error_type, match="non-transient"):
                await service.stop()
        assert calls == 1

    asyncio.run(exercise())
