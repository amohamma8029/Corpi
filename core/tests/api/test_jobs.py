"""Job-status API: read-only views under the existing auth middleware."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from core.api.app import create_app
from core.jobs.runner import JobRunner, StepContext, StepDefinition, StepOutcome
from core.persistence.database import Database, DatabaseConfig
from core.persistence.models import new_id
from core.persistence.store import PersistenceStore

TOKEN = "jobs-api-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_store(tmp_path: Path) -> PersistenceStore:
    database = Database(DatabaseConfig(tmp_path / "data"))
    database.initialize()
    return PersistenceStore(database)


def make_client(store: PersistenceStore) -> TestClient:
    app = create_app(token=TOKEN, store=store)
    return TestClient(app, base_url="http://127.0.0.1")


def seed_job(store: PersistenceStore) -> str:
    notebook_id = new_id()
    store.create_notebook(notebook_id, "Jobs")
    source_id = new_id()
    store.create_source(source_id, notebook_id, "jobs.pdf")
    version = store.create_source_version(source_id, "a" * 64, None)
    job = store.create_job("jobs:api", "demo", source_id, source_version_id=version.id)
    store.create_job_status(job.id)
    return job.id


def test_jobs_require_authentication(tmp_path: Path) -> None:
    client = make_client(make_store(tmp_path))
    assert client.get("/jobs").status_code == 401
    assert client.get(f"/jobs/{new_id()}").status_code == 401


def test_job_routes_report_an_unavailable_store(tmp_path: Path) -> None:
    app = create_app(token=TOKEN)
    client = TestClient(app, base_url="http://127.0.0.1")
    assert client.get("/jobs", headers=AUTH).status_code == 503
    assert client.get(f"/jobs/{new_id()}", headers=AUTH).status_code == 503


def test_list_and_detail_expose_state_and_progress(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    job_id = seed_job(store)
    store.claim_job(job_id, "core-a")
    store.start_step_attempt(job_id, "extract-text")
    client = make_client(store)

    listed = client.get("/jobs", headers=AUTH)
    assert listed.status_code == 200
    entries = listed.json()
    assert any(
        entry["job_id"] == job_id and entry["kind"] == "demo" and entry["status"] == "running"
        for entry in entries
    ), entries

    detail = client.get(f"/jobs/{job_id}", headers=AUTH)
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["job"]["status"] == "running"
    assert payload["job"]["current_step_key"] == "extract-text"
    assert payload["job"]["attempt_count"] == 1
    assert [step["step_key"] for step in payload["steps"]] == []
    attempts = payload["attempts"]
    assert [(item["step_key"], item["attempt_number"]) for item in attempts] == [
        ("extract-text", 1)
    ]

    missing = client.get(f"/jobs/{new_id()}", headers=AUTH)
    assert missing.status_code == 404


def test_api_stays_responsive_while_a_job_runs(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    release = threading.Event()

    def slow(context: StepContext) -> StepOutcome:
        release.wait(10)
        return StepOutcome(1, {"slow": True})

    runner = JobRunner(store, {"demo": [StepDefinition("slow", slow)]}, pool_size=1)
    app = create_app(token=TOKEN, store=store, runner=runner)
    job_id = seed_job(store)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        runner.notify()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = store.load_job_status(job_id)
            if status is not None and status.current_step_key == "slow":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("job did not start")

        started = time.monotonic()
        health = client.get("/health", headers=AUTH)
        elapsed = time.monotonic() - started
        assert health.status_code == 200
        assert elapsed < 1.0, f"request took {elapsed:.2f}s while a job was running"

        listed = client.get("/jobs", headers=AUTH).json()
        assert any(item["job_id"] == job_id and item["status"] == "running" for item in listed)

        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = store.load_job_status(job_id)
            if status is not None and status.status == "completed":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("job did not finish")


def test_health_stays_fast_while_the_database_writer_is_locked(tmp_path: Path) -> None:
    """The runner's database work runs on its own capacity, so a held writer
    lock never freezes the event loop that serves API requests."""
    store = make_store(tmp_path)
    runner = JobRunner(
        store, {"demo": [StepDefinition("quick", lambda context: StepOutcome(1, {}))]}
    )
    app = create_app(token=TOKEN, store=store, runner=runner)
    job_id = seed_job(store)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        holder = sqlite3.connect(store.database.config.database_path)
        holder.execute("BEGIN IMMEDIATE")
        try:
            runner.notify()  # the runner's claim blocks on the writer lock
            time.sleep(0.3)
            started = time.monotonic()
            health = client.get("/health", headers=AUTH)
            elapsed = time.monotonic() - started
            assert health.status_code == 200
            assert elapsed < 0.25, (
                f"/health took {elapsed:.2f}s while the database writer lock was held"
            )
        finally:
            holder.rollback()
            holder.close()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = store.load_job_status(job_id)
            if status is not None and status.status == "completed":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("the runner did not resume after the lock was released")
