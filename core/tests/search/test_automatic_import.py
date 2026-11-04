import time
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import native_text_pdf

from core.api.app import create_app
from core.ingest.service import PdfIngestService
from core.jobs.runner import JobRunner, RunnerSnapshot
from core.jobs.steps import build_pdf_extract_steps
from core.persistence.models import new_id, utc_timestamp
from core.persistence.store import PersistenceStore
from core.search.handoff import DEFAULT_PROFILE_ID, SearchCoordinator
from core.search.preparation import build_search_steps
from core.search.profiles import profiles, verify_installation
from core.search.query import QueryService
from core.search.storage import SearchStorage
from core.search.switching import NotebookModelService


def readiness_failure_state(
    store: PersistenceStore, source_id: str, snapshot: RunnerSnapshot
) -> dict:
    """Report source-scoped processing state, not PDF text, vectors or environment."""
    with store.database.connect() as connection:
        jobs = [
            dict(row)
            for row in connection.execute(
                "SELECT j.kind,s.status,s.current_step_key,s.attempt_count,"
                "s.failure_code FROM jobs j"
                " JOIN job_status s ON s.job_id=j.id WHERE j.source_id=?"
                " ORDER BY j.created_at,j.id",
                (source_id,),
            )
        ]
    return {"runner": asdict(snapshot), "jobs": jobs}


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_readiness_failure_reports_source_job_state_without_mutation(
    store: PersistenceStore, status: str
) -> None:
    notebook = store.create_notebook(new_id(), "Readiness diagnostics")
    source = store.create_source(new_id(), notebook.id, "diagnostic.pdf")
    job = store.create_job("diagnostic:target", "pdf-search", source.id)
    store.create_job_status(job.id)
    other = store.create_source(new_id(), notebook.id, "unrelated.pdf")
    unrelated = store.create_job("diagnostic:unrelated", "pdf-extract", other.id)
    store.create_job_status(unrelated.id)
    code = "embedding_worker_failed" if status == "failed" else None
    with store._transaction() as connection:
        connection.execute(
            "UPDATE job_status SET status=?,current_step_key='embed-search',"
            "attempt_count=1,failure_code=?,failure_reason=? WHERE job_id=?",
            (status, code, "source text must not be exported", job.id),
        )
    before = store.load_job_status(job.id)
    other_before = store.load_job_status(unrelated.id)
    snapshot = RunnerSnapshot("idle", None, None, (), ())
    state = readiness_failure_state(store, source.id, snapshot)
    assert state == {
        "runner": asdict(snapshot),
        "jobs": [
            {
                "kind": "pdf-search",
                "status": status,
                "current_step_key": "embed-search",
                "attempt_count": 1,
                "failure_code": code,
            }
        ],
    }
    assert store.load_job_status(job.id) == before
    assert store.load_job_status(unrelated.id) == other_before


def test_real_import_automatically_reaches_searchable_readiness(
    store: PersistenceStore,
    request: pytest.FixtureRequest,
) -> None:
    directory = request.config.getoption("--models-directory", default=None)
    if directory is None:
        pytest.skip("explicit installed-model directory required; no implicit downloads")
    models = Path(directory)
    profile = profiles()[DEFAULT_PROFILE_ID]
    verify_installation(models / profile.identity, profile)
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO model_installations(manifest_id,profile_id,state,updated_at)"
            " VALUES(?,?,'installed',?)",
            (profile.identity, profile.id, utc_timestamp()),
        )
    coordinator = SearchCoordinator(store, models)
    runner = JobRunner(
        store,
        {
            "pdf-extract": build_pdf_extract_steps(store),
            "pdf-search": build_search_steps(store, models),
        },
        reconcile=coordinator.reconcile,
        claim_blocking_reason=coordinator.claim_blocking_reason,
    )
    ingest = PdfIngestService(store, "automatic-import-test", notify_jobs=runner.notify)
    notebook_models = NotebookModelService(store, notify=runner.notify)
    app = create_app(
        "automatic-import-token",
        store=store,
        pdf_ingest=ingest,
        runner=runner,
        notebook_models=notebook_models,
        search_queries=QueryService(store, models),
    )
    notebook = store.create_notebook(new_id(), "Automatic import")
    headers = {"Authorization": "Bearer automatic-import-token"}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        created = client.post(
            f"/notebooks/{notebook.id}/pdf-acquisitions",
            headers=headers,
            json={"retry_key": new_id(), "filename": "rainfall.pdf"},
        )
        created.raise_for_status()
        uploaded = client.put(
            f"/pdf-acquisitions/{created.json()['id']}/content",
            headers={**headers, "Content-Type": "application/pdf"},
            content=native_text_pdf(
                "The experiment measured rainfall and soil moisture across study sites."
            ),
        )
        uploaded.raise_for_status()
        source = uploaded.json()["source_id"]
        deadline = time.monotonic() + 60
        ready = None
        while time.monotonic() < deadline:
            ready = store.load_readiness(source)
            if ready is not None:
                break
            time.sleep(0.05)
        assert ready is not None, readiness_failure_state(store, source, runner.snapshot())
        response = client.post(
            f"/notebooks/{notebook.id}/search",
            headers=headers,
            json={"query": "soil moisture", "mode": "words"},
        )
        response.raise_for_status()
        hits = response.json()["results"]
        assert hits and hits[0]["source_id"] == source and hits[0]["citations"]
        with store.database.connect() as connection:
            assert connection.execute("SELECT count(*) FROM search_handoffs").fetchone()[0] == 1
            assert (
                connection.execute("SELECT count(*) FROM jobs WHERE kind='pdf-search'").fetchone()[
                    0
                ]
                == 1
            )
        alternative = profiles()["multilingual-e5"]
        verify_installation(models / alternative.identity, alternative)
        with store._transaction() as connection:
            connection.execute(
                "INSERT INTO model_installations(manifest_id,profile_id,state,updated_at)"
                " VALUES(?,?,'installed',?)",
                (alternative.identity, alternative.id, utc_timestamp()),
            )
        selected = client.post(
            f"/notebooks/{notebook.id}/model",
            headers=headers,
            json={"profile_id": alternative.id, "generation": 0, "request_key": new_id()},
        )
        selected.raise_for_status()
        deadline = time.monotonic() + 60
        status = notebook_models.state(notebook.id)
        while time.monotonic() < deadline:
            status = notebook_models.state(notebook.id)
            if status["pending_manifest_id"] is None:
                break
            time.sleep(0.05)
        assert status["active_manifest_id"] == alternative.identity
        new_ready = store.load_readiness(source)
        assert new_ready is not None and new_ready.source_version_id == ready.source_version_id
        hits = SearchStorage(store).search(
            notebook.id, [new_ready.source_version_id], query="soil moisture"
        )
        assert hits and all(hit.manifest_id == alternative.identity for hit in hits)
