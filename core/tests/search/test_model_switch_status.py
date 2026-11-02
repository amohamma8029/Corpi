"""Terminal switch status is a current-source projection, never a retry or reset."""

import pytest
from fastapi.testclient import TestClient
from test_model_switch import pending_jobs
from test_storage import seed

from core.api.app import create_app
from core.search.handoff import invalidate_handoffs
from core.search.profiles import profiles
from core.search.switching import NotebookModelService

VECTOR = [1.0] + [0.0] * 383


def planned(store):
    notebook, source, version, index, _, _ = seed(store, "rainfall", VECTOR, profile_id="minilm-en")
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "status-switch")
    return service, notebook, source, version, index, pending_jobs(store, notebook)[0][0]


def snapshot(store):
    with store.database.connect() as connection:
        return {
            table: [
                tuple(row)
                for row in connection.execute("SELECT * FROM " + table + " ORDER BY rowid")
            ]
            for table in (
                "jobs",
                "job_status",
                "search_job_inputs",
                "search_generations",
                "search_batches",
                "search_handoffs",
                "notebook_profiles",
                "notebook_model_requests",
                "source_readiness",
                "source_readiness_indexes",
            )
        }


@pytest.mark.parametrize("outcome", ["failed", "cancelled", "invalidated"])
def test_terminal_switch_status_preserves_work_and_active_search(store, outcome):
    service, notebook, source, _, index, job = planned(store)
    assert service.state(notebook)["failed_sources"] == 0
    if outcome == "cancelled":
        store.cancel_job(job)
    else:
        store.finish_job(job, "failed", "embedding_worker_failed")
        if outcome == "invalidated":
            with store._transaction() as connection:
                invalidate_handoffs(connection, job_id=job, reason="controlled cancellation")
    before = snapshot(store)
    state = service.state(notebook)
    assert state["failed_sources"] == int(outcome == "failed")
    assert state["cancelled_sources"] == int(outcome != "failed")
    assert state["active_manifest_id"] == profiles()["minilm-en"].identity
    assert state["pending_manifest_id"] == profiles()["multilingual-e5"].identity
    assert state["ready_sources"] == 1
    assert snapshot(store) == before
    assert store.load_readiness(source).index_version_ids == (index,)


@pytest.mark.parametrize(
    "invalidation", ["generation", "profile", "review", "extraction", "trash", "unready"]
)
def test_terminal_switch_status_ignores_noncurrent_work(store, invalidation):
    service, notebook, source, _, _, job = planned(store)
    store.finish_job(job, "failed", "embedding_worker_failed")
    assert service.state(notebook)["failed_sources"] == 1
    if invalidation == "generation":
        service.choose(notebook, "multilingual-e5", 1, "new-current-switch")
    elif invalidation == "trash":
        store.trash_source(source, "9999-01-01T00:00:00.000000Z")
    else:
        replacement = None
        if invalidation == "extraction":
            _, _, _, other_index, _, _ = seed(
                store, "other notebook", VECTOR, profile_id="minilm-en"
            )
            replacement = store.load_index(other_index).index.extraction_version_id
        with store._transaction() as connection:
            if invalidation == "profile":
                connection.execute(
                    "UPDATE search_handoffs SET manifest_id=? WHERE job_id=?",
                    (profiles()["minilm-en"].identity, job),
                )
            elif invalidation == "review":
                connection.execute(
                    "UPDATE search_handoffs SET review_generation=1 WHERE job_id=?", (job,)
                )
            elif invalidation == "extraction":
                connection.execute(
                    "UPDATE search_handoffs SET extraction_version_id=? WHERE job_id=?",
                    (replacement, job),
                )
            else:
                connection.execute(
                    "DELETE FROM source_readiness_indexes WHERE source_id=?", (source,)
                )
    state = service.state(notebook)
    assert state["failed_sources"] == state["cancelled_sources"] == 0


def test_terminal_status_ignores_imports_and_other_notebooks_and_preserves_ready_history(store):
    service, notebook, source, _, index, job = planned(store)
    other, _, _, _, _, _ = seed(store, "other notebook", VECTOR, profile_id="minilm-en")
    service.choose(other, "multilingual-e5", 0, "other-switch")
    store.finish_job(pending_jobs(store, other)[0][0], "failed", "embedding_worker_failed")
    with store._transaction() as connection:
        connection.execute("UPDATE search_handoffs SET purpose='import' WHERE job_id=?", (job,))
    store.finish_job(job, "failed", "embedding_worker_failed")
    assert service.state(notebook)["failed_sources"] == 0
    assert service.state(other)["failed_sources"] == 1
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_handoffs SET purpose='model-switch' WHERE job_id=?", (job,)
        )
    store.create_source_version(source, "b" * 64)
    assert service.state(notebook)["failed_sources"] == 1
    assert store.load_readiness(source).index_version_ids == (index,)


def test_model_api_reports_terminal_status_and_cancel_does_not_retry_or_reset(store):
    service, notebook, _, _, _, job = planned(store)
    store.finish_job(job, "failed", "embedding_worker_failed")
    app = create_app("switch-status-test", store=store, notebook_models=service)
    headers = {"Authorization": "Bearer switch-status-test"}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        path = f"/notebooks/{notebook}/model"
        assert client.get(path).status_code == 401
        state = client.get(path, headers=headers).json()
        assert state["failed_sources"] == 1 and state["cancelled_sources"] == 0
        before = store.load_job_status(job)
        cancelled = client.post(
            path + "/cancel", headers=headers, json={"generation": state["generation"]}
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["pending_profile_id"] is None
        assert cancelled.json()["failed_sources"] == cancelled.json()["cancelled_sources"] == 0
        assert store.load_job_status(job).status == before.status == "failed"
        assert len(pending_jobs(store, notebook)) == 0
