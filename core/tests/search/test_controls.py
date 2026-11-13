import json
from pathlib import Path

import pytest
from test_storage import seed

from core.jobs.runner import SEARCH_PREPARATION_SECONDS, STEP_POLICIES
from core.persistence.errors import ConflictingWriteError
from core.search.controls import PreparationControl
from core.search.handoff import SearchCoordinator, record_handoff
from core.search.preparation import SearchLimits
from core.search.profiles import profiles

VECTOR = [1.0] + [0.0] * 383


def queued(store):
    notebook, source, _, index, prior, _ = seed(
        store,
        "rainfall",
        VECTOR,
        profile_id="minilm-en",
    )
    store.finish_job(prior, "completed")
    extraction = store.load_index(index).index.extraction_version_id
    with store._transaction() as connection:
        job = record_handoff(connection, extraction)
    assert job is not None
    return notebook, source, job


def test_cancel_and_resume_are_idempotent_and_preserve_old_policy_inputs_and_charges(store):
    _, source, job = queued(store)
    policy = '{"search_prepare":{"timeout_seconds":600,"attempt_cap":2,"backoff_base_seconds":2}}'
    with store._transaction() as connection:
        connection.execute("UPDATE job_status SET policy_json=? WHERE job_id=?", (policy, job))
        connection.execute(
            "UPDATE search_job_inputs SET spent_seconds=43.25 WHERE job_id=?", (job,)
        )
        before = dict(
            connection.execute("SELECT * FROM search_job_inputs WHERE job_id=?", (job,)).fetchone()
        )
    control = PreparationControl(store)
    assert control.cancel(source, job) == control.cancel(source, job) == job
    coordinator = SearchCoordinator(store, Path("unused"))
    coordinator.reconcile()
    assert store.load_job_status(job).status == "cancelled"
    resumed = control.resume(source, job, "same-action")
    assert control.resume(source, job, "same-action") == resumed != job
    with store.database.connect() as connection:
        after = dict(
            connection.execute(
                "SELECT * FROM search_job_inputs WHERE job_id=?", (resumed,)
            ).fetchone()
        )
        assert {k: v for k, v in before.items() if k != "job_id"} == {
            k: v for k, v in after.items() if k != "job_id"
        }
        assert (
            connection.execute(
                "SELECT policy_json FROM job_status WHERE job_id=?", (resumed,)
            ).fetchone()[0]
            == policy
        )
        assert (
            connection.execute(
                "SELECT job_id FROM search_handoffs WHERE job_id=?", (resumed,)
            ).fetchone()[0]
            == resumed
        )
    assert store.load_job_status(job).status == "cancelled"
    assert store.load_job_status(resumed).status == "queued"
    with pytest.raises(ConflictingWriteError):
        control.cancel(source, job)
    with pytest.raises(ConflictingWriteError):
        control.resume(source, job, "different-action")


@pytest.mark.parametrize("change", ["new-version", "model-generation", "budget", "attempts"])
def test_resume_rejects_stale_or_exhausted_work(store, change):
    notebook, source, job = queued(store)
    control = PreparationControl(store)
    control.cancel(source, job)
    if change == "new-version":
        store.create_source_version(source, "b" * 64)
    with store._transaction() as connection:
        if change == "model-generation":
            connection.execute(
                "UPDATE notebook_profiles SET selection_generation=1 WHERE notebook_id=?",
                (notebook,),
            )
        if change == "budget":
            connection.execute(
                "UPDATE search_job_inputs SET spent_seconds=1200 WHERE job_id=?", (job,)
            )
        if change == "attempts":
            connection.execute(
                "UPDATE job_status SET failure_code='embedding_attempt_limit' WHERE job_id=?",
                (job,),
            )
    with pytest.raises(ConflictingWriteError):
        control.resume(source, job, "invalid")
    with store.database.connect() as connection:
        assert not control.can_resume(connection, source, job)


def test_new_default_is_singleton_derived_and_does_not_replace_an_existing_choice(store):
    from core.persistence.models import new_id
    from core.search.handoff import (
        ensure_notebook_profile,
        preparation_limits,
        preparation_representation,
    )
    from core.search.preparation import effective_settings
    from core.search.representation import ARABIC

    notebook, _, _ = queued(store)
    new = store.create_notebook(new_id(), "New")
    with store._transaction() as connection:
        assert (
            ensure_notebook_profile(connection, notebook)["active_manifest_id"]
            == profiles()["minilm-en"].identity
        )
        assert (
            ensure_notebook_profile(connection, new.id)["active_manifest_id"]
            == profiles()["bekko-a8m-quint8-avx2"].identity
        )
    profile = profiles()["bekko-a8m-quint8-avx2"]
    limits = preparation_limits(profile)
    settings = effective_settings(profile, limits, None, preparation_representation(profile))
    assert limits.batch_size == 1 and settings["search_representation"] == ARABIC
    assert (
        settings["total_seconds"]
        == SearchLimits().total_seconds
        == SEARCH_PREPARATION_SECONDS
        == 1200
    )
    assert STEP_POLICIES["search_prepare"].timeout_seconds == 1200
    assert settings["memory_bytes"] == 1024**3
    assert settings["batch_seconds"] == 120 and settings["build_seconds"] == 300
    assert json.loads(json.dumps(settings))["cpu_memory_arena"] is True


@pytest.mark.parametrize("exhaustion", ["time", "attempts"])
def test_resume_checks_the_existing_generation_ledger_before_creating_another_job(
    store, exhaustion
):
    from core.persistence.models import AttemptGuard
    from core.search.chunking import make_chunks
    from core.search.storage import SearchStorage

    _, source, job = queued(store)
    store.claim_job(job, "resume-ledger")
    attempt = store.start_step_attempt(job, "plan-search")
    guard = AttemptGuard("resume-ledger", "plan-search", attempt.attempt_number)
    record = store.load_job_by_id(job)
    report = store.load_extraction_report(record.extraction_version_id)
    profile = profiles()["minilm-en"]
    with store.database.connect() as connection:
        settings = json.loads(
            connection.execute(
                "SELECT settings_json FROM search_job_inputs WHERE job_id=?", (job,)
            ).fetchone()[0]
        )
    chunks = make_chunks(report.passages, profile, lambda _: 1, lambda text: text)
    index = SearchStorage(store).create_generation(job, guard, profile, chunks, settings)
    control = PreparationControl(store)
    control.cancel(source, job)
    with store._transaction() as connection:
        if exhaustion == "time":
            connection.execute(
                "UPDATE search_generations SET spent_seconds=1200 WHERE index_version_id=?",
                (index,),
            )
        else:
            connection.execute(
                "UPDATE search_batches SET attempts=2 WHERE index_version_id=?", (index,)
            )
        count = connection.execute("SELECT count(*) FROM jobs").fetchone()[0]
    with pytest.raises(ConflictingWriteError, match="exhausted"):
        control.resume(source, job, "no-fresh-allowance")
    with store.database.connect() as connection:
        assert not control.can_resume(connection, source, job)
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == count


def test_source_control_routes_share_auth_and_only_offer_current_resumable_work(store):
    from fastapi.testclient import TestClient

    from core.api.app import create_app

    _, source, job = queued(store)
    headers = {"Authorization": "Bearer preparation-test"}
    with TestClient(
        create_app("preparation-test", store=store), base_url="http://127.0.0.1"
    ) as client:
        path = f"/sources/{source}/search/cancel"
        assert client.post(path, json={"job_id": job}).status_code == 401
        assert (
            client.post(
                path, json={"job_id": job}, headers={**headers, "Origin": "https://other.example"}
            ).status_code
            == 403
        )
        assert client.post(path, json={"job_id": job}, headers=headers).status_code == 200
        progress = client.get(f"/sources/{source}/search-progress", headers=headers).json()
        assert progress["status"] == "cancelled" and progress["can_resume"] is True
        assert progress["waiting_reason"] == "search preparation cancelled"
        assert progress["failure_reason"] == "search preparation cancelled"
        store.create_source_version(source, "b" * 64)
        progress = client.get(f"/sources/{source}/search-progress", headers=headers).json()
        assert progress["can_resume"] is False
        assert (
            client.post(
                f"/sources/{source}/search/resume",
                headers=headers,
                json={"job_id": job, "request_key": "stale"},
            ).status_code
            == 409
        )


def test_progress_keeps_fully_embedded_counts_separate_from_publication(store):
    from time import monotonic

    from fastapi.testclient import TestClient

    from core.api.app import create_app
    from core.persistence.models import AttemptGuard
    from core.search.chunking import make_chunks
    from core.search.provenance import runtime_versions
    from core.search.runtime import vector_bytes
    from core.search.storage import SearchStorage

    _, source, job = queued(store)
    store.claim_job(job, "progress-test")
    attempt = store.start_step_attempt(job, "plan-search")
    guard = AttemptGuard("progress-test", "plan-search", attempt.attempt_number)
    report = store.load_extraction_report(store.load_job_by_id(job).extraction_version_id)
    profile = profiles()["minilm-en"]
    with store.database.connect() as connection:
        settings = json.loads(
            connection.execute(
                "SELECT settings_json FROM search_job_inputs WHERE job_id=?", (job,)
            ).fetchone()[0]
        )
    storage = SearchStorage(store)
    index = storage.create_generation(
        job,
        guard,
        profile,
        make_chunks(report.passages, profile, lambda _: 1, lambda text: text),
        settings,
    )
    graph = store.load_index(index)
    store.commit_index_step(
        job, "plan-search", graph.index, graph.chunks, graph.mappings, 1, {}, guard=guard
    )
    store.finish_step_attempt(job, "plan-search", attempt.attempt_number, None)
    attempt = store.start_step_attempt(job, "embed-search")
    guard = AttemptGuard("progress-test", "embed-search", attempt.attempt_number)
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_batches SET attempts=1 WHERE index_version_id=?", (index,)
        )
    storage.record_batch_execution(
        job, guard, index, 0, 1, runtime_versions(), deadline=monotonic() + 20
    )
    storage.commit_batch(
        job,
        guard,
        index,
        0,
        [chunk.id for chunk in graph.chunks],
        [vector_bytes(VECTOR, 384)] * len(graph.chunks),
        execution_attempt=1,
    )
    with TestClient(
        create_app("progress-test", store=store), base_url="http://127.0.0.1"
    ) as client:
        for step in ("embed-search", "build-search", "publish-search"):
            if step != "embed-search":
                store.start_step_attempt(job, step)
            result = client.get(
                f"/sources/{source}/search-progress",
                headers={"Authorization": "Bearer progress-test"},
            ).json()
            assert result["step"] == step and result["status"] == "running"
            assert result["total_chunks"] == result["completed_chunks"] == len(graph.chunks) > 0
        assert store.load_job_status(job).status != "completed"
