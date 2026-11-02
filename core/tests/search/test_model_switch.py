from pathlib import Path

import pytest
from test_storage import seed

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import AttemptGuard, utc_timestamp
from core.persistence.store import PersistenceStore
from core.search.chunking import make_chunks
from core.search.handoff import SearchCoordinator, invalidate_handoffs
from core.search.profiles import profiles
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage
from core.search.switching import NotebookModelService, publish_replacement


def readiness(store, source):
    value = store.load_readiness(source)
    assert value is not None
    return value


def status(store, job):
    value = store.load_job_status(job)
    assert value is not None
    return value


def mark_installed(store, profile_id="multilingual-e5"):
    profile = profiles()[profile_id]
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO model_installations(manifest_id,profile_id,state,updated_at)"
            " VALUES(?,?,'installed',?) ON CONFLICT(manifest_id) DO UPDATE SET state='installed'",
            (profile.identity, profile.id, utc_timestamp()),
        )


def pending_jobs(store, notebook):
    with store.database.connect() as connection:
        return connection.execute(
            "SELECT j.id FROM search_handoffs h JOIN jobs j ON j.id=h.job_id"
            " JOIN sources s ON s.id=j.source_id WHERE s.notebook_id=?"
            " AND h.purpose='model-switch' AND h.state='scheduled' ORDER BY j.created_at",
            (notebook,),
        ).fetchall()


def complete(store, job_id):
    job = store.load_job_by_id(job_id)
    store.claim_job(job_id, "switch-tests")
    attempt = store.start_step_attempt(job_id, "replace")
    guard = AttemptGuard("switch-tests", "replace", attempt.attempt_number)
    profile = profiles()["multilingual-e5"]
    report = store.load_extraction_report(job.extraction_version_id)
    chunks = make_chunks(report.passages, profile, lambda _: 1, lambda text: text)
    storage = SearchStorage(store)
    index = storage.create_generation(
        job_id, guard, profile, chunks, {"batch_size": 32, "separator": profile.separator}
    )
    graph = store.load_index(index)
    storage.commit_batch(
        job_id,
        guard,
        index,
        0,
        [chunk.id for chunk in graph.chunks],
        [vector_bytes([1.0] + [0.0] * 383, 384) for _ in graph.chunks],
    )
    storage.finish_components(job_id, guard, index)
    store.commit_index_step(
        job_id, "replace", graph.index, graph.chunks, graph.mappings, 1, {}, guard=guard
    )
    publish_replacement(store, job_id, guard, index)
    return index, guard


def test_model_switch_keeps_old_search_until_every_ready_source_finishes(store: PersistenceStore):
    notebook, source_a, version_a, old_a, _, _ = seed(
        store, "First rainfall", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    _, source_b, version_b, old_b, _, _ = seed(
        store, "Second rainfall", [1.0] + [0.0] * 383, notebook_id=notebook, profile_id="minilm-en"
    )
    mark_installed(store)
    service = NotebookModelService(store)
    state = service.choose(notebook, "multilingual-e5", 0, "switch-one")
    assert state["active_manifest_id"] == profiles()["minilm-en"].identity
    jobs = pending_jobs(store, notebook)
    assert len(jobs) == 2
    complete(store, jobs[0][0])
    assert readiness(store, source_a).index_version_ids == (old_a,)
    assert readiness(store, source_b).index_version_ids == (old_b,)
    assert SearchStorage(store).search(notebook, [version_a, version_b], query="rainfall")
    complete(store, jobs[1][0])
    state = service.state(notebook)
    assert state["pending_manifest_id"] is None
    assert state["active_manifest_id"] == profiles()["multilingual-e5"].identity
    assert readiness(store, source_a).index_version_ids != (old_a,)
    assert readiness(store, source_b).index_version_ids != (old_b,)
    hits = SearchStorage(store).search(notebook, [version_a, version_b], query="rainfall")
    assert hits and all(hit.manifest_id == profiles()["multilingual-e5"].identity for hit in hits)


def test_cancelled_selection_does_not_hide_old_readiness_or_activate_late_output(
    store: PersistenceStore,
):
    notebook, source, _, old, _, _ = seed(
        store, "rainfall", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "cancel-selection")
    job = pending_jobs(store, notebook)[0][0]
    cancelled = service.cancel(notebook, 1)
    assert cancelled["pending_manifest_id"] is None
    assert status(store, job).status == "cancelled"
    assert readiness(store, source).index_version_ids == (old,)
    with pytest.raises(ConflictingWriteError):
        store.claim_job(job, "late")


def test_explicit_choice_is_idempotent_and_missing_model_waits(store: PersistenceStore):
    notebook, source, _, old, _, _ = seed(
        store, "rainfall", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    service = NotebookModelService(store)
    first = service.choose(notebook, "multilingual-e5", 0, "same-choice")
    repeated = service.choose(notebook, "multilingual-e5", 0, "same-choice")
    assert first["selection_generation"] == repeated["selection_generation"] == 1
    jobs = pending_jobs(store, notebook)
    assert len(jobs) == 1
    job = store.load_job_by_id(jobs[0][0])
    assert (
        SearchCoordinator(store, Path("unused")).claim_blocking_reason(job) == "model_not_installed"
    )
    assert readiness(store, source).index_version_ids == (old,)


def test_switch_targets_old_ready_pdf_while_a_new_replacement_is_pending(store: PersistenceStore):
    notebook, source, old_version, _, _, _ = seed(
        store, "rainfall", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    store.create_source_version(source, "b" * 64)
    mark_installed(store)
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "ready-history")
    job = store.load_job_by_id(pending_jobs(store, notebook)[0][0])
    assert job.source_version_id == old_version
    assert SearchCoordinator(store, Path("unused")).claim_blocking_reason(job) is None
    complete(store, job.id)
    assert readiness(store, source).source_version_id == old_version


def test_invalidated_completed_publication_cannot_count_toward_activation(store: PersistenceStore):
    notebook, source_a, _, old_a, _, _ = seed(
        store, "First rainfall", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    _, source_b, _, old_b, _, _ = seed(
        store, "Second rainfall", [1.0] + [0.0] * 383, notebook_id=notebook, profile_id="minilm-en"
    )
    mark_installed(store)
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "invalidated-publication")
    jobs = pending_jobs(store, notebook)
    complete(store, jobs[0][0])
    store.finish_job(jobs[0][0], "completed")
    with store._transaction() as connection:
        invalidate_handoffs(connection, job_id=jobs[0][0], reason="review reopened")
    assert status(store, jobs[0][0]).status == "completed"
    complete(store, jobs[1][0])
    assert service.state(notebook)["pending_manifest_id"] == profiles()["multilingual-e5"].identity
    assert readiness(store, source_a).index_version_ids == (old_a,)
    assert readiness(store, source_b).index_version_ids == (old_b,)
