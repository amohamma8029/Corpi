"""Search-specific review/publication/restart controls on disk,with controlled vectors."""

from pathlib import Path

import pytest
from test_model_switch import complete, mark_installed, pending_jobs, readiness
from test_search_expiry import reopen
from test_storage import seed

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import (
    AttemptGuard,
    ExtractionOutcome,
    ExtractionPage,
    ExtractionReview,
    ExtractionVersion,
    Passage,
    new_id,
    utc_timestamp,
)
from core.persistence.store import PersistenceStore
from core.search.chunking import eligible_passages, make_chunks
from core.search.handoff import SearchCoordinator
from core.search.profiles import profiles
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage, verify_generation
from core.search.switching import NotebookModelService, publish_replacement

VALUES = [1.0] + [0.0] * 383
VECTOR = vector_bytes(VALUES, 384)


def native_hits(store, notebook, versions):
    search = SearchStorage(store)
    manifest = NotebookModelService(store).state(notebook)["active_manifest_id"]
    word = search.search(notebook, versions, query="rainfall")
    vector = search.search(notebook, versions, vector=VECTOR, vector_manifest_id=manifest)
    return word, vector


def prepare(store, job_id, *, profile_id="minilm-en"):
    job = store.load_job_by_id(job_id)
    store.claim_job(job_id, "safety-tests")
    attempt = store.start_step_attempt(job_id, "prepare")
    guard = AttemptGuard("safety-tests", "prepare", attempt.attempt_number)
    profile = profiles()[profile_id]
    report = store.load_extraction_report(job.extraction_version_id)
    chunks = make_chunks(eligible_passages(report), profile, lambda _: 1, lambda text: text)
    storage = SearchStorage(store)
    index = storage.create_generation(
        job_id, guard, profile, chunks, {"batch_size": 32, "separator": profile.separator}
    )
    graph = store.load_index(index)
    storage.commit_batch(
        job_id, guard, index, 0, [c.id for c in graph.chunks], [VECTOR for _ in graph.chunks]
    )
    storage.finish_components(job_id, guard, index)
    store.commit_index_step(
        job_id, "prepare", graph.index, graph.chunks, graph.mappings, 1, {}, guard=guard
    )
    return index, guard


def partial_source(store):
    notebook = store.create_notebook(new_id(), "Partial review")
    with store._transaction() as connection:
        connection.execute(
            "INSERT INTO notebook_profiles(notebook_id,active_manifest_id,updated_at)"
            " VALUES(?,?,?)",
            (notebook.id, profiles()["minilm-en"].identity, utc_timestamp()),
        )
    source = store.create_source(new_id(), notebook.id, "partial.pdf")
    version = store.create_source_version(source.id, "c" * 64)
    extraction = ExtractionVersion(
        new_id(), source.id, version.id, "control", "1", "1", 1, {}, utc_timestamp()
    )
    texts = ("rainfall eligible evidence", "rainfall suspect secret", "rainfall visual secret")
    passages = [
        Passage(
            new_id(),
            source.id,
            version.id,
            extraction.id,
            i,
            text,
            1,
            {"page": i + 1, "reading_order": 0},
            utc_timestamp(),
        )
        for i, text in enumerate(texts)
    ]
    pages = [
        ExtractionPage(
            new_id(),
            source.id,
            version.id,
            extraction.id,
            i + 1,
            True,
            len(texts[i]),
            0,
            0,
            classification,
            utc_timestamp(),
        )
        for i, classification in enumerate(("extracted", "suspect", "unreadable"))
    ]
    review = ExtractionReview(
        extraction.id, source.id, version.id, "pending", 1, None, utc_timestamp()
    )
    store.save_extraction_result(
        extraction,
        passages,
        pages,
        [],
        ExtractionOutcome(extraction.id, source.id, version.id, "partial", utc_timestamp()),
        review,
    )
    store.record_review_decision(extraction.id, "confirmed", 1)
    with store.database.connect() as connection:
        job = connection.execute(
            "SELECT job_id FROM search_handoffs WHERE extraction_version_id=?", (extraction.id,)
        ).fetchone()[0]
    return notebook.id, source.id, version.id, extraction.id, passages, job


@pytest.mark.parametrize("decision", ["pending", "declined", "confirmed"])
def test_published_review_again_hides_native_hits_and_never_reuses_stale_review(
    store: PersistenceStore, decision
):
    notebook, source, version, extraction, passages, job = partial_source(store)
    index, guard = prepare(store, job)
    store.publish_readiness(job, source, version, [index], guard=guard)
    assert all(native_hits(store, notebook, [version]))
    graph = store.load_index(index)
    assert {m.passage_id for m in graph.mappings} == {passages[0].id}
    assert [p.text for p in store.load_extraction_report(extraction).passages] == [
        p.text for p in passages
    ]
    store.reopen_review(extraction)
    with pytest.raises(ConflictingWriteError, match="stale"):
        store.record_review_decision(extraction, "confirmed", 1)
    if decision != "pending":
        store.record_review_decision(extraction, decision, 2)
    restarted = reopen(store)
    SearchCoordinator(restarted, Path("unused")).reconcile()
    assert native_hits(restarted, notebook, [version]) == ((), ())
    with pytest.raises(ConflictingWriteError):
        restarted.publish_readiness(job, source, version, [index], guard=guard)
    assert restarted.load_index(index).chunks == graph.chunks
    if decision == "confirmed":
        with restarted.database.connect() as connection:
            fresh = connection.execute(
                "SELECT job_id FROM search_handoffs WHERE extraction_version_id=?"
                " AND review_generation=2",
                (extraction,),
            ).fetchone()[0]
        fresh_index, fresh_guard = prepare(restarted, fresh)
        restarted.publish_readiness(fresh, source, version, [fresh_index], guard=fresh_guard)
        assert all(native_hits(restarted, notebook, [version]))
        # Immutable extraction/profile/settings can reuse vectors after an explicit
        # current review. The old JOB remains invalid;new publication owns eligibility.
        assert fresh_index == index
        assert readiness(restarted, source).published_by_job_id == fresh


@pytest.mark.parametrize("action", ["cancel", "trash"])
@pytest.mark.parametrize("publication_first", [False, True])
def test_both_serialized_publication_invalidation_orderings_preserve_native_isolation(
    store: PersistenceStore, action, publication_first
):
    notebook, source, version, index, job, guard = seed(
        store, "rainfall target", VALUES, profile_id="minilm-en"
    )
    _, neighbour, neighbour_version, _, _, _ = seed(
        store, "rainfall neighbour", VALUES, notebook_id=notebook, profile_id="minilm-en"
    )
    store.remove_readiness(source)
    if publication_first:
        store.publish_readiness(job, source, version, [index], guard=guard)
        assert all(native_hits(store, notebook, [version]))
    if action == "cancel":
        store.cancel_job(job)
    else:
        store.trash_source(source, "9999-01-01T00:00:00.000000Z")
    with pytest.raises(ConflictingWriteError):
        store.publish_readiness(job, source, version, [index], guard=guard)
    restarted = reopen(store)
    assert native_hits(restarted, notebook, [version]) == ((), ())
    assert all(native_hits(restarted, notebook, [neighbour_version]))
    assert all(
        all(hit.source_id == neighbour for hit in method)
        for method in native_hits(restarted, notebook, [neighbour_version])
    )
    with restarted.database.connect() as connection:
        verify_generation(connection, index)  # Hidden,not silently discarded or corrupted.


def test_restart_during_switch_preserves_old_hits_and_only_activates_complete_source_set(store):
    notebook, source_a, version_a, old_a, _, _ = seed(
        store, "rainfall A", VALUES, profile_id="minilm-en"
    )
    _, source_b, version_b, old_b, _, _ = seed(
        store, "rainfall B", VALUES, notebook_id=notebook, profile_id="minilm-en"
    )
    mark_installed(store)
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "restart-switch")
    jobs = [r[0] for r in pending_jobs(store, notebook)]
    new_a, _ = complete(store, jobs[0])
    restarted = reopen(store)
    restarted.release_running_jobs()
    SearchCoordinator(restarted, Path("unused")).reconcile()
    assert [r[0] for r in pending_jobs(restarted, notebook)] == jobs
    assert readiness(restarted, source_a).index_version_ids == (old_a,)
    assert readiness(restarted, source_b).index_version_ids == (old_b,)
    assert all(native_hits(restarted, notebook, [version_a, version_b]))
    complete(restarted, jobs[1])
    # Publication before crash is durable,but its interrupted job is queued until
    # reclaimed. It must not count as running/completed during that interval.
    assert NotebookModelService(restarted).state(notebook)["pending_manifest_id"] is not None
    complete(restarted, jobs[0])
    assert NotebookModelService(restarted).state(notebook)["pending_manifest_id"] is None
    assert readiness(restarted, source_a).index_version_ids == (new_a,)
    assert all(native_hits(restarted, notebook, [version_a, version_b]))
    assert all(
        all(h.manifest_id == profiles()["multilingual-e5"].identity for h in method)
        for method in native_hits(restarted, notebook, [version_a, version_b])
    )


def test_new_choice_rejects_previous_completed_switch_publication_after_restart(store):
    notebook, source, version, old, _, _ = seed(store, "rainfall A", VALUES, profile_id="minilm-en")
    seed(store, "rainfall B", VALUES, notebook_id=notebook, profile_id="minilm-en")
    mark_installed(store)
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "first-switch")
    job = pending_jobs(store, notebook)[0][0]
    index, guard = complete(store, job)
    service.choose(notebook, "minilm-en", 1, "new-choice")
    restarted = reopen(store)
    SearchCoordinator(restarted, Path("unused")).reconcile()
    with pytest.raises(ConflictingWriteError):
        publish_replacement(restarted, job, guard, index)
    assert readiness(restarted, source).index_version_ids == (old,)
    assert all(native_hits(restarted, notebook, [version]))
    assert (
        NotebookModelService(restarted).state(notebook)["active_manifest_id"]
        == profiles()["minilm-en"].identity
    )


def test_source_added_during_switch_is_rebuilt_before_activation(store):
    notebook, _, _, _, _, _ = seed(store, "rainfall A", VALUES, profile_id="minilm-en")
    seed(store, "rainfall B", VALUES, notebook_id=notebook, profile_id="minilm-en")
    mark_installed(store)
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "growing-set")
    original = [r[0] for r in pending_jobs(store, notebook)]
    complete(store, original[0])
    _, added, version, old, _, _ = seed(
        store, "rainfall added", VALUES, notebook_id=notebook, profile_id="minilm-en"
    )
    SearchCoordinator(store, Path("unused")).reconcile()
    jobs = [r[0] for r in pending_jobs(store, notebook)]
    assert len(jobs) == 3
    complete(store, original[1])
    assert service.state(notebook)["pending_manifest_id"] is not None
    assert store.load_readiness(added).index_version_ids == (old,)
    complete(store, next(job for job in jobs if job not in original))
    assert service.state(notebook)["pending_manifest_id"] is None
    assert all(native_hits(store, notebook, [version]))
