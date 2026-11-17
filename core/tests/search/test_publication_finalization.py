"""Post-COMMIT failures must not undo accepted real-PDF replacement eligibility.

The PDF/API/extraction, runner, coordinator and native rows are real. The model
session returns controlled vectors: this is a lifecycle test, not model evidence.
On 4139df1 the fault cases finish as failed after displacing working readiness.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

import pytest
from pdf_fixtures import native_text_pdf
from test_import_expiry import IndexedImport
from test_model_switch import mark_installed, pending_jobs
from test_preparation_accounting import Clock
from test_replacement_safety import extract_candidate, upload
from test_replacement_safety import original as original
from test_search_expiry import reopen

from core.jobs import runner as runner_module
from core.jobs.runner import (
    JobRunner,
    StepCommitContext,
    StepContext,
    StepDefinition,
    _canonical_json,
)
from core.persistence.errors import ConflictingWriteError, DatabaseBusyError
from core.persistence.models import AttemptGuard, JSONValue
from core.search import preparation, storage, switching
from core.search.handoff import SearchCoordinator
from core.search.provenance import runtime_versions
from core.search.query import QueryService
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage
from core.search.switching import NotebookModelService

VECTOR = vector_bytes([1.0] + [0.0] * 383, 384)


@pytest.fixture
def publication_runtime(monkeypatch):
    class Tokenizer:
        def __init__(self, *_):
            pass

        def count(self, _):
            return 1

        def prepare(self, text):
            return text

    class Session:
        load_seconds = 0
        peak_bytes = 1024
        execution_runtime = runtime_versions()
        document_calls: list[tuple[str, ...]] = []

        def __init__(self, path, profile, **options):
            self.profile = profile

        def embed(self, texts, *, query=False, **options):
            if not query:
                self.document_calls.append(tuple(texts))
            return [VECTOR for _ in texts]

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    monkeypatch.setattr(preparation, "InputTokenizer", Tokenizer)
    monkeypatch.setattr(preparation, "EmbeddingSession", Session)
    return Session


def freeze_clock(monkeypatch):
    from core.persistence import store as persistence

    clock = Clock()
    for module in (preparation, runner_module, storage, switching, persistence):
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=clock.monotonic))
    return clock


@dataclass(frozen=True)
class Publication:
    notebook: str
    source: str
    version: str
    import_id: str
    job: str
    index: str
    profile_id: str
    steps: tuple[StepDefinition, ...]
    old: IndexedImport
    pdf_bytes: bytes


def prepare_publication(store, original, kind):
    notebook, client, old = original
    mark_installed(store, "minilm-en")
    if kind == "replacement":
        content = native_text_pdf("Replacement rainfall evidence remains linked to its saved PDF.")
        import_id = upload(client, notebook, content, old.item.source_id, filename="new.pdf")
        item = store.load_pdf_import(import_id)
        job = extract_candidate(store, import_id)
        profile_id = "minilm-en"
    else:
        mark_installed(store)
        NotebookModelService(store).choose(notebook, "multilingual-e5", 0, "finalization-switch")
        job = pending_jobs(store, notebook)[0][0]
        item = store.load_pdf_import(old.item.id)
        import_id = item.id
        profile_id = "multilingual-e5"
        content = old.content
    policy: dict[str, JSONValue] = {
        key: {
            "timeout_seconds": seconds,
            "attempt_cap": 2,
            "backoff_base_seconds": 2,
        }
        for key, seconds in (("index_build", 300), ("search_prepare", 1200))
    }
    store.claim_job(job, "publication-seed", policy=policy)
    steps = preparation.build_search_steps(store, store.database.config.data_dir / "models")
    for step in steps[:3]:
        attempt = store.start_step_attempt(job, step.step_key)
        guard = AttemptGuard("publication-seed", step.step_key, attempt.attempt_number)
        context = StepContext(
            job, step.step_key, attempt.attempt_number, lambda: False, lambda: False, guard
        )
        assert step.inputs is not None
        fingerprint = sha256(_canonical_json(step.inputs(context)).encode()).hexdigest()
        step.commit(
            store, StepCommitContext(job, step.step_key, guard, fingerprint, step.run(context))
        )
        store.finish_step_attempt(job, step.step_key, attempt.attempt_number, None)
    index = next(
        step.output_index_version_id
        for step in store.load_job_steps(job)
        if step.step_key == preparation.PLAN
    )
    assert index is not None
    store.release_running_jobs()
    return Publication(
        notebook,
        item.source_id,
        item.source_version_id,
        import_id,
        job,
        index,
        profile_id,
        steps,
        old,
        content,
    )


def protected_output(store, case):
    """Exact identities, vectors, mappings, batch attempts and saved inputs/policy."""
    with store.database.connect() as connection:
        queries = {
            "generation": (
                "SELECT index_version_id,manifest_id,identity_sha256,settings_json,expected_chunks "
                "FROM search_generations WHERE index_version_id=?"
            ),
            "chunks": "SELECT * FROM retrieval_chunks WHERE index_version_id=? ORDER BY ordinal",
            "mappings": "SELECT * FROM chunk_passages WHERE index_version_id=? ORDER BY rowid",
            "batches": "SELECT * FROM search_batches WHERE index_version_id=? ORDER BY ordinal",
            "executions": (
                "SELECT * FROM search_batch_executions WHERE index_version_id=? ORDER BY ordinal"
            ),
            "vectors": (
                "SELECT r.row_id,r.chunk_id,v.embedding FROM search_rows r "
                "JOIN search_vectors v ON v.rowid=r.row_id "
                "WHERE r.index_version_id=? ORDER BY r.row_id"
            ),
            "inputs": "SELECT * FROM search_job_inputs WHERE job_id=?",
            "policy": "SELECT policy_json FROM job_status WHERE job_id=?",
        }
        return {
            name: [
                tuple(row)
                for row in connection.execute(
                    sql, (case.job if name in {"inputs", "policy"} else case.index,)
                )
            ]
            for name, sql in queries.items()
        }


def generation_charge(store, case):
    with store.database.connect() as connection:
        return connection.execute(
            "SELECT spent_seconds FROM search_generations WHERE index_version_id=?", (case.index,)
        ).fetchone()[0]


def run_pending(store, case):
    async def exercise():
        import time

        coordinator = SearchCoordinator(store, store.database.config.data_dir / "models")
        runner = JobRunner(
            store,
            {"pdf-search": preparation.build_search_steps(store, coordinator.models_directory)},
            reconcile=coordinator.reconcile,
            claim_blocking_reason=coordinator.claim_blocking_reason,
        )
        await runner.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = store.load_job_status(case.job)
                assert status is not None
                if status.status in {"completed", "failed", "cancelled"} and (
                    runner.snapshot().claimed_job_id is None
                ):
                    return status
                await asyncio.sleep(0.01)
            raise AssertionError("publication/recovery did not settle through the actual runner")
        finally:
            await runner.stop()

    return asyncio.run(exercise())


def assert_saved_search(store, case, session):
    query = QueryService(store, store.database.config.data_dir / "models", session_factory=session)
    graph = store.load_index(case.index)
    expected_mappings = [
        (mapping.passage_id, mapping.ordinal, mapping.start_offset, mapping.end_offset)
        for mapping in graph.mappings
    ]
    managed = store.load_import_managed_file(case.import_id)
    assert (store.database.config.data_dir / managed.relative_path).read_bytes() == case.pdf_bytes
    filename = store.load_pdf_import(case.import_id).display_filename
    for mode in ("words", "meaning"):
        result = query.search(case.notebook, "rainfall", mode)
        assert result["ready_sources"] == 1 and len(result["results"]) == 1
        hit = result["results"][0]
        assert (
            hit["source_id"],
            hit["source_version_id"],
            hit["index_version_id"],
            hit["import_id"],
            hit["profile_id"],
        ) == (case.source, case.version, case.index, case.import_id, case.profile_id)
        assert hit["chunk_id"] == graph.chunks[0].id
        assert hit["text"] == graph.chunks[0].text and hit["pdf_role"] == "current"
        assert hit["filename"] == filename
        assert [
            (c["passage_id"], c["ordinal"], c["start_offset"], c["end_offset"])
            for c in hit["citations"]
        ] == expected_mappings
        assert all(c["page_number"] == 1 for c in hit["citations"])
    assert case.old.original.read_bytes() == case.old.content


@pytest.mark.parametrize("kind", ["replacement", "model-switch"])
@pytest.mark.parametrize("fault", ["timeout", "persistence"])
def test_post_commit_settlement_keeps_accepted_pdf_and_model_publication(
    store, original, monkeypatch, publication_runtime, kind, fault
):
    clock = freeze_clock(monkeypatch)
    case = prepare_publication(store, original, kind)
    before = protected_output(store, case)
    documents = list(publication_runtime.document_calls)
    charge = generation_charge(store, case)
    settle = preparation._settle_budget
    faults = []

    def failed_settlement(store_arg, context, table, column, identity, started, reserved):
        committed = any(
            step.step_key == "publish-search" for step in store_arg.load_job_steps(case.job)
        )
        if context.job_id == case.job and context.step_key == "publish-search" and committed:
            faults.append((store_arg.load_job_status(case.job).status, reserved))
            if fault == "persistence":
                raise DatabaseBusyError("controlled post-COMMIT settlement write failure")
            clock.advance(reserved + 1)
        return settle(store_arg, context, table, column, identity, started, reserved)

    monkeypatch.setattr(preparation, "_settle_budget", failed_settlement)
    status = run_pending(store, case)
    assert status.status == "completed", (
        "accepted publication was changed to failed after displacing old readiness",
        status,
    )
    assert len(faults) == 1 and faults[0][0] == "completed"
    assert generation_charge(store, case) >= charge + faults[0][1]
    assert protected_output(store, case) == before
    assert publication_runtime.document_calls == documents
    attempts = store.load_step_attempts(case.job, "publish-search")
    assert len(attempts) == 1 and attempts[0].finished_at is not None
    assert attempts[0].outcome == "ok" and attempts[0].failure_reason is None
    assert store.load_pdf_import(case.import_id).state == "ready"
    state = store.load_source_pdf_state(case.source)
    assert state is not None and state.current_import_id == case.import_id
    assert state.pending_import_id is None
    assert_saved_search(store, case, publication_runtime)
    after_charge = generation_charge(store, case)
    restarted = reopen(store)
    assert run_pending(restarted, case).status == "completed"
    assert protected_output(restarted, case) == before
    assert generation_charge(restarted, case) == after_charge
    assert restarted.load_step_attempts(case.job, "publish-search") == attempts
    assert publication_runtime.document_calls == documents
    assert_saved_search(restarted, case, publication_runtime)


@pytest.mark.parametrize("kind", ["replacement", "model-switch"])
@pytest.mark.parametrize("seam", ["finish-attempt", "finish-job"])
def test_post_commit_runner_writes_cannot_fail_an_accepted_publication(
    store, original, monkeypatch, publication_runtime, kind, seam
):
    case = prepare_publication(store, original, kind)
    before = protected_output(store, case)
    documents = list(publication_runtime.document_calls)
    calls = []
    if seam == "finish-attempt":
        finish_attempt = store.finish_step_attempt

        def failed_attempt(job_id, step_key, attempt_number, *args, **kwargs):
            if job_id == case.job and step_key == "publish-search":
                calls.append("finish-attempt")
                raise DatabaseBusyError("controlled post-COMMIT attempt bookkeeping failure")
            return finish_attempt(job_id, step_key, attempt_number, *args, **kwargs)

        monkeypatch.setattr(store, "finish_step_attempt", failed_attempt)
    else:
        finish_job = store.finish_job

        def failed_completion(job_id, status, *args, **kwargs):
            if job_id == case.job:
                calls.append(status)
                if status == "completed":
                    raise DatabaseBusyError("controlled post-COMMIT job bookkeeping failure")
            return finish_job(job_id, status, *args, **kwargs)

        monkeypatch.setattr(store, "finish_job", failed_completion)
    status = run_pending(store, case)
    assert status.status == "completed", (
        "runner bookkeeping changed accepted readiness into failed work",
        seam,
        status,
    )
    if seam == "finish-attempt":
        # This write is inside publication now. The old post-COMMIT write is
        # deliberately absent; 4139df1 calls this hook and then fails the job.
        assert calls == []
    else:
        # The generic completion write fails and its error reaches the generic
        # failure recorder. Its guarded write must refuse to fail a completed job.
        assert calls == ["completed", "failed"]
    assert protected_output(store, case) == before
    assert publication_runtime.document_calls == documents
    attempts = store.load_step_attempts(case.job, "publish-search")
    assert len(attempts) == 1 and attempts[0].outcome == "ok"
    assert attempts[0].finished_at is not None and attempts[0].failure_reason is None
    assert store.load_pdf_import(case.import_id).state == "ready"
    assert_saved_search(store, case, publication_runtime)


@pytest.mark.parametrize("kind", ["replacement", "model-switch"])
def test_final_publication_transaction_acquisition_consumes_its_existing_allowance(
    store, original, monkeypatch, publication_runtime, kind
):
    clock = freeze_clock(monkeypatch)
    case = prepare_publication(store, original, kind)
    before = protected_output(store, case)
    charge = generation_charge(store, case)
    documents = list(publication_runtime.document_calls)
    transaction = store._transaction
    publishing = False
    waits = []

    @contextmanager
    def delayed_acquisition(write=True):
        with transaction(write=write) as connection:
            if publishing:
                # The writer has just acquired its transaction after this wait.
                # Checkpoint-body delays are covered separately by the existing
                # accounting tests; the timer must not start afresh after BEGIN.
                clock.advance(301)
                waits.append(301)
            yield connection

    monkeypatch.setattr(store, "_transaction", delayed_acquisition)
    if kind == "replacement":
        publish = store.publish_readiness

        def delayed_publish(*args, **kwargs):
            nonlocal publishing
            publishing = True
            try:
                return publish(*args, **kwargs)
            finally:
                publishing = False

        monkeypatch.setattr(store, "publish_readiness", delayed_publish)
    else:
        publish_switch = switching.publish_replacement

        def delayed_switch(*args, **kwargs):
            nonlocal publishing
            publishing = True
            try:
                return publish_switch(*args, **kwargs)
            finally:
                publishing = False

        monkeypatch.setattr(switching, "publish_replacement", delayed_switch)
    status = run_pending(store, case)
    assert status.status == "failed" and status.failure_code == "search_time_limit"
    assert waits == [301] and generation_charge(store, case) >= charge + 301
    assert protected_output(store, case) == before
    assert publication_runtime.document_calls == documents
    readiness = store.load_readiness(case.source)
    assert readiness is not None and readiness.index_version_ids == (case.old.index,)
    state = store.load_source_pdf_state(case.source)
    assert state is not None and state.current_import_id == case.old.item.id
    assert store.load_pdf_import(case.old.item.id).state == "ready"
    assert not any(step.step_key == "publish-search" for step in store.load_job_steps(case.job))
    assert SearchStorage(store).search(
        case.notebook, [case.old.item.source_version_id], query="rainfall"
    )
    assert case.old.original.read_bytes() == case.old.content


@pytest.mark.parametrize("kind", ["replacement", "model-switch"])
def test_cancellation_before_final_publication_still_rejects_it(
    store, original, monkeypatch, publication_runtime, kind
):
    case = prepare_publication(store, original, kind)
    if kind == "replacement":
        publish = store.publish_readiness

        def cancelled(*args, **kwargs):
            store.cancel_job(case.job)
            return publish(*args, **kwargs)

        monkeypatch.setattr(store, "publish_readiness", cancelled)
    else:
        publish_switch = switching.publish_replacement

        def cancelled_switch(*args, **kwargs):
            NotebookModelService(store).cancel(case.notebook, 1)
            return publish_switch(*args, **kwargs)

        monkeypatch.setattr(switching, "publish_replacement", cancelled_switch)
    assert run_pending(store, case).status == "cancelled"
    readiness = store.load_readiness(case.source)
    assert readiness is not None and readiness.index_version_ids == (case.old.index,)
    assert not any(step.step_key == "publish-search" for step in store.load_job_steps(case.job))
    assert SearchStorage(store).search(
        case.notebook, [case.old.item.source_version_id], query="rainfall"
    )
    assert case.old.original.read_bytes() == case.old.content


def test_final_publication_is_complete_before_a_post_commit_cancellation(
    store, original, publication_runtime, monkeypatch
):
    case = prepare_publication(store, original, "replacement")
    settle = preparation._settle_budget
    decisions = []

    def cancel_after_commit(store_arg, context, *args):
        if context.step_key == "publish-search" and any(
            step.step_key == "publish-search" for step in store_arg.load_job_steps(case.job)
        ):
            try:
                store_arg.cancel_job(case.job)
            except ConflictingWriteError:
                decisions.append("already completed")
            else:
                decisions.append("cancelled")
        return settle(store_arg, context, *args)

    monkeypatch.setattr(preparation, "_settle_budget", cancel_after_commit)
    assert run_pending(store, case).status == "completed"
    assert decisions == ["already completed"]
    with pytest.raises(ConflictingWriteError):
        store.cancel_job(case.job)
    assert_saved_search(store, case, publication_runtime)
