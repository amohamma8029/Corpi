"""Versioned leases use real SQLite; faults control time, ownership or process exit."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_preparation_accounting as accounting
from test_failed_prefix_accounting import charge, stage

from core.jobs.runner import JobStepCancelled, PermanentStepFailure, StepContext
from core.persistence.database import Database
from core.persistence.errors import ConflictingWriteError, PersistenceError
from core.persistence.models import AttemptGuard
from core.persistence.store import PersistenceStore
from core.search import preparation
from core.search import storage as storage_module
from core.search.provenance import runtime_versions
from core.search.runtime import vector_bytes
from core.search.storage import SearchStorage, generation_identity
from core.search.supervisor import stop_owned_process_tree, worker_environment

planned = accounting.planned
VECTOR = vector_bytes([1.0] + [0.0] * 383, 384)


def embed_stage(store, planned, monkeypatch):
    clock, job, context, step, _ = stage(store, planned, 1)
    monkeypatch.setattr(storage_module, "time", SimpleNamespace(monotonic=clock.monotonic))
    sessions = []

    class Session:
        load_seconds = 1
        peak_bytes = 1024
        execution_runtime = runtime_versions()

        def __init__(self, *args, **kwargs):
            sessions.append(self)
            self.options = kwargs
            self.closed = False
            clock.advance(1)

        def embed(self, texts, **kwargs):
            # Borrowing here fails if validation/admission left a borrow or lock open.
            with store._transaction() as connection:
                assert connection.in_transaction
            clock.advance(2)
            return [VECTOR for _ in texts]

        def close(self):
            self.closed = True
            clock.advance(1)

    monkeypatch.setattr(preparation, "EmbeddingSession", Session)
    return clock, job, context, step, sessions, Session


def batch_state(store, job):
    with store.database.connect() as connection:
        return dict(
            connection.execute(
                "SELECT b.* FROM search_batches b JOIN job_steps s"
                " ON s.output_index_version_id=b.index_version_id WHERE s.job_id=?"
                " AND s.step_key=?",
                (job, preparation.PLAN),
            ).fetchone()
        )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("legacy_worker", [False, True])
def test_policy_selects_one_batch_lease_or_unchanged_legacy_loop(
    store, monkeypatch, planned, legacy, legacy_worker
):
    clock, job, context, _ = planned
    if legacy or legacy_worker:
        # Write a different job from the outset, not a mutation of immutable saved JSON.
        store.finish_job(job, "failed", "unused versioned control")
        item = store.load_job_by_id(job)
        settings = preparation.effective_settings

        def legacy_settings(*args, **kwargs):
            if legacy:
                kwargs.pop("budget_policy", None)
            if legacy_worker:
                kwargs.pop("openblas_threads", None)
            return settings(*args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(preparation, "effective_settings", legacy_settings)
            job = preparation.create_search_job(
                store,
                item.source_id,
                item.extraction_version_id,
                "minilm-en",
                limits=preparation.SearchLimits(total_seconds=20),
            )
        store.claim_job(job, "accounting-tests")
        attempt = store.start_step_attempt(job, preparation.PLAN)
        context = replace(
            context,
            job_id=job,
            attempt_number=attempt.attempt_number,
            guard=AttemptGuard("accounting-tests", preparation.PLAN, attempt.attempt_number),
        )
        planned = clock, job, context, preparation.build_search_steps(store, Path("unused"))
    clock, job, context, step, sessions, _ = embed_stage(store, planned, monkeypatch)
    saved = preparation._inputs(store, job)[1]
    identity = generation_identity(
        store.load_job_by_id(job), preparation.profiles()["minilm-en"], saved
    )
    before = charge(store, job)
    reserve, settle = preparation._reserve_budget, preparation._settle_budget
    reservations, settlements = [], []

    def reserved(*args):
        result = reserve(*args)
        reservations.append(result)
        return result

    def settled(*args):
        result = settle(*args)
        settlements.append(result)
        return result

    monkeypatch.setattr(preparation, "_reserve_budget", reserved)
    monkeypatch.setattr(preparation, "_settle_budget", settled)
    step.run(context)
    assert len(reservations) == len(settlements) == (5 if legacy else 2)
    assert charge(store, job) - before == pytest.approx(4)
    assert batch_state(store, job)["complete"] == 1
    assert batch_state(store, job)["attempts"] == 1
    assert all(s.closed for s in sessions)
    assert preparation._inputs(store, job)[1] == saved
    assert ("budget_policy" not in saved) == legacy
    assert ("openblas_threads" not in saved) == legacy_worker
    assert all(
        s.options.get("openblas_threads") == (None if legacy_worker else 1) for s in sessions
    )
    assert (
        generation_identity(store.load_job_by_id(job), preparation.profiles()["minilm-en"], saved)
        == identity
    )


@pytest.mark.parametrize("fault", ["input", "cancel", "stale", "settlement", "admission"])
def test_batch_failures_preserve_admission_and_charges(store, monkeypatch, planned, fault):
    clock, job, context, step, sessions, _ = embed_stage(store, planned, monkeypatch)
    before = charge(store, job)
    batch_inputs = SearchStorage.batch_inputs
    admitted = []
    # Cancellation is a concurrent writer, not a nested borrow of the input reader.
    external = PersistenceStore(Database(store.database.config))
    external.database.initialize()

    def failed_input(*args):
        admitted.append(True)
        assert (
            args[0]
            .execute(
                "SELECT spent_seconds FROM search_generations WHERE index_version_id=?", (args[1],)
            )
            .fetchone()[0]
            > before
        )
        if fault == "cancel":
            external.cancel_job(job)
            raise JobStepCancelled("controlled cancellation")
        if fault == "stale":
            external.release_running_jobs()
            external.claim_job(job, "replacement-owner")
            raise ConflictingWriteError("controlled stale owner")
        if fault in {"input", "settlement"}:
            clock.advance(3)
            raise TimeoutError("controlled batch input failure")
        return batch_inputs(*args)

    monkeypatch.setattr(SearchStorage, "batch_inputs", staticmethod(failed_input))
    reserve = preparation._reserve_budget
    settle = preparation._settle_budget
    reservations = 0

    def reserved(*args):
        nonlocal reservations
        reservations += 1
        if reservations == 2 and fault == "admission":
            raise PersistenceError("controlled batch admission failure")
        return reserve(*args)

    def settled(*args):
        if reservations == 2 and fault == "settlement":
            raise PersistenceError("controlled batch settlement failure")
        return settle(*args)

    monkeypatch.setattr(preparation, "_reserve_budget", reserved)
    monkeypatch.setattr(preparation, "_settle_budget", settled)
    expected = {
        "input": TimeoutError,
        "cancel": JobStepCancelled,
        "stale": ConflictingWriteError,
        "settlement": PermanentStepFailure,
        "admission": PersistenceError,
    }[fault]
    with pytest.raises(expected):
        step.run(context)
    assert admitted == ([] if fault == "admission" else [True])
    assert not sessions
    assert batch_state(store, job)["attempts"] == 0
    assert batch_state(store, job)["complete"] == 0
    delta = charge(store, job) - before
    assert delta == pytest.approx(3 if fault == "input" else 0 if fault == "admission" else 20)


def test_expired_batch_commit_rolls_back_native_vectors(store, monkeypatch, planned):
    clock, job, context, _, _, _ = embed_stage(store, planned, monkeypatch)
    assert context.guard is not None
    batch = batch_state(store, job)
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_batches SET attempts=1 WHERE index_version_id=?",
            (batch["index_version_id"],),
        )
    SearchStorage(store).record_batch_execution(
        job,
        context.guard,
        batch["index_version_id"],
        0,
        1,
        runtime_versions(),
        deadline=clock.monotonic() + 20,
    )
    times = iter([clock.monotonic(), clock.monotonic() + 30])
    monkeypatch.setattr(storage_module, "time", SimpleNamespace(monotonic=lambda: next(times)))
    with pytest.raises(TimeoutError, match="deadline expired"):
        SearchStorage(store).commit_batch(
            job,
            context.guard,
            batch["index_version_id"],
            0,
            json.loads(batch["chunk_ids_json"]),
            [VECTOR],
            deadline=clock.monotonic() + 20,
            execution_attempt=1,
        )
    assert batch_state(store, job)["complete"] == 0
    with store.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM search_vectors WHERE rowid IN"
                " (SELECT row_id FROM search_rows WHERE index_version_id=?)",
                (batch["index_version_id"],),
            ).fetchone()[0]
            == 0
        )


def test_slow_validation_cannot_start_inference_after_lease(store, monkeypatch, planned):
    clock, job, context, step, sessions, _ = embed_stage(store, planned, monkeypatch)
    inputs = SearchStorage.batch_inputs

    def slow(*args):
        clock.advance(21)
        return inputs(*args)

    monkeypatch.setattr(SearchStorage, "batch_inputs", staticmethod(slow))
    with pytest.raises(PermanentStepFailure) as failure:
        step.run(context)
    assert failure.value.code == "search_time_limit"
    assert not sessions and batch_state(store, job)["attempts"] == 0
    assert charge(store, job) >= 21


def test_exhausted_batch_never_calls_model(store, monkeypatch, planned):
    _, job, context, step, sessions, _ = embed_stage(store, planned, monkeypatch)
    index = batch_state(store, job)["index_version_id"]
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_batches SET attempts=2 WHERE index_version_id=?", (index,)
        )
    with pytest.raises(PermanentStepFailure) as failure:
        step.run(context)
    assert failure.value.code == "embedding_attempt_limit" and not sessions


def test_two_timeouts_close_both_sessions_and_keep_attempt_cap(store, monkeypatch, planned):
    clock, job, context, step, sessions, Session = embed_stage(store, planned, monkeypatch)
    monkeypatch.setattr(
        preparation, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.advance)
    )
    monkeypatch.setattr(preparation.random, "uniform", lambda *_: 2)

    def timeout(*args, **kwargs):
        clock.advance(2)
        raise TimeoutError("controlled early model timeout")

    monkeypatch.setattr(Session, "embed", timeout)
    began = clock.monotonic()
    with pytest.raises(PermanentStepFailure) as failure:
        step.run(context)
    assert failure.value.code == "embedding_timeout"
    assert len(sessions) == 2 and all(s.closed for s in sessions)
    assert batch_state(store, job)["attempts"] == 2
    assert batch_state(store, job)["complete"] == 0
    # Floating-point backoff may use one final 50ms tick. Charge every actual tick.
    assert charge(store, job) == pytest.approx(clock.monotonic() - began)
    assert 10 <= charge(store, job) <= 10.051


@pytest.mark.parametrize("change", ["cancel", "stale"])
def test_invalidated_inference_cannot_commit_or_refund(store, monkeypatch, planned, change):
    _, job, context, step, sessions, Session = embed_stage(store, planned, monkeypatch)
    embed = Session.embed

    def invalidated(*args, **kwargs):
        vectors = embed(*args, **kwargs)
        if change == "cancel":
            store.cancel_job(job)
        else:
            store.release_running_jobs()
            store.claim_job(job, "new-owner")
        return vectors

    monkeypatch.setattr(Session, "embed", invalidated)
    with pytest.raises(ConflictingWriteError):
        step.run(context)
    assert batch_state(store, job)["complete"] == 0
    assert batch_state(store, job)["attempts"] == 1
    assert all(s.closed for s in sessions)
    assert charge(store, job) == pytest.approx(20)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("legacy_worker", [False, True])
def test_explicit_resume_copies_the_saved_policy_without_reversion_or_upgrade(
    store, monkeypatch, planned, legacy, legacy_worker
):
    from test_controls import queued

    from core.search.controls import PreparationControl
    from core.search.handoff import SearchCoordinator

    settings = preparation.effective_settings

    def saved_settings(*args, **kwargs):
        if legacy:
            kwargs.pop("budget_policy", None)
        if legacy_worker:
            kwargs.pop("openblas_threads", None)
        return settings(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(preparation, "effective_settings", saved_settings)
        _, source, job = queued(store)
    with store._transaction() as connection:
        connection.execute("UPDATE search_job_inputs SET spent_seconds=7 WHERE job_id=?", (job,))
        connection.execute(
            "UPDATE job_status SET policy_json=? WHERE job_id=?",
            (
                '{"search_prepare":{"timeout_seconds":600,"attempt_cap":2,"backoff_base_seconds":2}}',
                job,
            ),
        )
        original = dict(
            connection.execute("SELECT * FROM search_job_inputs WHERE job_id=?", (job,)).fetchone()
        )
        policy = connection.execute(
            "SELECT policy_json FROM job_status WHERE job_id=?", (job,)
        ).fetchone()[0]
    control = PreparationControl(store)
    control.cancel(source, job)
    SearchCoordinator(store, Path("unused")).reconcile()
    assert store.load_job_status(job).status == "cancelled"
    resumed = control.resume(source, job, "saved-policy")
    with store.database.connect() as connection:
        after = dict(
            connection.execute(
                "SELECT * FROM search_job_inputs WHERE job_id=?", (resumed,)
            ).fetchone()
        )
        assert {k: v for k, v in original.items() if k != "job_id"} == {
            k: v for k, v in after.items() if k != "job_id"
        }
        assert (
            connection.execute(
                "SELECT policy_json FROM job_status WHERE job_id=?", (resumed,)
            ).fetchone()[0]
            == policy
        )
    assert ("budget_policy" not in json.loads(after["settings_json"])) == legacy
    assert ("openblas_threads" not in json.loads(after["settings_json"])) == legacy_worker


def test_unknown_policy_is_rejected_instead_of_using_legacy_rules(store, monkeypatch, planned):
    _, job, _, _ = planned
    item = store.load_job_by_id(job)
    settings = preparation.effective_settings

    def future_settings(*args, **kwargs):
        return {**settings(*args, **kwargs), "budget_policy": "unknown-version"}

    with monkeypatch.context() as scoped:
        scoped.setattr(preparation, "effective_settings", future_settings)
        future = preparation.create_search_job(
            store, item.source_id, item.extraction_version_id, "minilm-en"
        )
    with pytest.raises(ValueError, match="unsupported_search_budget_policy"):
        preparation._inputs(store, future)


def crash_worker(root, job, boundary):
    from core.persistence.database import DatabaseConfig

    database = Database(DatabaseConfig(Path(root)))
    database.initialize()
    store = PersistenceStore(database)
    storage = SearchStorage.commit_batch
    if boundary == "input":

        def crash_inputs(connection, index_id, chunk_ids):
            os._exit(73)

        SearchStorage.batch_inputs = staticmethod(crash_inputs)

    class Session:
        load_seconds = 0
        peak_bytes = 0
        execution_runtime = runtime_versions()

        def __init__(self, *args, **kwargs):
            if boundary == "admission":
                os._exit(73)

        def embed(self, texts, **kwargs):
            if boundary == "inference":
                os._exit(73)
            return [VECTOR for _ in texts]

        def close(self):
            pass

    def committed(self, *args, **kwargs):
        storage(self, *args, **kwargs)
        os._exit(73)

    if boundary == "commit":
        SearchStorage.commit_batch = committed
    preparation.EmbeddingSession = Session
    context = StepContext(
        job,
        "embed-search",
        1,
        lambda: False,
        lambda: False,
        AttemptGuard("accounting-tests", "embed-search", 1),
    )
    preparation.build_search_steps(store, Path("unused"))[1].run(context)


@pytest.mark.parametrize("boundary", ["input", "admission", "inference", "commit"])
@pytest.mark.parametrize(
    "planned", [preparation.SearchLimits(total_seconds=600, batch_seconds=20)], indirect=True
)
def test_actual_exit_retains_batch_reservation_and_restart_reuses_commits(
    store, monkeypatch, planned, boundary
):
    _, job, context, _, _, _ = embed_stage(store, planned, monkeypatch)
    before = charge(store, job)
    child = subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--crash-batch",
            str(store.database.config.data_dir),
            job,
            boundary,
        ],
        env=worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = child.communicate(timeout=25)
        assert child.returncode == 73, (stdout, stderr)
    finally:
        stop_owned_process_tree(child)
    database = Database(store.database.config)
    database.initialize()
    restarted = PersistenceStore(database)
    assert 20 <= charge(restarted, job) - before < 21
    batch = batch_state(restarted, job)
    assert batch["attempts"] == (0 if boundary == "input" else 1)
    assert batch["complete"] == (boundary == "commit")
    with restarted.database.connect() as connection:
        before_receipts = connection.execute(
            "SELECT * FROM search_batch_executions WHERE job_id=?", (job,)
        ).fetchall()
    assert len(before_receipts) == (1 if boundary in {"inference", "commit"} else 0)
    assert all(json.loads(row["runtime_json"]) == runtime_versions() for row in before_receipts)
    restarted.release_running_jobs()
    restarted.claim_job(job, "batch-restart")
    attempt = restarted.start_step_attempt(job, "embed-search")
    current = replace(
        context,
        attempt_number=attempt.attempt_number,
        guard=AttemptGuard("batch-restart", "embed-search", attempt.attempt_number),
    )
    preparation.build_search_steps(restarted, Path("unused"))[1].run(current)
    assert batch_state(restarted, job)["complete"] == 1
    assert batch_state(restarted, job)["attempts"] == (1 if boundary in {"input", "commit"} else 2)
    assert charge(restarted, job) >= before + 20
    with restarted.database.connect() as connection:
        after_receipts = connection.execute(
            "SELECT * FROM search_batch_executions WHERE job_id=? ORDER BY batch_attempt_number",
            (job,),
        ).fetchall()
    assert len(after_receipts) == (2 if boundary == "inference" else 1)
    assert [tuple(row) for row in after_receipts[: len(before_receipts)]] == [
        tuple(row) for row in before_receipts
    ]


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--crash-batch":
    crash_worker(*sys.argv[2:])
