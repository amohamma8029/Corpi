"""Failed metadata prefixes must consume a durable lease before retry or restart."""

import asyncio
import os
import subprocess
import sys
import time as real_time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_preparation_accounting as accounting

from core.jobs import runner as runner_module
from core.jobs.runner import (
    JobStepCancelled,
    PermanentStepFailure,
    StepCommitContext,
    StepContext,
    StepOutcome,
)
from core.persistence.database import Database, DatabaseConfig
from core.persistence.errors import PersistenceError
from core.persistence.models import AttemptGuard
from core.persistence.store import PersistenceStore
from core.search import preparation
from core.search.profiles import profiles
from core.search.storage import SearchStorage
from core.search.supervisor import stop_owned_process_tree, worker_environment

planned = accounting.planned


def charge(store, job):
    with store.database.connect() as connection:
        inputs = connection.execute(
            "SELECT spent_seconds FROM search_job_inputs WHERE job_id=?",
            (job,),
        ).fetchone()[0]
        row = connection.execute(
            "SELECT g.spent_seconds FROM search_generations g JOIN job_steps s"
            " ON s.output_index_version_id=g.index_version_id WHERE s.job_id=? AND s.step_key=?",
            (job, preparation.PLAN),
        ).fetchone()
        # Plan run can create a generation before its checkpoint is attached.
        if row is None:
            profile, settings, _ = preparation._inputs(store, job)
            identity = SearchStorage(store).lookup_generation(job, profile, settings)
            row = connection.execute(
                "SELECT spent_seconds FROM search_generations WHERE index_version_id=?",
                (identity,),
            ).fetchone()
        return inputs + (row[0] if row else 0)


def stage(store, planned, position, commit_phase=False):
    clock, job, context, steps = planned
    if position or commit_phase:
        outcome = steps[0].run(context)
        if position:
            steps[0].commit(
                store, StepCommitContext(job, preparation.PLAN, context.guard, "a" * 64, outcome)
            )
            store.finish_step_attempt(job, preparation.PLAN, context.attempt_number, None)
            attempt = store.start_step_attempt(job, steps[position].step_key)
            guard = AttemptGuard(
                "accounting-tests", steps[position].step_key, attempt.attempt_number
            )
            context = StepContext(
                job,
                steps[position].step_key,
                attempt.attempt_number,
                lambda: False,
                lambda: False,
                guard,
            )
    else:
        outcome = StepOutcome(1, {})
    return clock, job, context, steps[position], outcome


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize("fault", ["currency", "inputs"])
def test_failed_run_prefix_retains_elapsed_charge(store, monkeypatch, planned, position, fault):
    clock, job, context, step, _ = stage(store, planned, position)
    before = charge(store, job)

    def stalled(*_):
        clock.advance(15)
        raise TimeoutError("controlled transient metadata read timeout")

    with monkeypatch.context() as scoped:
        scoped.setattr(preparation, "_" + fault, stalled)
        with pytest.raises(TimeoutError):
            step.run(context)
    assert charge(store, job) - before >= 15
    assert not any(r.step_key == step.step_key for r in store.load_job_steps(job))


@pytest.mark.parametrize("position", range(4))
def test_failed_commit_prefix_retains_elapsed_charge(store, monkeypatch, planned, position):
    clock, job, context, step, outcome = stage(store, planned, position, True)
    assert context.guard is not None
    before = charge(store, job)

    def stalled(*_):
        clock.advance(15)
        raise TimeoutError("controlled commit metadata timeout")

    with monkeypatch.context() as scoped:
        scoped.setattr(preparation, "_inputs", stalled)
        with pytest.raises(TimeoutError):
            step.commit(
                store, StepCommitContext(job, step.step_key, context.guard, "a" * 64, outcome)
            )
    assert charge(store, job) - before >= 15


def test_failed_plan_generation_lookup_is_inside_the_durable_prefix(store, monkeypatch, planned):
    clock, job, context, steps = planned

    def stalled(*_):
        clock.advance(15)
        raise TimeoutError("controlled generation lookup timeout")

    with monkeypatch.context() as scoped:
        scoped.setattr(SearchStorage, "lookup_generation", stalled)
        with pytest.raises(TimeoutError):
            steps[0].run(context)
    assert charge(store, job) >= 15


def test_cancelled_metadata_cannot_refund_its_committed_prefix_lease(store, monkeypatch, planned):
    clock, job, context, steps = planned

    def cancelled(*_):
        clock.advance(15)
        store.cancel_job(job)
        raise JobStepCancelled("controlled current cancellation")

    monkeypatch.setattr(preparation, "_currency", cancelled)
    with pytest.raises(JobStepCancelled):
        steps[0].run(context)
    assert charge(store, job) == 20


def test_unavailable_settlement_keeps_full_committed_reservation(store, monkeypatch, planned):
    clock, job, context, steps = planned
    transaction = store._transaction
    calls = 0

    @contextmanager
    def unavailable_settlement(write=True):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PersistenceError("controlled unavailable settlement")
        with transaction(write=write) as connection:
            yield connection

    def stalled(*_):
        clock.advance(15)
        raise TimeoutError("controlled metadata timeout")

    with monkeypatch.context() as scoped:
        scoped.setattr(store, "_transaction", unavailable_settlement)
        scoped.setattr(preparation, "_currency", stalled)
        with pytest.raises(PersistenceError, match="unavailable settlement"):
            steps[0].run(context)
    assert charge(store, job) == 20


def test_failed_budget_admission_permits_no_metadata_or_processing(store, monkeypatch, planned):
    _, job, context, steps = planned
    entered = []

    @contextmanager
    def unavailable_admission(write=True):
        raise PersistenceError("controlled unavailable budget admission")
        yield  # pragma: no cover - contextmanager protocol;no connection is admitted.

    with monkeypatch.context() as scoped:
        scoped.setattr(store, "_transaction", unavailable_admission)
        scoped.setattr(preparation, "_currency", lambda *_: entered.append(True))
        with pytest.raises(PersistenceError, match="unavailable budget admission"):
            steps[0].run(context)
    assert entered == []
    assert store.load_job_steps(job) == ()
    # No reservation was committed and no preparation body ran. This is an
    # admission-barrier proof,NOT proof of durable metering while SQLite is down.


@pytest.mark.parametrize("restart", [False, True])
def test_two_failed_prefixes_cannot_reset_the_source_allowance(
    store, monkeypatch, planned, restart
):
    clock, previous, context, _ = planned
    old = store.load_job_by_id(previous)
    store.finish_job(previous, "failed", "unused control")
    with store._transaction() as connection:
        job = preparation.enqueue_search_job(
            connection,
            old.source_id,
            old.source_version_id,
            old.extraction_version_id,
            profiles()["minilm-en"],
            0,
            limits=preparation.SearchLimits(total_seconds=20),
            work_suffix="prefix-retry",
        )
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(monotonic=clock.monotonic))

    def stalled(*_):
        clock.advance(15)
        raise TimeoutError("controlled transient metadata timeout")

    monkeypatch.setattr(preparation, "_currency", stalled)
    if restart:
        store.claim_job(job, "prefix-first")
        attempt = store.start_step_attempt(job, preparation.PLAN)
        guard = AttemptGuard("prefix-first", preparation.PLAN, attempt.attempt_number)
        first = StepContext(
            job, preparation.PLAN, attempt.attempt_number, lambda: False, lambda: False, guard
        )
        with pytest.raises(TimeoutError):
            preparation.build_search_steps(store, Path("unused"))[0].run(first)
        assert charge(store, job) >= 15
        store.finish_step_attempt(job, preparation.PLAN, attempt.attempt_number, "metadata timeout")
        database = Database(store.database.config)
        database.initialize()
        store = PersistenceStore(database)

    async def execute():
        runner = runner_module.JobRunner(
            store,
            {"pdf-search": preparation.build_search_steps(store, Path("unused"))},
            backoff_sleep=lambda _: asyncio.sleep(0),
        )
        await runner.start()
        try:
            deadline = real_time.monotonic() + 10
            while real_time.monotonic() < deadline:
                status = store.load_job_status(job)
                if status and status.status == "failed":
                    assert status.failure_code == "search_time_limit", status
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("failed prefixes did not exhaust source budget")
        finally:
            await runner.stop()

    asyncio.run(execute())
    assert len(store.load_step_attempts(job, preparation.PLAN)) == 2
    assert charge(store, job) >= 30
    assert store.load_job_steps(job) == ()


def crash_worker(root, job):
    database = Database(DatabaseConfig(Path(root)))
    database.initialize()
    store = PersistenceStore(database)
    preparation._currency = lambda *_: os._exit(73)
    guard = AttemptGuard("accounting-tests", preparation.PLAN, 1)
    context = StepContext(job, preparation.PLAN, 1, lambda: False, lambda: False, guard)
    preparation.build_search_steps(store, Path("unused"))[0].run(context)


def test_actual_exit_during_first_metadata_read_keeps_full_reservation(store, planned):
    _, job, _, _ = planned
    child = subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--prefix-worker",
            str(store.database.config.data_dir),
            job,
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
    assert charge(restarted, job) == 20
    restarted.release_running_jobs()
    restarted.claim_job(job, "prefix-restart")
    attempt = restarted.start_step_attempt(job, preparation.PLAN)
    guard = AttemptGuard("prefix-restart", preparation.PLAN, attempt.attempt_number)
    context = StepContext(
        job, preparation.PLAN, attempt.attempt_number, lambda: False, lambda: False, guard
    )
    with pytest.raises(PermanentStepFailure) as failure:
        preparation.build_search_steps(restarted, Path("unused"))[0].run(context)
    assert failure.value.code == "search_time_limit"


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--prefix-worker":
    crash_worker(*sys.argv[2:])
