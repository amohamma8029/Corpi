"""Synthetic-step coverage for the durable job runner (issue #9)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

import pytest

from core.jobs.runner import (
    JobRunner,
    JobStepCancelled,
    PaidCallOutcomeUnknown,
    PermanentStepFailure,
    StepCommitContext,
    StepContext,
    StepDefinition,
    StepOutcome,
    StepPolicy,
    commit_checkpoint,
)
from core.persistence.errors import ConflictingWriteError
from core.persistence.models import AttemptGuard, new_id
from core.persistence.store import PersistenceStore

FAST_POLICY = StepPolicy(timeout_seconds=5.0, attempt_cap=2, backoff_base_seconds=0.0)


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class RunnerHarness:
    """Runs the runner's event loop on a background thread for sync tests."""

    def __init__(self, store: PersistenceStore, step_sets, **kwargs) -> None:
        self.runner = JobRunner(store, step_sets, **kwargs)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._main, daemon=True)
        self._started = threading.Event()

    def _main(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.runner.start())
        self._started.set()
        self.loop.run_forever()

    def start(self) -> None:
        self.thread.start()
        assert self._started.wait(5)

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self.runner.stop(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)

    def enqueue_and_notify(self, store: PersistenceStore, job_id: str) -> None:
        store.create_job_status(job_id)
        self.runner.notify()


def enqueue(store: PersistenceStore, kind: str, key: str, name: str) -> tuple[str, str]:
    notebook_id = new_id()
    store.create_notebook(notebook_id, f"Notebook {name}")
    source_id = new_id()
    store.create_source(source_id, notebook_id, name)
    source_version = store.create_source_version(source_id, "a" * 64, None)
    job = store.create_job(key, kind, source_id, source_version_id=source_version.id)
    return job.id, source_id


def step_recording(
    name: str,
    records: dict[str, list[int]],
    behaviour: Callable[[StepContext], StepOutcome],
    *,
    policy_class: str = "generic",
) -> StepDefinition:
    def run(context: StepContext) -> StepOutcome:
        records.setdefault(name, []).append(context.attempt_number)
        return behaviour(context)

    return StepDefinition(name, run, policy_class=policy_class)


def test_runs_steps_in_order_and_records_completions(store: PersistenceStore) -> None:
    order: list[str] = []

    def make(name: str) -> StepDefinition:
        def run(context: StepContext) -> StepOutcome:
            order.append(name)
            return StepOutcome(1, {"step": name})

        return StepDefinition(name, run)

    runner = RunnerHarness(store, {"demo": [make("one"), make("two")]})
    job_id, _ = enqueue(store, "demo", "demo:ordered", "ordered.pdf")
    runner.start()
    try:
        runner.enqueue_and_notify(store, job_id)
        assert wait_until(
            lambda: store.load_job_status(job_id).status == "completed"  # type: ignore[union-attr]
        ), store.load_job_status(job_id)
    finally:
        runner.stop()

    assert order == ["one", "two"]
    steps = store.load_job_steps(job_id)
    assert [step.step_key for step in steps] == ["one", "two"]
    assert [step.checkpoint for step in steps] == [{"step": "one"}, {"step": "two"}]
    status = store.load_job_status(job_id)
    assert status is not None
    assert status.attempt_count == 2
    for step in steps:
        attempts = store.load_step_attempts(job_id, step.step_key)
        assert [(item.attempt_number, item.finished_at is not None) for item in attempts] == [
            (1, True)
        ]


def test_failed_attempt_retries_then_completes(store: PersistenceStore) -> None:
    records: dict[str, list[int]] = {}

    def fail_once(context: StepContext) -> StepOutcome:
        if context.attempt_number == 1:
            raise RuntimeError("transient")
        return StepOutcome(1, {"ok": True})

    runner = RunnerHarness(
        store,
        {"demo": [step_recording("flaky", records, fail_once)]},
        policies={"generic": FAST_POLICY},
        backoff_sleep=lambda delay: None,
    )
    job_id, _ = enqueue(store, "demo", "demo:retry", "retry.pdf")
    runner.start()
    try:
        runner.enqueue_and_notify(store, job_id)
        assert wait_until(
            lambda: store.load_job_status(job_id).status == "completed"  # type: ignore[union-attr]
        ), store.load_job_status(job_id)
    finally:
        runner.stop()

    assert records["flaky"] == [1, 2]
    attempts = store.load_step_attempts(job_id, "flaky")
    assert [(item.attempt_number, item.failure_reason) for item in attempts] == [
        (1, "RuntimeError: transient"),
        (2, None),
    ]
    assert [step.step_key for step in store.load_job_steps(job_id)] == ["flaky"]


def test_permanent_failure_stops_with_reason(store: PersistenceStore) -> None:
    records: dict[str, list[int]] = {}

    def always_fail(context: StepContext) -> StepOutcome:
        raise RuntimeError("boom")

    runner = RunnerHarness(
        store,
        {"demo": [step_recording("doomed", records, always_fail)]},
        policies={"generic": FAST_POLICY},
        backoff_sleep=lambda delay: None,
    )
    job_id, _ = enqueue(store, "demo", "demo:doomed", "doomed.pdf")
    runner.start()
    try:
        runner.enqueue_and_notify(store, job_id)
        assert wait_until(
            lambda: store.load_job_status(job_id).status == "failed"  # type: ignore[union-attr]
        ), store.load_job_status(job_id)
    finally:
        runner.stop()

    assert records["doomed"] == [1, 2]
    status = store.load_job_status(job_id)
    assert status is not None
    assert status.failure_reason == "step doomed failed after 2 attempt(s)"
    assert store.load_job_steps(job_id) == ()
    attempts = store.load_step_attempts(job_id, "doomed")
    assert all(item.finished_at is not None for item in attempts)


def test_resume_reuses_committed_steps_and_reruns_pending(store: PersistenceStore) -> None:
    records: dict[str, list[int]] = {}
    release = threading.Event()

    def second_step(context: StepContext) -> StepOutcome:
        if context.attempt_number == 1:
            release.wait(10)
        return StepOutcome(1, {"two": True})

    steps = lambda: [  # noqa: E731 - short local factory for two runner generations
        step_recording("one", records, lambda context: StepOutcome(1, {"one": True})),
        step_recording("two", records, second_step),
    ]
    first = RunnerHarness(
        store,
        {"demo": steps()},
        policies={"generic": StepPolicy(30.0, 2, 0.0)},
    )
    job_id, _ = enqueue(store, "demo", "demo:resume", "resume.pdf")
    first.start()
    try:
        first.enqueue_and_notify(store, job_id)
        assert wait_until(lambda: records.get("two", []) == [1]), records
        assert [step.step_key for step in store.load_job_steps(job_id)] == ["one"]
    finally:
        first.stop()
        release.set()

    # An immediate restart recovers the previous instance's work without any
    # manual reset: start() releases running jobs because the data-directory
    # lock proves the old core is gone (no heartbeat-age cutoff).
    second = RunnerHarness(
        store,
        {"demo": steps()},
        policies={"generic": FAST_POLICY},
    )
    second.start()
    try:
        second.runner.notify()
        assert wait_until(
            lambda: store.load_job_status(job_id).status == "completed"  # type: ignore[union-attr]
        ), store.load_job_status(job_id)
    finally:
        second.stop()

    assert records["one"] == [1]  # committed step reused, never re-executed
    assert records["two"] == [1, 2]  # pending step re-attempted from its checkpoint state
    assert [step.step_key for step in store.load_job_steps(job_id)] == ["one", "two"]
    two_attempts = store.load_step_attempts(job_id, "two")
    assert [item.attempt_number for item in two_attempts] == [1, 2]


def test_cancel_before_claim_is_not_run(store: PersistenceStore) -> None:
    records: dict[str, list[int]] = {}
    runner = RunnerHarness(
        store,
        {"demo": [step_recording("never", records, lambda context: StepOutcome(1, {}))]},
    )
    job_id, _ = enqueue(store, "demo", "demo:cancelled", "cancelled.pdf")
    store.create_job_status(job_id)
    store.cancel_job(job_id, "user cancelled")

    runner.start()
    try:
        runner.runner.notify()
        time.sleep(0.5)
    finally:
        runner.stop()

    assert records == {}
    status = store.load_job_status(job_id)
    assert status is not None
    assert status.status == "cancelled"


def test_cooperative_cancel_during_step_stops_run_and_rejects_late_commits(
    store: PersistenceStore,
) -> None:
    records: dict[str, list[int]] = {}
    observed_cancel = threading.Event()

    def cooperative(context: StepContext) -> StepOutcome:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if context.is_cancelled():
                observed_cancel.set()
                raise JobStepCancelled()
            time.sleep(0.01)
        raise RuntimeError("never cancelled")

    runner = RunnerHarness(
        store,
        {
            "demo": [
                step_recording("cooperative", records, cooperative),
                step_recording("after", records, lambda context: StepOutcome(1, {})),
            ]
        },
        policies={"generic": StepPolicy(15.0, 1, 0.0)},
    )
    job_id, _ = enqueue(store, "demo", "demo:cooperative", "coop.pdf")
    runner.start()
    try:
        runner.enqueue_and_notify(store, job_id)
        assert wait_until(lambda: records.get("cooperative", []) == [1]), records
        store.cancel_job(job_id, "user cancelled")
        assert wait_until(lambda: observed_cancel.is_set()), "step never observed cancellation"
        # Let the runner observe the stopped step before asserting.
        assert wait_until(
            lambda: store.load_step_attempts(job_id, "cooperative")[0].failure_reason == "cancelled"
        )
    finally:
        runner.stop()

    assert "after" not in records
    status = store.load_job_status(job_id)
    assert status is not None
    assert status.status == "cancelled"
    assert store.load_job_steps(job_id) == ()
    # A late completion for a cancelled job is rejected by the guarded commit.
    with pytest.raises(ConflictingWriteError):
        store.commit_step(job_id, "late", 1, {"late": True}, guard=AttemptGuard("core-x", 1))


def test_stuck_pool_is_reported_and_recovers(store: PersistenceStore) -> None:
    records: dict[str, list[int]] = {}
    release = threading.Event()

    def stuck(context: StepContext) -> StepOutcome:
        release.wait(10)
        return StepOutcome(1, {"stuck": True})

    def quick(context: StepContext) -> StepOutcome:
        return StepOutcome(1, {"quick": True})

    runner = RunnerHarness(
        store,
        {
            "stuck": [step_recording("stuck", records, stuck)],
            "quick": [step_recording("quick", records, quick)],
        },
        pool_size=1,
        policies={"generic": StepPolicy(0.15, 1, 0.0)},
        poll_interval_seconds=0.05,
    )
    stuck_job, _ = enqueue(store, "stuck", "demo:stuck", "stuck.pdf")
    quick_job, _ = enqueue(store, "quick", "demo:quick", "quick.pdf")
    runner.start()
    try:
        runner.enqueue_and_notify(store, stuck_job)
        assert wait_until(lambda: records.get("stuck", []) == [1]), records
        # The attempt times out while its thread keeps the only worker busy.
        assert wait_until(lambda: runner.runner.snapshot().state == "stalled")
        runner.enqueue_and_notify(store, quick_job)
        assert wait_until(
            lambda: bool(
                (status := store.load_job_status(quick_job)) and status.failure_reason  # type: ignore[union-attr]
            )
        )
        quick_status = store.load_job_status(quick_job)
        assert quick_status is not None
        assert quick_status.status == "queued"
        assert "worker" in (quick_status.failure_reason or "")

        # Releasing the stuck thread frees the pool; the queued job then runs.
        release.set()
        assert wait_until(
            lambda: store.load_job_status(quick_job).status == "completed"  # type: ignore[union-attr]
        ), store.load_job_status(quick_job)
        assert store.load_job_status(quick_job).failure_reason is None  # type: ignore[union-attr]
        stuck_status = store.load_job_status(stuck_job)
        assert stuck_status is not None
        assert stuck_status.status == "failed"
        assert "timed out" in (stuck_status.failure_reason or "")
    finally:
        release.set()
        runner.stop()


def test_release_running_jobs_recovers_every_running_job(store: PersistenceStore) -> None:
    """An immediate restart recovers fresh-heartbeat jobs too: the exclusive
    data-directory lock, not a heartbeat age, proves the previous core is gone."""
    first, _ = enqueue(store, "demo", "demo:fresh-restart", "fresh-restart.pdf")
    second, _ = enqueue(store, "demo", "demo:also-fresh", "also-fresh.pdf")
    store.create_job_status(first)
    store.create_job_status(second)
    store.claim_job(first, "core-old")
    store.claim_job(second, "core-old")

    released = store.release_running_jobs()
    assert set(released) == {first, second}
    for job_id in (first, second):
        status = store.load_job_status(job_id)
        assert status is not None
        assert status.status == "queued"
        assert status.owner_instance_id is None
    assert store.claim_job(first, "core-new").owner_instance_id == "core-new"


def test_attempt_cap_and_policy_survive_restart(store: PersistenceStore) -> None:
    records: dict[str, list[int]] = {}
    gate = threading.Event()

    def second_step(context: StepContext) -> StepOutcome:
        if context.attempt_number == 1:
            gate.wait(10)
        return StepOutcome(1, {"two": True})

    steps = lambda: [  # noqa: E731 - two runner generations
        step_recording("one", records, lambda context: StepOutcome(1, {"one": True})),
        step_recording("two", records, second_step),
    ]
    cap_one = StepPolicy(timeout_seconds=30.0, attempt_cap=1, backoff_base_seconds=0.0)
    first = RunnerHarness(store, {"demo": steps()}, policies={"generic": cap_one})
    job_id, _ = enqueue(store, "demo", "demo:cap-restart", "cap-restart.pdf")
    first.start()
    try:
        first.enqueue_and_notify(store, job_id)
        assert wait_until(lambda: records.get("two", []) == [1]), records
    finally:
        first.stop()
        gate.set()

    status = store.load_job_status(job_id)
    assert status is not None
    assert isinstance(status.policy, dict)
    recorded_generic = status.policy.get("generic")
    assert isinstance(recorded_generic, dict)
    assert recorded_generic.get("attempt_cap") == 1

    # Restart with looser defaults: the recorded one-attempt cap still applies.
    second = RunnerHarness(
        store, {"demo": steps()}, policies={"generic": FAST_POLICY}, backoff_sleep=lambda d: None
    )
    second.start()
    try:
        second.runner.notify()
        assert wait_until(
            lambda: store.load_job_status(job_id).status == "failed"  # type: ignore[union-attr]
        ), store.load_job_status(job_id)
    finally:
        second.stop()

    assert records["two"] == [1], "resumed work must not exceed the recorded cap"
    status = store.load_job_status(job_id)
    assert status is not None
    assert "attempt cap of 1" in (status.failure_reason or "")


def test_permanent_failure_is_not_retried(store: PersistenceStore) -> None:
    records: dict[str, list[int]] = {}

    def permanent(context: StepContext) -> StepOutcome:
        raise PermanentStepFailure("unsupported document")

    runner = RunnerHarness(
        store,
        {"demo": [step_recording("forever", records, permanent)]},
        policies={"generic": FAST_POLICY},
        backoff_sleep=lambda d: None,
    )
    job_id, _ = enqueue(store, "demo", "demo:permanent", "permanent.pdf")
    runner.start()
    try:
        runner.enqueue_and_notify(store, job_id)
        assert wait_until(
            lambda: store.load_job_status(job_id).status == "failed"  # type: ignore[union-attr]
        ), store.load_job_status(job_id)
    finally:
        runner.stop()

    assert records["forever"] == [1]
    status = store.load_job_status(job_id)
    assert status is not None
    assert "failed permanently" in (status.failure_reason or "")


def test_unknown_paid_call_outcome_is_not_retried(store: PersistenceStore) -> None:
    records: dict[str, list[int]] = {}

    def unknown(context: StepContext) -> StepOutcome:
        raise PaidCallOutcomeUnknown("provider did not acknowledge the request")

    runner = RunnerHarness(
        store,
        {"demo": [step_recording("paid", records, unknown)]},
        policies={"generic": FAST_POLICY},
        backoff_sleep=lambda d: None,
    )
    job_id, _ = enqueue(store, "demo", "demo:paid-unknown", "paid-unknown.pdf")
    runner.start()
    try:
        runner.enqueue_and_notify(store, job_id)
        assert wait_until(
            lambda: store.load_job_status(job_id).status == "failed"  # type: ignore[union-attr]
        ), store.load_job_status(job_id)
    finally:
        runner.stop()

    assert records["paid"] == [1], "an unknown paid-call outcome must never auto-retry"
    status = store.load_job_status(job_id)
    assert status is not None
    assert "review required" in (status.failure_reason or "")


def test_changed_step_inputs_fail_with_a_clear_conflict_on_resume(
    store: PersistenceStore,
) -> None:
    records: dict[str, list[int]] = {}
    settings = {"parser": "v1"}
    gate = threading.Event()

    def third_step(context: StepContext) -> StepOutcome:
        if context.attempt_number == 1:
            gate.wait(10)
        return StepOutcome(1, {"three": True})

    def second_step(context: StepContext) -> StepOutcome:
        records.setdefault("two", []).append(context.attempt_number)
        return StepOutcome(1, {"two": True})

    def make_steps() -> list[StepDefinition]:
        return [
            step_recording("one", records, lambda context: StepOutcome(1, {"one": True})),
            StepDefinition(
                "two",
                second_step,
                inputs=lambda context: {"parser": settings["parser"]},
            ),
            step_recording("three", records, third_step),
        ]

    run = RunnerHarness(
        store, {"demo": make_steps()}, policies={"generic": StepPolicy(30.0, 2, 0.0)}
    )
    job_id, _ = enqueue(store, "demo", "demo:changed-inputs", "changed-inputs.pdf")
    run.start()
    try:
        run.enqueue_and_notify(store, job_id)
        assert wait_until(lambda: records.get("three", []) == [1]), records
        assert [step.step_key for step in store.load_job_steps(job_id)] == ["one", "two"]
    finally:
        run.stop()
        gate.set()

    settings["parser"] = "v2"

    resumed = RunnerHarness(store, {"demo": make_steps()}, policies={"generic": FAST_POLICY})
    resumed.start()
    try:
        resumed.runner.notify()
        assert wait_until(
            lambda: store.load_job_status(job_id).status == "failed"  # type: ignore[union-attr]
        ), store.load_job_status(job_id)
    finally:
        resumed.stop()

    status = store.load_job_status(job_id)
    assert status is not None
    assert "inputs changed" in (status.failure_reason or ""), status.failure_reason
    assert records["three"] == [1], "changed work must not silently reuse the old result"


def test_finish_race_does_not_stop_later_jobs(store: PersistenceStore) -> None:
    """Cancellation between the final checkpoint and completion must not kill
    the runner or leave later jobs queued."""
    records: dict[str, list[int]] = {}
    cancelled_once = {"done": False}

    def cancel_raced_commit(persist: PersistenceStore, commit: StepCommitContext) -> None:
        commit_checkpoint(persist, commit)
        if not cancelled_once["done"]:
            cancelled_once["done"] = True
            persist.cancel_job(commit.job_id, "cancelled at the finish line")

    first_steps = [
        StepDefinition(
            "one",
            lambda context: StepOutcome(1, {"one": True}),
            commit=cancel_raced_commit,
        )
    ]
    second_steps = [step_recording("quick", records, lambda context: StepOutcome(1, {}))]

    runner = RunnerHarness(
        store,
        {"demo": first_steps, "quick": second_steps},
        policies={"generic": FAST_POLICY},
        backoff_sleep=lambda d: None,
    )
    first_job, _ = enqueue(store, "demo", "demo:finish-race", "finish-race.pdf")
    second_job, _ = enqueue(store, "quick", "demo:after-race", "after-race.pdf")
    runner.start()
    try:
        runner.enqueue_and_notify(store, first_job)
        assert wait_until(lambda: cancelled_once["done"]), "commit path never ran"
        runner.enqueue_and_notify(store, second_job)
        assert wait_until(
            lambda: store.load_job_status(second_job).status == "completed"  # type: ignore[union-attr]
        ), store.load_job_status(second_job)
        assert runner.runner.snapshot().state != "stopping"
    finally:
        runner.stop()

    first_status = store.load_job_status(first_job)
    assert first_status is not None
    assert first_status.status == "cancelled"
