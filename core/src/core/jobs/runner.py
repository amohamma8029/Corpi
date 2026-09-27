"""Durable background job runner (subproblem 1.3, issue #9).

One runner per core process — the single-core data-directory lock guarantees no
second core shares the database. The runner claims one job at a time, executes
its steps on a bounded worker pool, and commits each step's completion record
together with its output. On restart, completion is derived from committed
``job_steps`` rows (never from mutable status flags), so interrupted work
resumes without duplicating committed results.

Timeouts bound how long the runner waits for an attempt; they do not stop a
running thread. Steps that need a guaranteed stop must run as terminable tools
(a subprocess) when 1.4-1.6 add them; cooperative steps check ``is_cancelled``
at safe points. If every worker stays occupied past its attempt deadline, the
runner stops claiming new jobs, reports the stall, and annotates queued jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import JobStatus, JSONValue, new_id
from core.persistence.store import PersistenceStore


@dataclass(frozen=True, slots=True)
class StepPolicy:
    """Starting defaults from issue #9's decision 5, configurable per step class."""

    timeout_seconds: float
    attempt_cap: int
    backoff_base_seconds: float
    backoff_factor: float = 2.0
    backoff_max_seconds: float = 30.0
    jitter: float = 0.25


STEP_POLICIES: dict[str, StepPolicy] = {
    "model_call": StepPolicy(120.0, 3, 2.0),
    "ocr_page": StepPolicy(60.0, 2, 1.0, backoff_max_seconds=5.0),
    "parse": StepPolicy(300.0, 2, 2.0),
    "embedding_batch": StepPolicy(120.0, 2, 2.0),
    "index_build": StepPolicy(300.0, 2, 2.0),
    "file_write": StepPolicy(60.0, 2, 1.0, backoff_max_seconds=10.0),
    "generic": StepPolicy(120.0, 2, 2.0),
}


class JobStepCancelled(Exception):
    """Raised by a cooperative step that observed cancellation."""


@dataclass(frozen=True, slots=True)
class StepOutcome:
    checkpoint_format_version: int
    checkpoint: JSONValue


@dataclass(frozen=True, slots=True)
class StepContext:
    job_id: str
    step_key: str
    attempt_number: int
    is_cancelled: Callable[[], bool]


StepRun = Callable[[StepContext], StepOutcome]
StepCommit = Callable[[PersistenceStore, str, str, StepOutcome], None]


def commit_checkpoint(
    store: PersistenceStore,
    job_id: str,
    step_key: str,
    outcome: StepOutcome,
) -> None:
    """Default commit: a completion record with its checkpoint and no version output."""
    store.commit_step(job_id, step_key, outcome.checkpoint_format_version, outcome.checkpoint)


@dataclass(frozen=True, slots=True)
class StepDefinition:
    step_key: str
    run: StepRun
    policy_class: str = "generic"
    commit: StepCommit = commit_checkpoint


@dataclass(slots=True)
class RunnerSnapshot:
    state: str  # "idle" | "running" | "stalled" | "stopping"
    blocked_reason: str | None
    claimed_job_id: str | None
    queued_job_ids: tuple[str, ...]


def _default_owner_instance_id() -> str:
    return f"core-{new_id()}"


def _iso_seconds_ago(seconds: float) -> str:
    moment = datetime.now(UTC) - timedelta(seconds=seconds)
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


class JobRunner:
    """Claims queued jobs and executes their steps with durable checkpoints."""

    def __init__(
        self,
        store: PersistenceStore,
        step_sets: Mapping[str, Iterable[StepDefinition]],
        *,
        owner_instance_id: str | None = None,
        pool_size: int | None = None,
        policies: Mapping[str, StepPolicy] | None = None,
        poll_interval_seconds: float = 0.25,
        orphan_heartbeat_seconds: float = 60.0,
        backoff_sleep: Callable[[float], Awaitable[None] | None] | None = None,
    ) -> None:
        self._store = store
        self._step_sets = {kind: tuple(steps) for kind, steps in step_sets.items()}
        self._owner = owner_instance_id or _default_owner_instance_id()
        self._pool_size = pool_size or max(1, min(4, os.cpu_count() or 1))
        self._policies: dict[str, StepPolicy] = {**STEP_POLICIES, **dict(policies or {})}
        self._poll_interval = poll_interval_seconds
        self._orphan_heartbeat_seconds = orphan_heartbeat_seconds
        self._backoff_sleep = backoff_sleep or asyncio.sleep
        self._executor = ThreadPoolExecutor(
            max_workers=self._pool_size, thread_name_prefix="corpi-job"
        )
        self._wakeup: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = threading.Event()
        self._claimed_job_id: str | None = None
        self._blocked_reason: str | None = None
        self._inflight: dict[tuple[str, str, int], float] = {}
        self._inflight_lock = threading.Lock()
        self._annotated_jobs: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("runner is already started")
        self._wakeup = asyncio.Event()
        self._rescue_orphaned_jobs()
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._stopping.set()
        if self._wakeup is not None:
            self._wakeup.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._executor.shutdown(wait=False, cancel_futures=True)

    def notify(self) -> None:
        """Wake the runner after new work is enqueued."""
        if self._wakeup is not None:
            self._wakeup.set()

    def snapshot(self) -> RunnerSnapshot:
        return RunnerSnapshot(
            state=self._state_name(),
            blocked_reason=self._blocked_reason,
            claimed_job_id=self._claimed_job_id,
            queued_job_ids=tuple(job.job_id for job in self._store.list_claimable_jobs()),
        )

    def _state_name(self) -> str:
        if self._stopping.is_set():
            return "stopping"
        if self._blocked_reason is not None:
            return "stalled"
        if self._claimed_job_id is not None:
            return "running"
        return "idle"

    def _rescue_orphaned_jobs(self) -> None:
        cutoff = _iso_seconds_ago(self._orphan_heartbeat_seconds)
        self._store.release_orphaned_jobs(cutoff)

    # -- main loop ---------------------------------------------------------

    async def _run_forever(self) -> None:
        assert self._wakeup is not None
        while not self._stopping.is_set():
            stall_reason = self._stall_reason()
            if stall_reason is not None:
                self._enter_stall(stall_reason)
                await asyncio.sleep(self._poll_interval)
                continue
            if self._blocked_reason is not None:
                self._exit_stall()
            job = self._claim_next_job()
            if job is None:
                self._wakeup.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wakeup.wait(), timeout=self._poll_interval)
                continue
            await self._run_job(job)

    def _claim_next_job(self) -> JobStatus | None:
        for candidate in self._store.list_claimable_jobs():
            try:
                claimed = self._store.claim_job(candidate.job_id, self._owner)
            except ConflictingWriteError:
                continue
            self._claimed_job_id = claimed.job_id
            return claimed
        return None

    async def _run_job(self, status: JobStatus) -> None:
        job_id = status.job_id
        try:
            job = self._store.load_job_by_id(job_id)
            steps = self._step_sets.get(job.kind)
            if steps is None:
                self._store.finish_job(
                    job_id, "failed", f"no step set is registered for job kind {job.kind!r}"
                )
                return
            committed = {step.step_key for step in self._store.load_job_steps(job_id)}
            for step in steps:
                if self._stopping.is_set():
                    return
                if step.step_key in committed:
                    continue  # reuse: committed completion records are the proof
                if not await self._run_step(job_id, step):
                    return
            self._store.finish_job(job_id, "completed")
        finally:
            self._claimed_job_id = None

    async def _run_step(self, job_id: str, step: StepDefinition) -> bool:
        """Run one step with retries. Returns False when the job must stop."""
        policy = self._policies.get(step.policy_class, self._policies["generic"])
        for attempt_number in range(1, policy.attempt_cap + 1):
            current = self._store.load_job_status(job_id)
            if current is None or current.status != "running":
                return False
            attempt = self._store.start_step_attempt(job_id, step.step_key)
            context = StepContext(
                job_id=job_id,
                step_key=step.step_key,
                attempt_number=attempt.attempt_number,
                is_cancelled=lambda: self._observed_cancelled(job_id),
            )
            loop = asyncio.get_running_loop()
            key = (job_id, step.step_key, attempt.attempt_number)
            # Track the WORKER future (a timed-out await does not free the
            # thread), so a fully stuck pool stays visible until threads end.
            worker_future = self._executor.submit(step.run, context)
            self._begin_inflight(key, policy)
            worker_future.add_done_callback(
                lambda _future, done_key=key: self._end_inflight(done_key)
            )
            future = asyncio.wrap_future(worker_future, loop=loop)
            try:
                outcome = await asyncio.wait_for(future, timeout=policy.timeout_seconds)
            except TimeoutError:
                # The worker thread keeps running; the inflight entry stays until
                # the thread truly finishes, so a fully stuck pool is detected.
                self._finish_attempt(
                    job_id, step.step_key, attempt.attempt_number, "attempt timed out"
                )
                if attempt_number >= policy.attempt_cap:
                    self._fail_job(job_id, step, attempt_number, "timed out")
                    return False
                await self._backoff(policy, attempt_number)
                continue
            except JobStepCancelled:
                self._finish_attempt(job_id, step.step_key, attempt.attempt_number, "cancelled")
                return False
            except Exception as error:  # noqa: BLE001 - failures are recorded, not raised
                self._finish_attempt(
                    job_id,
                    step.step_key,
                    attempt.attempt_number,
                    f"{type(error).__name__}: {error}",
                )
                if attempt_number >= policy.attempt_cap:
                    self._fail_job(job_id, step, attempt_number, "failed")
                    return False
                await self._backoff(policy, attempt_number)
                continue

            try:
                step.commit(self._store, job_id, step.step_key, outcome)
            except ConflictingWriteError:
                # The guarded commit transaction saw a cancellation or a
                # finished job: the late result must not be recorded.
                self._finish_attempt(
                    job_id, step.step_key, attempt.attempt_number, "commit rejected"
                )
                return False
            self._finish_attempt(job_id, step.step_key, attempt.attempt_number, None)
            return True
        return False

    def _observed_cancelled(self, job_id: str) -> bool:
        status = self._store.load_job_status(job_id)
        return status is None or status.status == "cancelled"

    def _finish_attempt(
        self,
        job_id: str,
        step_key: str,
        attempt_number: int,
        failure_reason: str | None,
    ) -> None:
        self._store.finish_step_attempt(job_id, step_key, attempt_number, failure_reason)

    def _fail_job(self, job_id: str, step: StepDefinition, attempts: int, outcome: str) -> None:
        self._store.finish_job(
            job_id,
            "failed",
            f"step {step.step_key} {outcome} after {attempts} attempt(s)",
        )

    async def _backoff(self, policy: StepPolicy, attempt_number: int) -> None:
        delay = min(
            policy.backoff_max_seconds,
            policy.backoff_base_seconds * (policy.backoff_factor ** (attempt_number - 1)),
        )
        jitter = delay * policy.jitter * random.uniform(-1.0, 1.0)
        result = self._backoff_sleep(max(0.0, delay + jitter))
        if result is not None:
            await result

    # -- stuck-worker detection -------------------------------------------

    def _begin_inflight(self, key: tuple[str, str, int], policy: StepPolicy) -> None:
        with self._inflight_lock:
            self._inflight[key] = time.monotonic() + policy.timeout_seconds

    def _end_inflight(self, key: tuple[str, str, int]) -> None:
        with self._inflight_lock:
            self._inflight.pop(key, None)

    def _stall_reason(self) -> str | None:
        """All workers occupied past their attempt deadline: report, do not wait silently."""
        now = time.monotonic()
        with self._inflight_lock:
            if len(self._inflight) < self._pool_size:
                return None
            if any(deadline > now for deadline in self._inflight.values()):
                return None
        return (
            f"all {self._pool_size} worker(s) are occupied by timed-out attempts; "
            "queued jobs cannot start until the stuck steps finish or the core restarts"
        )

    def _enter_stall(self, reason: str) -> None:
        self._blocked_reason = reason
        for job in self._store.list_claimable_jobs():
            try:
                self._store.annotate_queued_job(job.job_id, reason)
                self._annotated_jobs.add(job.job_id)
            except ConflictingWriteError:
                pass

    def _exit_stall(self) -> None:
        self._blocked_reason = None
        for job_id in tuple(self._annotated_jobs):
            try:
                self._store.annotate_queued_job(job_id, None)
            except ConflictingWriteError:
                pass
        self._annotated_jobs.clear()
