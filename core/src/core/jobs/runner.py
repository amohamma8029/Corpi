"""Durable background job runner (subproblem 1.3, issue #9).

One runner per core process — the single-core data-directory lock guarantees no
second core shares the database. The runner claims one job at a time, executes
its steps on a bounded worker pool, and commits each step's completion record
together with its output after verifying the attempt guard in the same
database transaction. On restart, completion is derived from committed
``job_steps`` rows (never from mutable status flags), so interrupted work
resumes without duplicating committed results.

Timeouts bound how long the runner waits for an attempt; they do not stop a
running thread. Steps that need a guaranteed stop must run as terminable tools
(a subprocess) when 1.4-1.6 add them; cooperative steps poll ``should_stop``,
which reports cancellation, runner shutdown, and the attempt deadline.

All database operations run on a dedicated single-thread executor so a busy
database never blocks the event loop, API requests, or the processing workers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from typing import Any

from core.persistence.errors import ConflictingWriteError
from core.persistence.models import (
    AttemptGuard,
    Job,
    JobStatus,
    JSONValue,
    new_id,
)
from core.persistence.store import PersistenceStore

logger = logging.getLogger(__name__)

WAITING_FOR_IMPLEMENTATION = {
    "pdf-extract": "PDF text extraction is waiting for the M1.5 processing steps.",
}


class PermanentStepFailure(Exception):
    """A step failure that must not be retried (invalid input, unsupported data)."""


class PaidCallOutcomeUnknown(Exception):
    """A paid call whose outcome is unknown; never retried automatically."""


class JobStepCancelled(Exception):
    """Raised by a cooperative step that observed cancellation or shutdown."""


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
    should_stop: Callable[[], bool]


StepRun = Callable[[StepContext], StepOutcome]
StepInputs = Callable[[StepContext], JSONValue]


@dataclass(frozen=True, slots=True)
class StepCommitContext:
    job_id: str
    step_key: str
    guard: AttemptGuard
    inputs_sha256: str
    outcome: StepOutcome


StepCommit = Callable[[PersistenceStore, StepCommitContext], None]


def commit_checkpoint(store: PersistenceStore, commit: StepCommitContext) -> None:
    """Default commit: completion record and checkpoint with no version output."""
    store.commit_step(
        commit.job_id,
        commit.step_key,
        commit.outcome.checkpoint_format_version,
        commit.outcome.checkpoint,
        guard=commit.guard,
        inputs_sha256=commit.inputs_sha256,
    )


@dataclass(frozen=True, slots=True)
class StepDefinition:
    step_key: str
    run: StepRun
    policy_class: str = "generic"
    commit: StepCommit = commit_checkpoint
    inputs: StepInputs | None = None
    paid: bool = False


@dataclass(slots=True)
class RunnerSnapshot:
    state: str  # "idle" | "running" | "stalled" | "stopping"
    blocked_reason: str | None
    claimed_job_id: str | None
    queued_job_ids: tuple[str, ...]
    pending_failure_jobs: tuple[str, ...]


def _default_owner_instance_id() -> str:
    return f"core-{new_id()}"


def _canonical_json(value: JSONValue) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def default_inputs_fingerprint(job: Job, step: StepDefinition) -> str:
    """Fingerprint of the stable inputs a step runs with unless it overrides."""
    payload: JSONValue = {
        "kind": job.kind,
        "source_id": job.source_id,
        "source_version_id": job.source_version_id,
        "extraction_version_id": job.extraction_version_id,
        "index_version_id": job.index_version_id,
        "step_key": step.step_key,
        "policy_class": step.policy_class,
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _policy_number(entry: dict[str, JSONValue], key: str, default: float) -> float:
    value = entry.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"retry policy field {key!r} is not a number")
    return float(value)


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
        backoff_sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self._store = store
        self._step_sets = {kind: tuple(steps) for kind, steps in step_sets.items()}
        self._owner = owner_instance_id or _default_owner_instance_id()
        self._pool_size = pool_size or max(1, min(4, os.cpu_count() or 1))
        self._policies: dict[str, StepPolicy] = {**STEP_POLICIES, **dict(policies or {})}
        self._poll_interval = poll_interval_seconds
        self._backoff_sleep = backoff_sleep or asyncio.sleep
        self._executor = ThreadPoolExecutor(
            max_workers=self._pool_size, thread_name_prefix="corpi-job"
        )
        # Database operations get their own single-worker capacity so a busy
        # or locked database never blocks the event loop or the step workers.
        self._db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpi-db")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wakeup: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = threading.Event()
        self._claimed_job_id: str | None = None
        self._blocked_reason: str | None = None
        self._inflight: dict[tuple[str, str, int], float] = {}
        self._inflight_lock = threading.Lock()
        self._annotated_jobs: set[str] = set()
        self._pending_job_failures: list[tuple[str, str]] = []

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("runner is already started")
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        # The exclusive data-directory lock proves a previous core is gone, so
        # every job still marked running belongs to that dead instance and can
        # be requeued immediately — no heartbeat-age cutoff.
        await self._db(self._store.release_running_jobs)
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
        self._loop = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._db_executor.shutdown(wait=False, cancel_futures=True)

    def notify(self) -> None:
        """Wake the runner after new work is enqueued."""
        loop = self._loop
        wakeup = self._wakeup
        if loop is not None and wakeup is not None:
            loop.call_soon_threadsafe(wakeup.set)

    def snapshot(self) -> RunnerSnapshot:
        return RunnerSnapshot(
            state=self._state_name(),
            blocked_reason=self._blocked_reason,
            claimed_job_id=self._claimed_job_id,
            queued_job_ids=tuple(job.job_id for job in self._store.list_claimable_jobs()),
            pending_failure_jobs=tuple(job_id for job_id, _ in self._pending_job_failures),
        )

    def _state_name(self) -> str:
        if self._stopping.is_set():
            return "stopping"
        if self._blocked_reason is not None:
            return "stalled"
        if self._claimed_job_id is not None:
            return "running"
        return "idle"

    async def _db(self, function: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._db_executor, function, *args)

    # -- main loop ---------------------------------------------------------

    async def _run_forever(self) -> None:
        assert self._wakeup is not None
        while not self._stopping.is_set():
            try:
                await self._loop_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - expected errors must not stop later work
                logger.exception("job runner iteration failed; continuing")
                await asyncio.sleep(self._poll_interval)

    async def _loop_once(self) -> None:
        assert self._wakeup is not None
        await self._drain_pending_failures()
        stall_reason = self._stall_reason()
        if stall_reason is not None:
            await self._db(self._enter_stall, stall_reason)
            await asyncio.sleep(self._poll_interval)
            return
        if self._blocked_reason is not None:
            await self._db(self._exit_stall)
        job = await self._claim_next_job()
        if job is None:
            self._wakeup.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=self._poll_interval)
            return
        await self._run_job(job)

    async def _claim_next_job(self) -> JobStatus | None:
        candidates = await self._db(self._store.list_claimable_jobs)
        for candidate in candidates:
            try:
                job = await self._db(self._store.load_job_by_id, candidate.job_id)
            except Exception:  # noqa: BLE001 - unknown jobs are simply not claimable
                continue
            waiting_reason = WAITING_FOR_IMPLEMENTATION.get(job.kind)
            if waiting_reason is not None and job.kind not in self._step_sets:
                # M1.4 can persist the saved original and queued handoff before
                # M1.5 provides extraction. Keep this known kind visibly queued;
                # genuinely unknown kinds retain the ordinary failed-job path.
                if candidate.failure_reason != waiting_reason:
                    try:
                        await self._db(
                            self._store.annotate_queued_job, candidate.job_id, waiting_reason
                        )
                    except ConflictingWriteError:
                        pass
                continue
            policy_payload = self._policy_payload(job)
            try:
                claimed = await self._db(
                    partial(
                        self._store.claim_job,
                        candidate.job_id,
                        self._owner,
                        policy=policy_payload,
                    )
                )
            except ConflictingWriteError:
                continue
            self._claimed_job_id = claimed.job_id
            return claimed
        return None

    def _policy_payload(self, job: Job) -> JSONValue:
        """The resolved step-class policies for a job kind, as storable JSON."""
        steps = self._step_sets.get(job.kind)
        if steps is None:
            return {}
        payload: dict[str, JSONValue] = {}
        for step in steps:
            policy = self._policies.get(step.policy_class, self._policies["generic"])
            payload[step.policy_class] = {
                "timeout_seconds": policy.timeout_seconds,
                "attempt_cap": policy.attempt_cap,
                "backoff_base_seconds": policy.backoff_base_seconds,
                "backoff_factor": policy.backoff_factor,
                "backoff_max_seconds": policy.backoff_max_seconds,
                "jitter": policy.jitter,
            }
        return payload

    def _policy_for(self, status: JobStatus, step: StepDefinition) -> StepPolicy:
        """Use the policy recorded at claim so a resumed job keeps its allowance."""
        recorded = status.policy
        if isinstance(recorded, dict):
            entry = recorded.get(step.policy_class)
            if isinstance(entry, dict):
                try:
                    return StepPolicy(
                        timeout_seconds=_policy_number(entry, "timeout_seconds", 120.0),
                        attempt_cap=int(_policy_number(entry, "attempt_cap", 2.0)),
                        backoff_base_seconds=_policy_number(entry, "backoff_base_seconds", 2.0),
                        backoff_factor=_policy_number(entry, "backoff_factor", 2.0),
                        backoff_max_seconds=_policy_number(entry, "backoff_max_seconds", 30.0),
                        jitter=_policy_number(entry, "jitter", 0.25),
                    )
                except ValueError:
                    logger.warning("recorded retry policy is invalid; using defaults")
        return self._policies.get(step.policy_class, self._policies["generic"])

    async def _run_job(self, status: JobStatus) -> None:
        job_id = status.job_id
        try:
            job = await self._db(self._store.load_job_by_id, job_id)
            steps = self._step_sets.get(job.kind)
            if steps is None:
                await self._db(
                    self._store.finish_job,
                    job_id,
                    "failed",
                    f"no step set is registered for job kind {job.kind!r}",
                )
                return
            committed_rows = await self._db(self._store.load_job_steps, job_id)
            committed = {row.step_key: row for row in committed_rows}
            for step in steps:
                if self._stopping.is_set():
                    return
                existing = committed.get(step.step_key)
                if existing is not None:
                    fingerprint = await self._fingerprint(job, step)
                    if existing.inputs_sha256 is not None and existing.inputs_sha256 != fingerprint:
                        await self._db(
                            self._store.finish_job,
                            job_id,
                            "failed",
                            f"step {step.step_key} inputs changed; "
                            "start new work under a new job key",
                        )
                        return
                    continue  # reuse: the committed completion record is the proof
                if not await self._run_step(status, job, step):
                    return
            try:
                await self._db(self._store.finish_job, job_id, "completed")
            except ConflictingWriteError:
                # Cancellation or a competing transition won between the final
                # checkpoint and completion: the job is no longer ours to finish.
                logger.info("job %s could not be marked complete (state changed)", job_id)
        except Exception as error:  # noqa: BLE001 - the job must reach a truthful state
            # A runner-side failure (for example a transient database error)
            # must not abandon the job as `running`: record a clear failed
            # state so it is visible and never silently stuck. When the state
            # changed underneath (already cancelled, for example) the finish
            # attempt is refused and suppressed — that state is truthful too.
            logger.exception("job %s failed with a runner error", job_id)
            await self._record_job_failure(job_id, f"runner error: {type(error).__name__}")
        finally:
            self._claimed_job_id = None

    async def _record_job_failure(self, job_id: str, reason: str) -> None:
        """Record a failed outcome, keeping it pending when the write fails.

        A database error while recording the failure must not abandon the job
        as `running`: the outcome stays pending and is retried on every loop
        iteration until it is durable (or the job reached a truthful state
        another way).
        """
        try:
            await self._db(self._store.finish_job, job_id, "failed", reason)
        except ConflictingWriteError:
            pass  # the job already reached a truthful state
        except Exception:  # noqa: BLE001 - the outcome is kept pending instead
            logger.exception("could not record failure for job %s; keeping it pending", job_id)
            self._pending_job_failures.append((job_id, reason))

    async def _drain_pending_failures(self) -> None:
        if not self._pending_job_failures:
            return
        remaining: list[tuple[str, str]] = []
        for job_id, reason in self._pending_job_failures:
            try:
                await self._db(self._store.finish_job, job_id, "failed", reason)
            except ConflictingWriteError:
                pass  # already truthful (for example cancelled underneath)
            except Exception:  # noqa: BLE001 - stay pending until it can be recorded
                remaining.append((job_id, reason))
        self._pending_job_failures = remaining

    async def _fingerprint(self, job: Job, step: StepDefinition) -> str:
        if step.inputs is None:
            return default_inputs_fingerprint(job, step)
        probe = StepContext(
            job_id=job.id,
            step_key=step.step_key,
            attempt_number=0,
            is_cancelled=lambda: False,
            should_stop=lambda: False,
        )
        try:
            payload = step.inputs(probe)
        except Exception:  # noqa: BLE001 - unreadable inputs are handled by the step
            return default_inputs_fingerprint(job, step)
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    async def _run_step(self, status: JobStatus, job: Job, step: StepDefinition) -> bool:
        """Run one step with retries. Returns False when the job must stop."""
        policy = self._policy_for(status, step)
        fingerprint = await self._fingerprint(job, step)
        while True:
            attempts = await self._db(self._store.load_step_attempts, job.id, step.step_key)
            spent = len(attempts)
            if spent >= policy.attempt_cap:
                await self._db(
                    self._store.finish_job,
                    job.id,
                    "failed",
                    f"step {step.step_key} exhausted its attempt cap of {policy.attempt_cap}",
                )
                return False
            if step.paid:
                unfinished = [item for item in attempts if item.finished_at is None]
                unknown_done = [
                    item
                    for item in attempts
                    if item.finished_at is not None and item.outcome == "unknown"
                ]
                if unfinished or unknown_done:
                    # A crash left an unresolved call, or an attempt already
                    # recorded an unknown outcome (even with a finish
                    # timestamp): the call's outcome is unknown, so it is never
                    # repeated automatically — flag it for review.
                    for item in unfinished:
                        with contextlib.suppress(ConflictingWriteError):
                            await self._db(
                                partial(
                                    self._store.finish_step_attempt,
                                    job.id,
                                    step.step_key,
                                    item.attempt_number,
                                    "unresolved; paid-call outcome unknown",
                                    outcome="unknown",
                                )
                            )
                    reason = (
                        f"step {step.step_key} has an unresolved paid-call attempt; review required"
                        if unfinished
                        else f"step {step.step_key} has a recorded unknown paid-call "
                        "outcome; review required"
                    )
                    await self._db(self._store.finish_job, job.id, "failed", reason)
                    return False
            current = await self._db(self._store.load_job_status, job.id)
            if current is None or current.status != "running":
                return False
            attempt = await self._db(self._store.start_step_attempt, job.id, step.step_key)
            deadline = time.monotonic() + policy.timeout_seconds
            context = StepContext(
                job_id=job.id,
                step_key=step.step_key,
                attempt_number=attempt.attempt_number,
                is_cancelled=lambda: self._observed_cancelled(job.id),
                should_stop=lambda deadline=deadline: (
                    self._stopping.is_set()
                    or time.monotonic() >= deadline
                    or self._observed_cancelled(job.id)
                ),
            )
            key = (job.id, step.step_key, attempt.attempt_number)
            worker_future = self._executor.submit(step.run, context)
            self._begin_inflight(key, policy)
            worker_future.add_done_callback(
                lambda _future, done_key=key: self._end_inflight(done_key)
            )
            try:
                outcome = await asyncio.wait_for(
                    asyncio.wrap_future(worker_future), timeout=policy.timeout_seconds
                )
            except TimeoutError:
                # The worker thread keeps running; the inflight entry stays
                # until it truly finishes, so a fully stuck pool is detected.
                if step.paid:
                    # One transaction records the unknown outcome and fails the
                    # job. Even a crash between writes can never make this call
                    # replayable: the attempt keeps outcome 'unknown' and
                    # recovery checks that, finish timestamp or not.
                    with contextlib.suppress(ConflictingWriteError):
                        await self._db(
                            self._store.record_unknown_paid_outcome,
                            job.id,
                            step.step_key,
                            attempt.attempt_number,
                            "timed out; paid-call outcome unknown",
                            f"step {step.step_key} timed out during a paid call; "
                            "outcome unknown, review required",
                        )
                    return False
                await self._db(
                    self._store.finish_step_attempt,
                    job.id,
                    step.step_key,
                    attempt.attempt_number,
                    "attempt timed out",
                )
                if attempt.attempt_number >= policy.attempt_cap:
                    await self._db(
                        self._store.finish_job,
                        job.id,
                        "failed",
                        f"step {step.step_key} timed out after {attempt.attempt_number} attempt(s)",
                    )
                    return False
                await self._backoff(policy, attempt.attempt_number)
                continue
            except JobStepCancelled:
                await self._db(
                    self._store.finish_step_attempt,
                    job.id,
                    step.step_key,
                    attempt.attempt_number,
                    "cancelled",
                )
                return False
            except PaidCallOutcomeUnknown as error:
                # One transaction records outcome 'unknown' on the attempt and
                # fails the job: a crash cannot leave recovery treating this
                # call as safe to repeat (same guarantee as the timeout path).
                with contextlib.suppress(ConflictingWriteError):
                    await self._db(
                        self._store.record_unknown_paid_outcome,
                        job.id,
                        step.step_key,
                        attempt.attempt_number,
                        f"paid-call outcome unknown; not retried automatically: {error}",
                        f"step {step.step_key} has an unknown paid-call outcome; review required",
                    )
                return False
            except PermanentStepFailure as error:
                await self._db(
                    self._store.finish_step_attempt,
                    job.id,
                    step.step_key,
                    attempt.attempt_number,
                    f"PermanentStepFailure: {error}",
                )
                await self._db(
                    self._store.finish_job,
                    job.id,
                    "failed",
                    f"step {step.step_key} failed permanently: {error}",
                )
                return False
            except Exception as error:  # noqa: BLE001 - failures are recorded, not raised
                await self._db(
                    self._store.finish_step_attempt,
                    job.id,
                    step.step_key,
                    attempt.attempt_number,
                    f"{type(error).__name__}: {error}",
                )
                if attempt.attempt_number >= policy.attempt_cap:
                    await self._db(
                        self._store.finish_job,
                        job.id,
                        "failed",
                        f"step {step.step_key} failed after {attempt.attempt_number} attempt(s)",
                    )
                    return False
                await self._backoff(policy, attempt.attempt_number)
                continue

            try:
                await self._db(
                    step.commit,
                    self._store,
                    StepCommitContext(
                        job_id=job.id,
                        step_key=step.step_key,
                        guard=AttemptGuard(self._owner, step.step_key, attempt.attempt_number),
                        inputs_sha256=fingerprint,
                        outcome=outcome,
                    ),
                )
            except ConflictingWriteError:
                # The guarded commit saw a cancellation, a superseded attempt,
                # or a finished job: the late result must not be recorded.
                await self._db(
                    self._store.finish_step_attempt,
                    job.id,
                    step.step_key,
                    attempt.attempt_number,
                    "commit rejected",
                )
                return False
            await self._db(
                self._store.finish_step_attempt, job.id, step.step_key, attempt.attempt_number, None
            )
            return True

    def _observed_cancelled(self, job_id: str) -> bool:
        status = self._store.load_job_status(job_id)
        return status is None or status.status == "cancelled"

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
