"""Runner-owned search preparation with durable batches and conservative time leases."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from core.jobs.runner import (
    SEARCH_PREPARATION_SECONDS,
    JobStepCancelled,
    PermanentStepFailure,
    StepCommitContext,
    StepContext,
    StepDefinition,
    StepOutcome,
)
from core.persistence.errors import ConflictingWriteError
from core.persistence.models import AttemptGuard, JSONValue, new_id, utc_timestamp
from core.persistence.store import PersistenceStore, _verify_attempt_current
from core.search.chunking import CHUNKER_VERSION, eligible_passages, make_chunks
from core.search.profiles import ModelProfile, profiles
from core.search.representation import CANONICAL, settings_for
from core.search.runtime import InputTokenizer
from core.search.storage import SearchStorage, generation_identity
from core.search.supervisor import (
    EmbeddingFailure,
    EmbeddingSession,
    EmbeddingStopped,
    openblas_options,
)

PLAN = "plan-search"
BATCH_BUDGET_POLICY = "batch-reservation-v1"


def _release_planning_heap() -> str:
    """Return unused GNU allocator pages before inference, inside its metadata lease."""
    if sys.platform != "linux":
        return "not_applicable"
    import ctypes
    import gc

    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (OSError, AttributeError):
        return "not_available"
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return "released" if trim(0) else "unchanged"


@dataclass(frozen=True)
class SearchLimits:
    batch_size: int = 32
    total_seconds: float = SEARCH_PREPARATION_SECONDS
    batch_seconds: float = 120
    build_seconds: float = 300
    memory_bytes: int = 1024 * 1024 * 1024
    max_text_bytes: int = 20_000_000
    max_chunks: int = 10_000

    def __post_init__(self) -> None:
        integers = (self.batch_size, self.memory_bytes, self.max_text_bytes, self.max_chunks)
        durations = (self.total_seconds, self.batch_seconds, self.build_seconds)
        if (
            any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in integers)
            or self.batch_size > 32
            or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in durations)
        ):
            raise ValueError("invalid_search_limits")


def effective_settings(
    profile: ModelProfile,
    limits: SearchLimits,
    overlap: int | None,
    representation: str = CANONICAL,
    *,
    budget_policy: str | None = None,
    openblas_threads: int | None = None,
) -> dict[str, JSONValue]:
    selected_overlap = profile.overlap_tokens if overlap is None else overlap
    if not 0 <= selected_overlap < profile.chunk_tokens:
        raise ValueError("invalid_chunk_overlap")
    rules = settings_for(representation)
    if rules and limits.batch_size != 1:
        raise ValueError("derived_search_requires_singleton")
    if budget_policy not in (None, BATCH_BUDGET_POLICY):
        raise ValueError("unsupported_search_budget_policy")
    settings: dict[str, JSONValue] = {
        **rules,
        **({"budget_policy": budget_policy} if budget_policy is not None else {}),
        "chunker_version": CHUNKER_VERSION,
        "batch_size": limits.batch_size,
        "separator": profile.separator,
        "chunk_tokens": profile.chunk_tokens,
        "overlap_tokens": selected_overlap,
        "input_tokens": profile.input_tokens,
        "threads": 2,
        "cpu_memory_arena": profile.id != "multilingual-e5",
        "idle_worker_policy": "stop-spinning-after-run-v1",
        "tokenizer_parallelism": False,
        "onnxruntime": version("onnxruntime"),
        "fastembed": version("fastembed"),
        "tokenizers": version("tokenizers"),
        "total_seconds": limits.total_seconds,
        "batch_seconds": limits.batch_seconds,
        "build_seconds": limits.build_seconds,
        "memory_bytes": limits.memory_bytes,
        "max_text_bytes": limits.max_text_bytes,
        "max_chunks": limits.max_chunks,
        "dimensions": profile.dimensions,
        "vector_format": profile.vector_format,
    }
    if openblas_threads is not None:
        openblas_options({"openblas_threads": openblas_threads})
        settings["openblas_threads"] = 1
    return settings


def create_search_job(
    store: PersistenceStore,
    source_id: str,
    extraction_id: str,
    profile_id: str,
    *,
    limits: SearchLimits | None = None,
    overlap: int | None = None,
    representation: str = CANONICAL,
) -> str:
    limits = limits or SearchLimits()
    profile = profiles()[profile_id]
    report = store.load_extraction_report(extraction_id)
    if report.extraction.source_id != source_id:
        raise ValueError("search_source_mismatch")
    eligible_passages(report)
    review = report.review.generation if report.review else 0
    with store._transaction() as connection:
        return enqueue_search_job(
            connection,
            source_id,
            report.extraction.source_version_id,
            extraction_id,
            profile,
            review,
            limits=limits,
            overlap=overlap,
            representation=representation,
        )


def enqueue_search_job(
    connection: sqlite3.Connection,
    source_id: str,
    source_version_id: str,
    extraction_id: str,
    profile: ModelProfile,
    review: int,
    *,
    limits: SearchLimits,
    overlap: int | None = None,
    work_suffix: str | None = None,
    representation: str = CANONICAL,
) -> str:
    """Job, exact inputs and queued status commit together with the caller's handoff."""
    settings = effective_settings(
        profile,
        limits,
        overlap,
        representation,
        budget_policy=BATCH_BUDGET_POLICY,
        openblas_threads=1,
    )
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    key = f"pdf-search:{extraction_id}:{review}:{profile.identity}:{fingerprint}"
    if work_suffix:
        key += ":" + work_suffix
    timestamp = utc_timestamp()
    connection.execute(
        """INSERT INTO jobs(id,logical_work_key,kind,source_id,source_version_id,
           extraction_version_id,index_version_id,created_at,updated_at)
           VALUES(?,?,'pdf-search',?,?,?,NULL,?,?) ON CONFLICT(logical_work_key) DO NOTHING""",
        (new_id(), key, source_id, source_version_id, extraction_id, timestamp, timestamp),
    )
    job = connection.execute("SELECT * FROM jobs WHERE logical_work_key=?", (key,)).fetchone()
    if job is None or (
        job["kind"],
        job["source_id"],
        job["source_version_id"],
        job["extraction_version_id"],
    ) != (
        "pdf-search",
        source_id,
        source_version_id,
        extraction_id,
    ):
        raise ConflictingWriteError("search work identity already has different targets")
    connection.execute(
        """INSERT OR IGNORE INTO search_job_inputs(job_id,profile_id,settings_json,
           review_generation) VALUES(?,?,?,?)""",
        (job["id"], profile.id, json.dumps(settings), review),
    )
    saved = connection.execute(
        "SELECT * FROM search_job_inputs WHERE job_id=?", (job["id"],)
    ).fetchone()
    if (
        saved["profile_id"] != profile.id
        or saved["review_generation"] != review
        or json.loads(saved["settings_json"]) != settings
    ):
        raise ConflictingWriteError("search work identity already has different inputs")
    connection.execute(
        "INSERT INTO job_status(job_id,status,updated_at) VALUES(?,'queued',?)"
        " ON CONFLICT(job_id) DO NOTHING",
        (job["id"], timestamp),
    )
    return job["id"]


def _guard(context: StepContext) -> AttemptGuard:
    if context.guard is None:
        raise ValueError("search_requires_attempt_guard")
    return context.guard


def _inputs(store: PersistenceStore, job_id: str) -> tuple[ModelProfile, dict, int]:
    with store.database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM search_job_inputs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row is None:
            raise ValueError("search_inputs_missing")
        settings = json.loads(row["settings_json"])
        if "budget_policy" in settings and settings["budget_policy"] != BATCH_BUDGET_POLICY:
            raise ValueError("unsupported_search_budget_policy")
        openblas_options(settings)
        return (
            profiles()[row["profile_id"]],
            settings,
            row["review_generation"],
        )


def _currency(store: PersistenceStore, context: StepContext) -> None:
    if context.should_stop():
        if context.is_cancelled():
            raise JobStepCancelled("search preparation stopped")
        raise TimeoutError("search preparation stopped at deadline or shutdown")
    # Passages/pages are immutable inputs. Recheck the small mutable decision
    # records instead of decoding the entire extraction twice for every batch.
    with store.database.connect() as connection:
        row = connection.execute(
            """SELECT i.review_generation,o.outcome,r.state,r.generation
               FROM jobs j JOIN search_job_inputs i ON i.job_id=j.id
               JOIN extraction_outcomes o ON o.extraction_version_id=j.extraction_version_id
               LEFT JOIN extraction_reviews r ON r.extraction_version_id=j.extraction_version_id
               WHERE j.id=?""",
            (context.job_id,),
        ).fetchone()
    if row is None or (row["generation"] or 0) != row["review_generation"]:
        raise PermanentStepFailure("extraction review changed", code="search_review_changed")
    if row["outcome"] != "complete" and not (
        row["outcome"] == "partial" and row["state"] == "confirmed"
    ):
        raise PermanentStepFailure("extraction is not confirmed", code="search_review_changed")
    with store.database.connect() as connection:
        handoff = connection.execute(
            "SELECT * FROM search_handoffs WHERE job_id=?", (context.job_id,)
        ).fetchone()
        if handoff is not None and handoff["purpose"] == "model-switch":
            from core.search.switching import switch_current

            if handoff["state"] == "cancelled" or not switch_current(connection, handoff):
                raise PermanentStepFailure(
                    "model selection changed", code="model_selection_changed"
                )


def _reserve_budget(
    connection: sqlite3.Connection,
    context: StepContext,
    limit: float,
    maximum: float,
    table: str,
    column: str,
    identity: str,
) -> tuple[float, float]:
    _verify_attempt_current(connection, context.job_id, context.step_key, _guard(context))
    spent = connection.execute(
        f"SELECT spent_seconds FROM {table} WHERE {column}=?", (identity,)
    ).fetchone()[0]
    reserved = min(maximum, limit - spent)
    if reserved <= 0:
        raise PermanentStepFailure("search preparation budget exhausted", code="search_time_limit")
    connection.execute(
        f"UPDATE {table} SET spent_seconds=spent_seconds+? WHERE {column}=?",
        (reserved, identity),
    )
    return spent, reserved


def _settle_budget(
    store: PersistenceStore,
    context: StepContext,
    table: str,
    column: str,
    identity: str,
    start: float,
    reserved: float,
) -> float:
    elapsed = time.monotonic() - start
    try:
        with store._transaction() as connection:
            _verify_attempt_current(connection, context.job_id, context.step_key, _guard(context))
            elapsed = time.monotonic() - start
            connection.execute(
                f"UPDATE {table} SET spent_seconds=spent_seconds+? WHERE {column}=?",
                (elapsed - reserved, identity),
            )
    except ConflictingWriteError:
        # An invalidated attempt cannot refund;its full committed lease remains.
        elapsed = max(elapsed, reserved)
    if elapsed > reserved:
        raise PermanentStepFailure("search preparation budget exhausted", code="search_time_limit")
    return elapsed


@contextmanager
def _budget(
    store: PersistenceStore,
    context: StepContext,
    limit: float,
    maximum: float,
    index_id: str | None = None,
    *,
    started: float | None = None,
):
    """Charge before work; a crash retains the entire unfinished reservation.

    Normal completion/failure refunds unused time through the same current guard.
    A cancelled/stale attempt cannot refund another attempt's consumed budget.
    """
    start = time.monotonic() if started is None else started
    table, column, identity = (
        ("search_generations", "index_version_id", index_id)
        if index_id
        else ("search_job_inputs", "job_id", context.job_id)
    )
    with store._transaction() as connection:
        _, reserved = _reserve_budget(connection, context, limit, maximum, table, column, identity)
    try:
        yield reserved
    finally:
        _settle_budget(store, context, table, column, identity, start, reserved)


@dataclass
class _MetadataLease:
    index_id: str | None
    spent_seconds: float
    deadline: float


@contextmanager
def _metadata_budget(
    store: PersistenceStore, context: StepContext, started: float, *, run_seconds: float = 0.0
) -> Iterator[_MetadataLease]:
    """Atomically resolve the ledger/reserve BEFORE fallible preparation reads.

    The admission transaction reads only immutable inputs and the exact generation
    identity. No review/extraction/model work is admitted if that transaction fails.
    A DB unavailable before the first durable write cannot meter its own outage;
    normal runner attempt bounds still apply. After admission,a crash or failed
    settlement retains the full reservation,not an inferred zero-cost prefix.
    """
    with store._transaction() as connection:
        _verify_attempt_current(connection, context.job_id, context.step_key, _guard(context))
        row = connection.execute(
            "SELECT i.*,j.source_id,j.source_version_id,j.extraction_version_id"
            " FROM search_job_inputs i JOIN jobs j ON j.id=i.job_id WHERE i.job_id=?",
            (context.job_id,),
        ).fetchone()
        if row is None:
            raise ValueError("search_inputs_missing")
        settings = json.loads(row["settings_json"])
        profile = profiles()[row["profile_id"]]
        job = SimpleNamespace(
            **{key: row[key] for key in ("source_id", "source_version_id", "extraction_version_id")}
        )
        generation = connection.execute(
            "SELECT index_version_id FROM search_generations WHERE identity_sha256=?",
            (generation_identity(job, profile, settings),),
        ).fetchone()
        index = generation[0] if generation else None
        table, column, identity = (
            ("search_generations", "index_version_id", index)
            if index
            else ("search_job_inputs", "job_id", context.job_id)
        )
        if index and row["spent_seconds"]:
            connection.execute(
                "UPDATE search_generations SET spent_seconds=spent_seconds+?"
                " WHERE index_version_id=?",
                (row["spent_seconds"], index),
            )
            connection.execute(
                "UPDATE search_job_inputs SET spent_seconds=0 WHERE job_id=?",
                (context.job_id,),
            )
        maximum = (
            settings["batch_seconds"]
            if context.step_key == "embed-search"
            else settings["build_seconds"] - run_seconds
        )
        spent, reserved = _reserve_budget(
            connection, context, settings["total_seconds"], maximum, table, column, identity
        )
    lease = _MetadataLease(index, spent, started + reserved)
    try:
        yield lease
    finally:
        lease.spent_seconds += _settle_budget(
            store, context, table, column, identity, started, reserved
        )


def build_search_steps(
    store: PersistenceStore, models_directory: Path
) -> tuple[StepDefinition, ...]:
    storage = SearchStorage(store)

    def index_id(job_id: str) -> str:
        rows = store.load_job_steps(job_id)
        return next(row.output_index_version_id for row in rows if row.step_key == PLAN)  # type: ignore[return-value]

    def inputs(context: StepContext) -> JSONValue:
        profile, settings, review = _inputs(store, context.job_id)
        return {"manifest": profile.identity, "settings": settings, "review_generation": review}

    def plan(context: StepContext) -> StepOutcome:
        entered = time.monotonic() if context.started_at is None else context.started_at
        with _metadata_budget(store, context, entered) as lease:
            _currency(store, context)
            profile, settings, _ = _inputs(store, context.job_id)
            existing = storage.lookup_generation(context.job_id, profile, settings)
        if existing is not None:
            return StepOutcome(1, {"index_id": existing})
        started = time.monotonic()
        with _budget(
            store,
            context,
            settings["total_seconds"],
            max(0, lease.deadline - started),
            started=started,
        ):
            job = store.load_job_by_id(context.job_id)
            assert job.extraction_version_id is not None
            report = store.load_extraction_report(job.extraction_version_id)
            tokenizer = InputTokenizer(models_directory / profile.identity, profile)
            chunks = make_chunks(
                eligible_passages(report),
                replace(profile, overlap_tokens=settings["overlap_tokens"]),
                tokenizer.count,
                tokenizer.prepare,
                max_text_bytes=settings["max_text_bytes"],
                max_chunks=settings["max_chunks"],
            )
            _currency(store, context)
            identity = storage.create_generation(
                context.job_id, _guard(context), profile, chunks, settings
            )
        return StepOutcome(1, {"index_id": identity})

    def commit_plan(store_arg: PersistenceStore, commit: StepCommitContext) -> None:
        started = commit.outcome.completed_at
        if started is None:
            started = time.monotonic()
        assert isinstance(commit.outcome.checkpoint, dict)
        identity = str(commit.outcome.checkpoint["index_id"])
        context = StepContext(
            commit.job_id,
            commit.step_key,
            commit.guard.expected_attempt_number,
            lambda: False,
            lambda: False,
            commit.guard,
        )
        with _metadata_budget(
            store_arg, context, started, run_seconds=commit_run_seconds(commit)
        ) as lease:
            _, settings, _ = _inputs(store_arg, commit.job_id)
            if lease.index_id != identity:
                raise ConflictingWriteError("search checkpoint generation changed")
        started = time.monotonic()
        with _budget(
            store_arg,
            context,
            settings["total_seconds"],
            max(0, lease.deadline - started),
            identity,
            started=started,
        ) as allowance:
            deadline = started + allowance
            graph = store_arg.load_index(identity)
            if time.monotonic() >= deadline:
                raise PermanentStepFailure(
                    "search preparation budget exhausted", code="search_time_limit"
                )
            try:
                store_arg.commit_index_step(
                    commit.job_id,
                    commit.step_key,
                    graph.index,
                    graph.chunks,
                    graph.mappings,
                    1,
                    commit.outcome.checkpoint,
                    guard=commit.guard,
                    inputs_sha256=commit.inputs_sha256,
                    search_deadline=deadline,
                )
            except TimeoutError as error:
                raise PermanentStepFailure(
                    "search preparation budget exhausted", code="search_time_limit"
                ) from error

    def commit_checkpoint(store_arg: PersistenceStore, commit: StepCommitContext) -> None:
        started = commit.outcome.completed_at
        if started is None:
            started = time.monotonic()
        context = StepContext(
            commit.job_id,
            commit.step_key,
            commit.guard.expected_attempt_number,
            lambda: False,
            lambda: False,
            commit.guard,
        )
        with _metadata_budget(
            store_arg, context, started, run_seconds=commit_run_seconds(commit)
        ) as lease:
            _, settings, _ = _inputs(store_arg, commit.job_id)
            identity = index_id(commit.job_id)
            if lease.index_id != identity:
                raise ConflictingWriteError("search checkpoint generation changed")
        started = time.monotonic()
        with _budget(
            store_arg,
            context,
            settings["total_seconds"],
            max(0, lease.deadline - started),
            identity,
            started=started,
        ) as allowance:
            try:
                if commit.step_key == "publish-search":
                    publish_output(commit, identity, started + allowance)
                    return
                store_arg.commit_step(
                    commit.job_id,
                    commit.step_key,
                    commit.outcome.checkpoint_format_version,
                    commit.outcome.checkpoint,
                    guard=commit.guard,
                    inputs_sha256=commit.inputs_sha256,
                    search_deadline=started + allowance,
                )
            except (TimeoutError, ConflictingWriteError) as error:
                if time.monotonic() < started + allowance:
                    raise
                raise PermanentStepFailure(
                    "search preparation budget exhausted", code="search_time_limit"
                ) from error

    def commit_run_seconds(commit: StepCommitContext) -> float:
        # A build's run and commit are ONE attempt,not two independent300s budgets.
        elapsed = 0.0
        if (
            commit.step_key != "embed-search"
            and commit.outcome.started_at is not None
            and commit.outcome.completed_at is not None
        ):
            elapsed = commit.outcome.completed_at - commit.outcome.started_at
        return elapsed

    def embed(context: StepContext) -> StepOutcome:
        # Reuse is attempt/thread-local; every transaction and borrow still ends
        # before inference. Closing/checkpointing remains inside charged time.
        with ExitStack() as connections:
            connections.enter_context(store.database.connection_scope())
            return embed_scoped(context, connections.close)

    def embed_scoped(context: StepContext, close_connections: Callable[[], None]) -> StepOutcome:
        entered = time.monotonic() if context.started_at is None else context.started_at
        with _metadata_budget(store, context, entered) as lease:
            _currency(store, context)
            profile, settings, _ = _inputs(store, context.job_id)
            identity = index_id(context.job_id)
            with store.database.connect() as connection:
                batches = connection.execute(
                    "SELECT * FROM search_batches WHERE index_version_id=? ORDER BY ordinal",
                    (identity,),
                ).fetchall()
            parent_heap_release = _release_planning_heap()
        step_started = time.monotonic()
        charged_before = lease.spent_seconds
        original_context = context
        source_deadline = step_started + max(0, settings["total_seconds"] - charged_before - 2)
        context = replace(
            context,
            should_stop=lambda: (
                original_context.should_stop() or time.monotonic() >= source_deadline
            ),
        )

        def check_currency() -> None:
            if context.is_cancelled():
                raise JobStepCancelled("embedding stopped")
            if time.monotonic() >= source_deadline:
                raise PermanentStepFailure(
                    "search preparation budget exhausted", code="search_time_limit"
                )
            _currency(store, context)

        session = None
        peak = 0
        load_seconds = 0.0
        batch_times = []

        def reserved_batch(batch: sqlite3.Row) -> None:
            """One durable lease covers validation through commit, never a DB lock.

            Crashes retain the full reservation even before attempt admission.
            Saved jobs without this policy continue through the legacy loop below.
            """
            nonlocal session, load_seconds, peak
            while True:
                attempts = None
                try:
                    started = time.monotonic()
                    with _budget(
                        store,
                        context,
                        settings["total_seconds"],
                        settings["batch_seconds"],
                        identity,
                        started=started,
                    ) as allowance:
                        deadline = min(started + allowance, source_deadline)

                        def remaining(deadline: float = deadline) -> float:
                            seconds = deadline - time.monotonic() - 2
                            if seconds <= 0:
                                raise PermanentStepFailure(
                                    "search preparation budget exhausted", code="search_time_limit"
                                )
                            return seconds

                        check_currency()
                        ids = json.loads(batch["chunk_ids_json"])
                        with store.database.connect() as connection:
                            texts, input_hash = storage.batch_inputs(connection, identity, ids)
                        remaining()
                        with store._transaction() as connection:
                            _verify_attempt_current(
                                connection, context.job_id, context.step_key, _guard(context)
                            )
                            attempts = connection.execute(
                                "SELECT attempts FROM search_batches WHERE index_version_id=?"
                                " AND ordinal=?",
                                (identity, batch["ordinal"]),
                            ).fetchone()[0]
                            if attempts >= 2:
                                raise PermanentStepFailure(
                                    "embedding batch attempts exhausted",
                                    code="embedding_attempt_limit",
                                )
                            connection.execute(
                                "UPDATE search_batches SET attempts=attempts+1"
                                " WHERE index_version_id=? AND ordinal=?",
                                (identity, batch["ordinal"]),
                            )
                        if session is None:
                            session = EmbeddingSession(
                                models_directory / profile.identity,
                                profile,
                                should_stop=context.should_stop,
                                load_timeout=remaining(),
                                memory_limit=settings["memory_bytes"],
                                memory_arena=settings["cpu_memory_arena"],
                                stop_spinning=settings.get("idle_worker_policy")
                                == "stop-spinning-after-run-v1",
                                **openblas_options(settings),
                            )
                            load_seconds += session.load_seconds
                        start = time.monotonic()
                        vectors = session.embed(texts, timeout=remaining())
                        check_currency()
                        storage.commit_batch(
                            context.job_id,
                            _guard(context),
                            identity,
                            batch["ordinal"],
                            ids,
                            vectors,
                            search_input_sha256=input_hash,
                            deadline=deadline,
                        )
                        batch_times.append(time.monotonic() - start)
                    peak = max(peak, session.peak_bytes)
                    return
                except EmbeddingStopped as error:
                    check_currency()
                    raise TimeoutError("embedding stopped at deadline or shutdown") from error
                except EmbeddingFailure as error:
                    raise PermanentStepFailure(error.code, code=error.code) from error
                except TimeoutError:
                    # Validation/admission failures do not consume a model retry here.
                    if attempts is None:
                        raise
                    if session is not None:
                        peak = max(peak, session.peak_bytes)
                        session.close()
                        session = None
                    if attempts >= 1:
                        raise PermanentStepFailure(
                            "embedding batch deadline exhausted", code="embedding_timeout"
                        ) from None
                    with _budget(
                        store,
                        context,
                        settings["total_seconds"],
                        settings["batch_seconds"],
                        identity,
                    ):
                        until = time.monotonic() + random.uniform(1.5, 2.5)
                        while time.monotonic() < until:
                            check_currency()
                            time.sleep(0.05)

        try:
            for batch in batches:
                if batch["complete"]:
                    continue
                if settings.get("budget_policy") == BATCH_BUDGET_POLICY:
                    reserved_batch(batch)
                    continue
                with _budget(
                    store, context, settings["total_seconds"], settings["batch_seconds"], identity
                ):
                    check_currency()
                    ids = json.loads(batch["chunk_ids_json"])
                    with store.database.connect() as connection:
                        texts, search_input_sha256 = storage.batch_inputs(connection, identity, ids)
                while True:
                    with _budget(
                        store,
                        context,
                        settings["total_seconds"],
                        settings["batch_seconds"],
                        identity,
                    ):
                        with store._transaction() as connection:
                            _verify_attempt_current(
                                connection, context.job_id, context.step_key, _guard(context)
                            )
                            attempts = connection.execute(
                                "SELECT attempts FROM search_batches WHERE index_version_id=?"
                                " AND ordinal=?",
                                (identity, batch["ordinal"]),
                            ).fetchone()[0]
                            if attempts >= 2:
                                raise PermanentStepFailure(
                                    "embedding batch attempts exhausted",
                                    code="embedding_attempt_limit",
                                )
                            connection.execute(
                                "UPDATE search_batches SET attempts=attempts+1"
                                " WHERE index_version_id=?"
                                " AND ordinal=?",
                                (identity, batch["ordinal"]),
                            )
                    try:
                        if session is None:
                            with _budget(
                                store,
                                context,
                                settings["total_seconds"],
                                settings["batch_seconds"],
                                identity,
                            ) as allowance:
                                session = EmbeddingSession(
                                    models_directory / profile.identity,
                                    profile,
                                    should_stop=context.should_stop,
                                    load_timeout=max(0.01, allowance - 2),
                                    memory_limit=settings["memory_bytes"],
                                    memory_arena=settings["cpu_memory_arena"],
                                    stop_spinning=settings.get("idle_worker_policy")
                                    == "stop-spinning-after-run-v1",
                                    **openblas_options(settings),
                                )
                                load_seconds += session.load_seconds
                        with _budget(
                            store,
                            context,
                            settings["total_seconds"],
                            settings["batch_seconds"],
                            identity,
                        ) as allowance:
                            start = time.monotonic()
                            vectors = session.embed(texts, timeout=max(0.01, allowance - 2))
                            check_currency()
                            storage.commit_batch(
                                context.job_id,
                                _guard(context),
                                identity,
                                batch["ordinal"],
                                ids,
                                vectors,
                                search_input_sha256=search_input_sha256,
                            )
                            batch_times.append(time.monotonic() - start)
                        peak = max(peak, session.peak_bytes)
                        break
                    except EmbeddingStopped as error:
                        check_currency()
                        raise TimeoutError("embedding stopped at deadline or shutdown") from error
                    except EmbeddingFailure as error:
                        raise PermanentStepFailure(error.code, code=error.code) from error
                    except TimeoutError:
                        if session is not None:
                            peak = max(peak, session.peak_bytes)
                            session.close()
                            session = None
                        if attempts >= 1:
                            raise PermanentStepFailure(
                                "embedding batch deadline exhausted", code="embedding_timeout"
                            ) from None
                        with _budget(
                            store,
                            context,
                            settings["total_seconds"],
                            settings["batch_seconds"],
                            identity,
                        ):
                            until = time.monotonic() + random.uniform(1.5, 2.5)
                            while time.monotonic() < until:
                                check_currency()
                                time.sleep(0.05)
        finally:
            try:
                if session is not None:
                    peak = max(peak, session.peak_bytes)
                    session.close()
            finally:
                close_connections()
            # Include work between leases (decision checks, input reads, IPC,
            # checkpoint bookkeeping, backoff, and stop/reap), not only inference.
            exhausted = False
            try:
                with store._transaction() as connection:
                    _verify_attempt_current(
                        connection, context.job_id, context.step_key, _guard(context)
                    )
                    charged = connection.execute(
                        "SELECT spent_seconds FROM search_generations WHERE index_version_id=?",
                        (identity,),
                    ).fetchone()[0]
                    overhead = max(0, time.monotonic() - step_started - (charged - charged_before))
                    connection.execute(
                        "UPDATE search_generations SET spent_seconds=spent_seconds+?"
                        " WHERE index_version_id=?",
                        (overhead, identity),
                    )
                    exhausted = charged + overhead >= settings["total_seconds"]
            except ConflictingWriteError:
                pass  # Invalidated work retains its pre-committed reservations.
            if exhausted:
                raise PermanentStepFailure(
                    "search preparation budget exhausted", code="search_time_limit"
                )
        return StepOutcome(
            1,
            {
                "index_id": identity,
                "child_peak_bytes": peak,
                "model_load_seconds": load_seconds,
                "parent_heap_release": parent_heap_release,
                "batch_seconds": batch_times,
            },
        )

    def build(context: StepContext) -> StepOutcome:
        entered = time.monotonic() if context.started_at is None else context.started_at
        with _metadata_budget(store, context, entered) as lease:
            _currency(store, context)
            _, settings, _ = _inputs(store, context.job_id)
            identity = index_id(context.job_id)
        started = time.monotonic()
        with _budget(
            store,
            context,
            settings["total_seconds"],
            max(0, lease.deadline - started),
            identity,
            started=started,
        ):
            storage.finish_components(context.job_id, _guard(context), identity)
        return StepOutcome(1, {"index_id": identity})

    def publish(context: StepContext) -> StepOutcome:
        started = time.monotonic() if context.started_at is None else context.started_at
        with _metadata_budget(store, context, started):
            _currency(store, context)
            identity = index_id(context.job_id)
            _inputs(store, context.job_id)
            return StepOutcome(1, {"index_id": identity, "ready": True})

    def publish_output(commit: StepCommitContext, identity: str, deadline: float) -> None:
        # Publish and final checkpoint share ONE transaction and its final deadline
        # check. A slow checkpoint cannot leave a switched notebook with failed output.
        job = store.load_job_by_id(commit.job_id)
        assert job.source_version_id is not None
        checkpoint = (
            commit.outcome.checkpoint_format_version,
            commit.outcome.checkpoint,
            commit.inputs_sha256,
        )
        with store.database.connect() as connection:
            handoff = connection.execute(
                "SELECT purpose FROM search_handoffs WHERE job_id=?", (job.id,)
            ).fetchone()
        if handoff is not None and handoff[0] == "model-switch":
            from core.search.switching import publish_replacement

            publish_replacement(
                store, job.id, commit.guard, identity, deadline=deadline, checkpoint=checkpoint
            )
        else:
            store.publish_readiness(
                job.id,
                job.source_id,
                job.source_version_id,
                [identity],
                guard=commit.guard,
                search_deadline=deadline,
                search_checkpoint=checkpoint,
            )

    return (
        StepDefinition(PLAN, plan, "index_build", commit=commit_plan, inputs=inputs),
        StepDefinition(
            "embed-search", embed, "search_prepare", commit=commit_checkpoint, inputs=inputs
        ),
        StepDefinition(
            "build-search", build, "index_build", commit=commit_checkpoint, inputs=inputs
        ),
        StepDefinition(
            "publish-search", publish, "index_build", commit=commit_checkpoint, inputs=inputs
        ),
    )
