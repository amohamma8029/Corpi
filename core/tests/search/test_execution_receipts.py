"""Actual runtime belongs to an attempt, not rewritten historical settings."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_preparation_accounting as accounting
from test_batch_reservations import VECTOR, batch_state, embed_stage
from test_controls import queued
from test_embedding_supervisor import command

from core.jobs.runner import StepContext
from core.persistence.errors import ConflictingWriteError, IntegrityViolationError
from core.persistence.models import AttemptGuard
from core.search import preparation
from core.search import storage as storage_module
from core.search.controls import PreparationControl
from core.search.handoff import SearchCoordinator
from core.search.profiles import profiles
from core.search.provenance import runtime_versions
from core.search.storage import SearchStorage, purge_search
from core.search.supervisor import EmbeddingFailure, EmbeddingSession

planned = accounting.planned


def receipts(store):
    with store.database.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM search_batch_executions"
                " ORDER BY index_version_id,ordinal,batch_attempt_number"
            )
        ]


def admit(store, clock, job, context):
    batch = batch_state(store, job)
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_batches SET attempts=1 WHERE index_version_id=?",
            (batch["index_version_id"],),
        )
    assert context.guard is not None
    SearchStorage(store).record_batch_execution(
        job,
        context.guard,
        batch["index_version_id"],
        0,
        1,
        runtime_versions(),
        deadline=clock.monotonic() + 20,
    )
    return batch


def test_receipt_is_durable_before_text_and_metadata_does_not_hold_database_lock(
    store, monkeypatch, planned
):
    clock, job, context, step, _, session_type = embed_stage(store, planned, monkeypatch)
    original = session_type.embed

    def checked(self, texts, **options):
        saved = receipts(store)
        assert len(saved) == 1 and saved[0]["job_id"] == job
        assert json.loads(saved[0]["runtime_json"]) == runtime_versions()
        # A separate write transaction must be available before the IPC send.
        with store._transaction() as connection:
            assert connection.in_transaction
        return original(self, texts, **options)

    monkeypatch.setattr(session_type, "embed", checked)
    before = preparation._inputs(store, job)[1]
    step.run(context)
    assert preparation._inputs(store, job)[1] == before
    assert batch_state(store, job)["attempts"] == 1
    assert batch_state(store, job)["complete"] == 1


@pytest.mark.parametrize("legacy_policy", [False, True])
def test_resumed_old_job_records_new_worker_without_relabelling_or_resetting(
    store, monkeypatch, legacy_policy
):
    calculate = preparation.effective_settings

    def old_settings(*args, **kwargs):
        if legacy_policy:
            kwargs.pop("budget_policy", None)
        return {**calculate(*args, **kwargs), "fastembed": "0.8.1"}

    with monkeypatch.context() as scoped:
        scoped.setattr(preparation, "effective_settings", old_settings)
        _, source, job = queued(store)
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_job_inputs SET spent_seconds=43.25 WHERE job_id=?", (job,)
        )
        old_input = connection.execute(
            "SELECT settings_json FROM search_job_inputs WHERE job_id=?", (job,)
        ).fetchone()[0]
        old_vectors = connection.execute("SELECT rowid,embedding FROM search_vectors").fetchall()
    control = PreparationControl(store)
    control.cancel(source, job)
    SearchCoordinator(store, Path("unused")).reconcile()
    resumed = control.resume(source, job, "runtime-upgrade")
    with store.database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM search_job_inputs WHERE job_id=?", (resumed,)
        ).fetchone()
        assert row["settings_json"] == old_input and row["spent_seconds"] == 43.25
    store.claim_job(resumed, "accounting-tests")
    attempt = store.start_step_attempt(resumed, preparation.PLAN)
    guard = AttemptGuard("accounting-tests", preparation.PLAN, attempt.attempt_number)
    context = StepContext(
        resumed, preparation.PLAN, attempt.attempt_number, lambda: False, lambda: False, guard
    )
    clock = accounting.Clock()
    monkeypatch.setattr(
        preparation,
        "InputTokenizer",
        lambda *args: SimpleNamespace(count=lambda text: 1, prepare=lambda text: text),
    )
    monkeypatch.setattr(preparation, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr("core.persistence.store.time", SimpleNamespace(monotonic=clock.monotonic))
    stages = preparation.build_search_steps(store, Path("unused"))
    _, _, context, step, _, _ = embed_stage(store, (clock, resumed, context, stages), monkeypatch)
    step.run(context)
    saved = receipts(store)
    assert len(saved) == 1 and saved[0]["job_id"] == resumed
    assert json.loads(saved[0]["runtime_json"])["fastembed"] == "0.9.1"
    batch = batch_state(store, resumed)
    with store.database.connect() as connection:
        generation = connection.execute(
            "SELECT * FROM search_generations WHERE index_version_id=?",
            (batch["index_version_id"],),
        ).fetchone()
        assert json.loads(generation["settings_json"])["fastembed"] == "0.8.1"
        assert generation["spent_seconds"] >= 43.25
        assert (
            connection.execute(
                "SELECT settings_json FROM search_job_inputs WHERE job_id=?", (job,)
            ).fetchone()[0]
            == old_input
        )
        assert [
            tuple(row)
            for row in connection.execute(
                "SELECT rowid,embedding FROM search_vectors ORDER BY rowid LIMIT ?",
                (len(old_vectors),),
            )
        ] == [tuple(row) for row in old_vectors]
    assert store.load_job_status(job).status == "cancelled"


def test_receipt_write_failure_prevents_inference_and_preserves_consumed_attempt(
    store, monkeypatch, planned
):
    _, job, context, step, sessions, session_type = embed_stage(store, planned, monkeypatch)
    monkeypatch.setattr(session_type, "embed", lambda *a, **k: pytest.fail("unrecorded inference"))

    def fail(*args, **kwargs):
        raise ConflictingWriteError("receipt unavailable")

    monkeypatch.setattr(SearchStorage, "record_batch_execution", fail)
    with pytest.raises(ConflictingWriteError, match="receipt unavailable"):
        step.run(context)
    assert receipts(store) == [] and batch_state(store, job)["attempts"] == 1
    assert all(session.closed for session in sessions)


@pytest.mark.parametrize("legacy_policy", [False, True])
def test_partial_old_generation_reuses_old_vectors_and_records_only_new_batch(
    store, monkeypatch, legacy_policy
):
    import hashlib

    from core.search.chunking import Chunk, PassageSlice

    calculate = preparation.effective_settings

    def old_settings(*args, **kwargs):
        if legacy_policy:
            kwargs.pop("budget_policy", None)
        return {**calculate(*args, **kwargs), "fastembed": "0.8.1", "batch_size": 1}

    with monkeypatch.context() as scoped:
        scoped.setattr(preparation, "effective_settings", old_settings)
        _, source, old_job = queued(store)
    clock = accounting.Clock()
    monkeypatch.setattr(preparation, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr("core.persistence.store.time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(
        preparation,
        "InputTokenizer",
        lambda *args: SimpleNamespace(count=lambda text: 1, prepare=lambda text: text),
    )

    def two_chunks(passages, *args, **kwargs):
        passage = passages[0]
        assert passage.text == "rainfall"
        return (
            Chunk("rain", (PassageSlice(passage, 0, 4),)),
            Chunk("fall", (PassageSlice(passage, 4, 8),)),
        )

    monkeypatch.setattr(preparation, "make_chunks", two_chunks)

    def plan_context(job):
        store.claim_job(job, "accounting-tests")
        attempt = store.start_step_attempt(job, preparation.PLAN)
        guard = AttemptGuard("accounting-tests", preparation.PLAN, attempt.attempt_number)
        return StepContext(
            job, preparation.PLAN, attempt.attempt_number, lambda: False, lambda: False, guard
        )

    stages = preparation.build_search_steps(store, Path("unused"))
    embed_stage(store, (clock, old_job, plan_context(old_job), stages), monkeypatch)
    old_batch = batch_state(store, old_job)
    identity = old_batch["index_version_id"]
    graph = store.load_index(identity)
    # Seed a pre-upgrade completed batch as data. No old execution receipt exists.
    with store._transaction() as connection:
        row_id = connection.execute(
            "SELECT row_id FROM search_rows WHERE chunk_id=?", (graph.chunks[0].id,)
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO search_vectors(rowid,embedding) VALUES(?,?)", (row_id, VECTOR)
        )
        connection.execute(
            "UPDATE search_batches SET attempts=1,complete=1,vectors_sha256=?"
            " WHERE index_version_id=? AND ordinal=0",
            (hashlib.sha256(VECTOR).hexdigest(), identity),
        )
        connection.execute(
            "UPDATE search_generations SET spent_seconds=43.25 WHERE index_version_id=?",
            (identity,),
        )
        settings_before = connection.execute(
            "SELECT settings_json FROM search_generations WHERE index_version_id=?", (identity,)
        ).fetchone()[0]
    control = PreparationControl(store)
    control.cancel(source, old_job)
    SearchCoordinator(store, Path("unused")).reconcile()
    resumed = control.resume(source, old_job, "partial-upgrade")
    stages = preparation.build_search_steps(store, Path("unused"))
    _, _, context, step, _, session_type = embed_stage(
        store, (clock, resumed, plan_context(resumed), stages), monkeypatch
    )
    called = []
    embed = session_type.embed

    def only_pending(self, texts, **options):
        called.append(texts)
        assert texts == ["fall"]
        return embed(self, texts, **options)

    monkeypatch.setattr(session_type, "embed", only_pending)
    step.run(context)
    assert called == [["fall"]]
    saved = receipts(store)
    assert len(saved) == 1 and saved[0]["ordinal"] == 1 and saved[0]["job_id"] == resumed
    assert json.loads(saved[0]["runtime_json"])["fastembed"] == "0.9.1"
    assert store.load_index(identity) == graph
    with store.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT embedding FROM search_vectors WHERE rowid=?", (row_id,)
            ).fetchone()[0]
            == VECTOR
        )
        generation = connection.execute(
            "SELECT * FROM search_generations WHERE index_version_id=?", (identity,)
        ).fetchone()
        assert (
            generation["settings_json"] == settings_before and generation["spent_seconds"] >= 43.25
        )
        assert [
            (row["attempts"], row["complete"])
            for row in connection.execute(
                "SELECT * FROM search_batches WHERE index_version_id=? ORDER BY ordinal",
                (identity,),
            )
        ] == [(1, 1), (1, 1)]


def test_repeated_receipt_admission_is_idempotent(store, monkeypatch, planned):
    clock, job, context, _, _, _ = embed_stage(store, planned, monkeypatch)
    batch = admit(store, clock, job, context)
    before = receipts(store)
    clock.advance(1)
    assert context.guard is not None
    SearchStorage(store).record_batch_execution(
        job,
        context.guard,
        batch["index_version_id"],
        0,
        1,
        runtime_versions(),
        deadline=clock.monotonic() + 20,
    )
    assert receipts(store) == before and batch_state(store, job)["attempts"] == 1


@pytest.mark.parametrize("execution_attempt", [None, 1, 2])
def test_commit_requires_exact_current_attempt_receipt(
    store, monkeypatch, planned, execution_attempt
):
    clock, job, context, _, _, _ = embed_stage(store, planned, monkeypatch)
    if execution_attempt == 1:
        batch = batch_state(store, job)
        with store._transaction() as connection:
            connection.execute(
                "UPDATE search_batches SET attempts=1 WHERE index_version_id=?",
                (batch["index_version_id"],),
            )
    else:
        batch = admit(store, clock, job, context)
    assert context.guard is not None
    with pytest.raises(ConflictingWriteError, match="receipt missing or stale"):
        SearchStorage(store).commit_batch(
            job,
            context.guard,
            batch["index_version_id"],
            0,
            json.loads(batch["chunk_ids_json"]),
            [VECTOR],
            execution_attempt=execution_attempt,
        )
    assert batch_state(store, job)["complete"] == 0


def test_expired_receipt_rolls_back_and_stale_owner_cannot_record(store, monkeypatch, planned):
    clock, job, context, _, _, _ = embed_stage(store, planned, monkeypatch)
    batch = batch_state(store, job)
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_batches SET attempts=1 WHERE index_version_id=?",
            (batch["index_version_id"],),
        )
    assert context.guard is not None
    times = iter([clock.monotonic(), clock.monotonic() + 30])
    monkeypatch.setattr(storage_module, "time", SimpleNamespace(monotonic=lambda: next(times)))
    with pytest.raises(TimeoutError, match="deadline expired"):
        SearchStorage(store).record_batch_execution(
            job,
            context.guard,
            batch["index_version_id"],
            0,
            1,
            runtime_versions(),
            deadline=clock.monotonic() + 20,
        )
    assert receipts(store) == []
    monkeypatch.setattr(storage_module, "time", SimpleNamespace(monotonic=clock.monotonic))
    wrong = replace(context.guard, owner_instance_id="another-owner")
    with pytest.raises(ConflictingWriteError):
        SearchStorage(store).record_batch_execution(
            job,
            wrong,
            batch["index_version_id"],
            0,
            1,
            runtime_versions(),
            deadline=clock.monotonic() + 20,
        )
    assert receipts(store) == []


@pytest.mark.parametrize("version_only", [False, True])
def test_receipt_is_immutable_and_removed_by_existing_source_cleanup(
    store, monkeypatch, planned, version_only
):
    from test_storage import seed

    clock, job, context, step, _, _ = embed_stage(store, planned, monkeypatch)
    step.run(context)
    assert len(receipts(store)) == 1
    unrelated = seed(store, "other rainfall", [1.0] + [0.0] * 383)
    store.finish_job(unrelated[4], "completed")
    prior = store.load_job_by_id(unrelated[4])
    other = preparation.create_search_job(
        store, prior.source_id, prior.extraction_version_id, "bge-small-en"
    )
    store.claim_job(other, "accounting-tests")
    attempt = store.start_step_attempt(other, preparation.PLAN)
    guard = AttemptGuard("accounting-tests", preparation.PLAN, attempt.attempt_number)
    other_context = StepContext(
        other, preparation.PLAN, attempt.attempt_number, lambda: False, lambda: False, guard
    )
    stages = preparation.build_search_steps(store, Path("unused"))
    _, _, other_context, other_step, _, _ = embed_stage(
        store, (clock, other, other_context, stages), monkeypatch
    )
    other_step.run(other_context)
    unrelated_receipts = [row for row in receipts(store) if row["job_id"] == other]
    assert len(unrelated_receipts) == 1
    with pytest.raises(IntegrityViolationError):
        with store._transaction() as connection:
            connection.execute("UPDATE search_batch_executions SET runtime_json='{}'")
    item = store.load_job_by_id(job)
    with store._transaction() as connection:
        purge_search(
            connection,
            source_id=item.source_id,
            version_id=item.source_version_id if version_only else None,
        )
    assert receipts(store) == unrelated_receipts
    assert store.load_source(unrelated[1]).id == unrelated[1]
    with store.database.connect() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("bad", ["missing", "version", "extra", "object"])
def test_missing_or_mismatched_child_runtime_is_rejected_before_sending_text(tmp_path, bad):
    runtime = runtime_versions()
    if bad == "version":
        runtime["fastembed"] = "0.8.1"
    elif bad == "extra":
        runtime["private"] = "must-not-be-disclosed"
    value = (
        None
        if bad == "missing"
        else {"native": ["must-not-be-disclosed"]}
        if bad == "object"
        else runtime
    )
    profile = profiles()["minilm-en"]
    ready: dict[str, object] = {"type": "ready", "manifest": profile.identity, "request": 0}
    if value is not None:
        ready["runtime"] = value
    marker = tmp_path / "sent-text"
    body = (
        f"print(json.dumps({ready!r}),flush=True); line=sys.stdin.readline(); "
        f"line and open({str(marker)!r},'w').write('sent')"
    )
    with pytest.raises(EmbeddingFailure, match="^runtime_mismatch$") as error:
        EmbeddingSession(tmp_path, profile, command=command(profile, body), load_timeout=10)
    assert str(error.value) == "runtime_mismatch" and not marker.exists()


def test_populated_old_search_snapshot_restores_exact_vectors_settings_attempts_and_charges(
    tmp_path,
):
    import shutil

    from test_storage import seed

    from core.persistence.database import Database, DatabaseConfig
    from core.persistence.migrations import load_migrations
    from core.persistence.store import PersistenceStore

    old = Database(DatabaseConfig(tmp_path / "old"), migrations=load_migrations()[:14])
    old.initialize()
    store = PersistenceStore(old)
    _, _, _, index, _, _ = seed(
        store,
        "saved rainfall",
        [1.0] + [0.0] * 383,
        profile_id="minilm-en",
        runtime_settings={"fastembed": "0.8.1"},
    )
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_generations SET spent_seconds=43.25 WHERE index_version_id=?", (index,)
        )

    def saved(database):
        with database.connect() as connection:
            tables = (
                "jobs",
                "job_steps",
                "job_step_attempts",
                "index_versions",
                "retrieval_chunks",
                "chunk_passages",
                "search_generations",
                "search_batches",
                "search_components",
                "source_readiness",
                "source_readiness_indexes",
            )
            result = {
                table: [
                    tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")
                ]
                for table in tables
            }
            result["vectors"] = [
                tuple(row)
                for row in connection.execute(
                    "SELECT rowid,embedding FROM search_vectors ORDER BY rowid"
                )
            ]
            result["words"] = [
                tuple(row)
                for row in connection.execute("SELECT rowid,text FROM search_words ORDER BY rowid")
            ]
            return result

    before = saved(old)
    current = Database(old.config)
    current.initialize()
    assert saved(current) == before
    with current.connect() as connection:
        assert connection.execute("SELECT count(*) FROM search_batch_executions").fetchone()[0] == 0
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    shutil.copyfile(
        old.config.snapshot_dir / "snapshot-r0014.sqlite3", restored_root / "corpi.sqlite3"
    )
    restored = Database(DatabaseConfig(restored_root), migrations=load_migrations()[:14])
    restored.initialize()
    assert saved(restored) == before
