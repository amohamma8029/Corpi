"""Deterministic wall-time faults around real native search persistence,not inference speed."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_storage import seed

from core.jobs.runner import PermanentStepFailure, StepCommitContext, StepContext
from core.persistence.models import AttemptGuard
from core.persistence.store import PersistenceStore
from core.search import preparation


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def planned(store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, request):
    _, source, _, old_index, prior, _ = seed(
        store, "rainfall measurements", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    store.finish_job(prior, "completed")
    extraction = store.load_index(old_index).index.extraction_version_id

    class Tokenizer:
        def __init__(self, *_):
            pass

        def count(self, _):
            return 1

        def prepare(self, text):
            return text

    clock = Clock()
    monkeypatch.setattr(preparation, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr("core.persistence.store.time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr("core.search.storage.time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(preparation, "InputTokenizer", Tokenizer)
    job = preparation.create_search_job(
        store,
        source,
        extraction,
        "minilm-en",
        limits=getattr(request, "param", preparation.SearchLimits(total_seconds=20)),
    )
    store.claim_job(job, "accounting-tests")
    attempt = store.start_step_attempt(job, preparation.PLAN)
    guard = AttemptGuard("accounting-tests", preparation.PLAN, attempt.attempt_number)
    context = StepContext(
        job, preparation.PLAN, attempt.attempt_number, lambda: False, lambda: False, guard
    )
    steps = preparation.build_search_steps(store, Path("unused"))
    return clock, job, context, steps


def test_plan_checkpoint_read_is_charged_and_cannot_complete_after_source_deadline(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, planned
):
    clock, job, context, steps = planned
    outcome = steps[0].run(context)
    load = store.load_index

    def slow_graph(identity):
        clock.advance(21)
        return load(identity)

    monkeypatch.setattr(store, "load_index", slow_graph)
    with pytest.raises(PermanentStepFailure) as failure:
        steps[0].commit(
            store, StepCommitContext(job, preparation.PLAN, context.guard, "a" * 64, outcome)
        )
    assert failure.value.code == "search_time_limit"
    assert store.load_job_steps(job) == ()
    with store.database.connect() as connection:
        spent = connection.execute(
            "SELECT spent_seconds FROM search_generations WHERE index_version_id=?",
            (outcome.checkpoint["index_id"],),
        ).fetchone()[0]
    assert spent >= 21


def test_currency_and_input_reads_are_inside_the_planning_charge(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, planned
):
    clock, job, context, steps = planned
    currency = preparation._currency

    def slow_currency(*args):
        clock.advance(3)
        currency(*args)

    monkeypatch.setattr(preparation, "_currency", slow_currency)
    outcome = steps[0].run(context)
    with store.database.connect() as connection:
        charged = sum(
            connection.execute(sql, (identity,)).fetchone()[0]
            for sql, identity in (
                ("SELECT spent_seconds FROM search_job_inputs WHERE job_id=?", job),
                (
                    "SELECT spent_seconds FROM search_generations WHERE index_version_id=?",
                    outcome.checkpoint["index_id"],
                ),
            )
        )
    assert charged == pytest.approx(6)


@pytest.mark.parametrize(
    "planned", [preparation.SearchLimits(total_seconds=100, build_seconds=10)], indirect=True
)
def test_build_run_and_commit_share_one_attempt_deadline(store, monkeypatch, planned):
    clock, job, context, steps = planned
    currency = preparation._currency

    def timed_currency(*args):
        clock.advance(3)
        return currency(*args)

    monkeypatch.setattr(preparation, "_currency", timed_currency)
    began = clock.monotonic()
    outcome = replace(steps[0].run(context), started_at=began, completed_at=clock.monotonic())
    load_index = store.load_index

    def slow_graph(identity):
        clock.advance(5)
        return load_index(identity)

    monkeypatch.setattr(store, "load_index", slow_graph)
    with pytest.raises(PermanentStepFailure) as failure:
        steps[0].commit(
            store, StepCommitContext(job, preparation.PLAN, context.guard, "a" * 64, outcome)
        )
    assert failure.value.code == "search_time_limit"
    assert store.load_job_steps(job) == ()


def test_expired_plan_write_rolls_back_its_completion_but_keeps_durable_chunks(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, planned
):
    from core.persistence import store as persistence

    clock, job, context, steps = planned
    outcome = steps[0].run(context)
    save_index = persistence._save_index

    def slow_save(*args):
        result = save_index(*args)
        clock.advance(21)
        return result

    monkeypatch.setattr(persistence, "_save_index", slow_save)
    with pytest.raises(PermanentStepFailure) as failure:
        steps[0].commit(
            store, StepCommitContext(job, preparation.PLAN, context.guard, "a" * 64, outcome)
        )
    assert failure.value.code == "search_time_limit"
    assert store.load_job_steps(job) == ()
    assert store.load_index(outcome.checkpoint["index_id"]).chunks


@pytest.mark.parametrize("position", [1, 2, 3])
def test_every_later_checkpoint_write_is_charged_and_rolls_back_on_deadline(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, planned, position
):
    from core.persistence import store as persistence

    clock, job, context, steps = planned
    outcome = steps[0].run(context)
    steps[0].commit(
        store, StepCommitContext(job, preparation.PLAN, context.guard, "a" * 64, outcome)
    )
    store.finish_step_attempt(job, preparation.PLAN, context.attempt_number, None)
    step = steps[position]
    attempt = store.start_step_attempt(job, step.step_key)
    guard = AttemptGuard("accounting-tests", step.step_key, attempt.attempt_number)
    graph = None
    old_ready = None
    if position == 3:
        from core.search.runtime import vector_bytes
        from core.search.storage import SearchStorage

        graph = store.load_index(outcome.checkpoint["index_id"])
        storage = SearchStorage(store)
        storage.commit_batch(
            job,
            guard,
            graph.index.id,
            0,
            [c.id for c in graph.chunks],
            [vector_bytes([1.0] + [0.0] * 383, 384) for _ in graph.chunks],
        )
        storage.finish_components(job, guard, graph.index.id)
        old_ready = store.load_readiness(graph.index.source_id)
    save_step = persistence._save_job_step

    def slow_checkpoint(*args, **kwargs):
        result = save_step(*args, **kwargs)
        clock.advance(21)
        return result

    monkeypatch.setattr(persistence, "_save_job_step", slow_checkpoint)
    with pytest.raises(PermanentStepFailure) as failure:
        step.commit(store, StepCommitContext(job, step.step_key, guard, "b" * 64, outcome))
    assert failure.value.code == "search_time_limit"
    assert [r.step_key for r in store.load_job_steps(job)] == [preparation.PLAN]
    if position == 3:
        assert graph is not None
        assert store.load_readiness(graph.index.source_id) == old_ready


def test_slow_switch_final_checkpoint_rolls_back_activation_and_all_readiness(store, monkeypatch):
    from test_model_switch import mark_installed, pending_jobs
    from test_search_safety import prepare

    from core.persistence import store as persistence
    from core.search.switching import NotebookModelService, publish_replacement

    notebook, source, _, old, _, _ = seed(
        store, "rainfall", [1.0] + [0.0] * 383, profile_id="minilm-en"
    )
    mark_installed(store)
    service = NotebookModelService(store)
    service.choose(notebook, "multilingual-e5", 0, "slow-final-checkpoint")
    job = pending_jobs(store, notebook)[0][0]
    index, _ = prepare(store, job, profile_id="multilingual-e5")
    attempt = store.start_step_attempt(job, "publish-search")
    guard = AttemptGuard("safety-tests", "publish-search", attempt.attempt_number)
    clock = Clock()
    monkeypatch.setattr(persistence, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr("core.search.switching.time", SimpleNamespace(monotonic=clock.monotonic))
    save_step = persistence._save_job_step

    def slow_step(*args, **kwargs):
        result = save_step(*args, **kwargs)
        clock.advance(21)
        return result

    monkeypatch.setattr("core.search.switching._save_job_step", slow_step)
    with pytest.raises(persistence.ConflictingWriteError, match="deadline expired"):
        publish_replacement(
            store,
            job,
            guard,
            index,
            deadline=clock.monotonic() + 20,
            checkpoint=(1, {"ready": True}, "a" * 64),
        )
    assert store.load_readiness(source).index_version_ids == (old,)
    assert service.state(notebook)["pending_manifest_id"] is not None
    assert [row.step_key for row in store.load_job_steps(job)] == ["prepare"]
    with store.database.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM profile_search_publications").fetchone()[0]
            == 0
        )


def test_runner_preparation_wall_including_handoffs_and_checkpoints_matches_charge(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, planned
):
    import asyncio
    import time

    from core.jobs import runner as runner_module
    from core.persistence import store as persistence
    from core.search.runtime import vector_bytes

    clock, prior, context, _ = planned
    item = store.load_job_by_id(prior)
    assert item.extraction_version_id is not None
    store.finish_job(prior, "failed", "unused short-limit control")
    job = preparation.create_search_job(
        store,
        item.source_id,
        item.extraction_version_id,
        "minilm-en",
        limits=preparation.SearchLimits(total_seconds=200),
    )
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(monotonic=clock.monotonic))

    class Session:
        load_seconds = 1
        peak_bytes = 1024

        def __init__(self, *args, **kwargs):
            clock.advance(1)

        def embed(self, texts, **kwargs):
            clock.advance(2)
            return [vector_bytes([1.0] + [0.0] * 383, 384) for _ in texts]

        def close(self):
            clock.advance(1)

    monkeypatch.setattr(preparation, "EmbeddingSession", Session)
    save_step = persistence._save_job_step

    def timed_step(*args, **kwargs):
        result = save_step(*args, **kwargs)
        clock.advance(3)
        return result

    monkeypatch.setattr(persistence, "_save_job_step", timed_step)
    finish_attempt = store.finish_step_attempt

    def timed_bookkeeping(*args, **kwargs):
        clock.advance(2)
        return finish_attempt(*args, **kwargs)

    monkeypatch.setattr(store, "finish_step_attempt", timed_bookkeeping)
    steps = preparation.build_search_steps(store, Path("unused"))
    original_commit = steps[0].commit

    def queued_commit(*args):
        clock.advance(5)  # Time between a worker result and its DB commit.
        return original_commit(*args)

    steps = (replace(steps[0], commit=queued_commit), *steps[1:])

    async def execute():
        runner = runner_module.JobRunner(store, {"pdf-search": steps})
        await runner.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = store.load_job_status(job)
                assert status is not None
                if status.status in ("completed", "failed"):
                    assert status.status == "completed", status
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("finite preparation did not settle")
        finally:
            await runner.stop()

    began = clock.monotonic()
    asyncio.run(execute())
    with store.database.connect() as connection:
        charged = connection.execute(
            "SELECT g.spent_seconds FROM search_generations g JOIN job_steps s"
            " ON s.output_index_version_id=g.index_version_id WHERE s.job_id=? AND s.step_key=?",
            (job, preparation.PLAN),
        ).fetchone()[0]
    # Final attempt-status bookkeeping follows completed publication and is not
    # preparation. Earlier bookkeeping,5s handoff,load,inference,reap and all4
    # actual native checkpoint writes ARE charged,without double-counting.
    assert charged == pytest.approx(clock.monotonic() - began - 2)
    assert charged == pytest.approx(27)
    assert len(store.load_job_steps(job)) == 4


def test_runner_records_permanent_commit_failure_and_continues_next_job(store: PersistenceStore):
    import asyncio
    import time

    from core.jobs.runner import JobRunner, StepDefinition, StepOutcome

    _, source, version, _, previous, _ = seed(store, "rainfall", [1.0] + [0.0] * 383)
    store.finish_job(previous, "completed")
    failing = store.create_job("failed-commit", "commit-fault", source, version)
    following = store.create_job("following-commit", "following", source, version)
    store.create_job_status(failing.id)
    store.create_job_status(following.id)

    def fail_commit(*_):
        raise PermanentStepFailure("budget exhausted", code="search_time_limit")

    async def execute():
        runner = JobRunner(
            store,
            {
                "commit-fault": [
                    StepDefinition("commit", lambda _: StepOutcome(1, {}), commit=fail_commit)
                ],
                "following": [StepDefinition("done", lambda _: StepOutcome(1, {}))],
            },
        )
        await runner.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                following_status = store.load_job_status(following.id)
                assert following_status is not None
                if following_status.status == "completed":
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("next job did not complete")
        finally:
            await runner.stop()

    asyncio.run(execute())
    status = store.load_job_status(failing.id)
    assert status is not None
    assert status.status == "failed" and status.failure_code == "search_time_limit"
    assert len(store.load_step_attempts(failing.id, "commit")) == 1
    assert store.load_job_steps(failing.id) == ()
