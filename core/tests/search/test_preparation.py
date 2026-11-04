import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_storage import seed

from core.jobs.runner import JobRunner, PermanentStepFailure, StepContext
from core.persistence.store import PersistenceStore
from core.search.preparation import (
    SearchLimits,
    _budget,
    build_search_steps,
    create_search_job,
    effective_settings,
)
from core.search.profiles import profiles


@pytest.mark.parametrize("elapsed", [0.05, 0.3])
def test_unfinished_reservation_survives_cancellation_and_blocks_a_fresh_allowance(
    store: PersistenceStore, monkeypatch: pytest.MonkeyPatch, elapsed: float
) -> None:
    clock = SimpleNamespace(now=0.0)
    # Database scheduling must not choose whether this test's 200 ms lease expires.
    # Exercise both outcomes explicitly without changing the durable reservation.
    monkeypatch.setattr(
        "core.search.preparation.time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    _, _, _, index, job, guard = seed(store, "measurements", [1.0] + [0.0] * 383)
    context = StepContext(
        job, guard.step_key, guard.expected_attempt_number, lambda: False, lambda: False, guard
    )
    lease = _budget(store, context, 0.2, 0.2, index)
    assert lease.__enter__() == pytest.approx(0.2)
    with pytest.raises(PermanentStepFailure, match="budget exhausted"):
        with _budget(store, context, 0.2, 0.2, index):
            pass
    store.cancel_job(job)
    clock.now = elapsed
    if elapsed > 0.2:
        with pytest.raises(PermanentStepFailure, match="budget exhausted") as failure:
            lease.__exit__(None, None, None)
        assert failure.value.code == "search_time_limit"
    else:
        lease.__exit__(None, None, None)
    # A separate connection sees the committed conservative charge, not a reset.
    with store.database.connect() as connection:
        assert connection.execute(
            "SELECT spent_seconds FROM search_generations WHERE index_version_id=?", (index,)
        ).fetchone()[0] == pytest.approx(0.2)


def test_normal_budget_completion_charges_elapsed_time_not_full_reservation(
    store: PersistenceStore,
) -> None:
    _, _, _, index, job, guard = seed(store, "measurements", [1.0] + [0.0] * 383)
    context = StepContext(
        job, guard.step_key, guard.expected_attempt_number, lambda: False, lambda: False, guard
    )
    with _budget(store, context, 1, 1, index):
        time.sleep(0.01)
    with store.database.connect() as connection:
        spent = connection.execute(
            "SELECT spent_seconds FROM search_generations WHERE index_version_id=?", (index,)
        ).fetchone()[0]
    assert 0.01 <= spent < 1


def test_chunker_and_effective_settings_are_part_of_work_identity() -> None:
    profile = profiles()["minilm-en"]
    original = effective_settings(profile, SearchLimits(), None)
    reduced = effective_settings(profile, SearchLimits(), 8)
    assert original["chunker_version"] == "passage-token-v2"
    assert original["overlap_tokens"] == 32 and reduced["overlap_tokens"] == 8
    assert original["onnxruntime"]
    assert original["cpu_memory_arena"] is True
    assert original["idle_worker_policy"] == "stop-spinning-after-run-v1"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_time_limit_cannot_disable_deadlines(value: float) -> None:
    with pytest.raises(ValueError, match="invalid_search_limits"):
        SearchLimits(total_seconds=value)


def test_exhausted_source_budget_is_permanent_not_an_automatic_step_retry(
    store: PersistenceStore,
    monkeypatch: pytest.MonkeyPatch,
):
    # The one-second source allowance is intentionally below the embed stop margin.
    # Hold matching runner/step/store clocks so database scheduling cannot exhaust it
    # in planning first. The real test wait and database timestamps stay untouched.
    worker_clock = SimpleNamespace(monotonic=lambda: 0.0)
    monkeypatch.setattr("core.search.preparation.time", worker_clock)
    monkeypatch.setattr("core.jobs.runner.time", worker_clock)
    monkeypatch.setattr("core.persistence.store.time", worker_clock)
    _, source, _, index, prior_job, _ = seed(
        store,
        "rainfall",
        [1.0] + [0.0] * 383,
        profile_id="minilm-en",
    )
    store.finish_job(prior_job, "completed")
    graph = store.load_index(index)

    class Tokenizer:
        def __init__(self, *_):
            pass

        def count(self, _):
            return 1

        def prepare(self, text):
            return text

    monkeypatch.setattr("core.search.preparation.InputTokenizer", Tokenizer)
    job = create_search_job(
        store,
        source,
        graph.index.extraction_version_id,
        "minilm-en",
        limits=SearchLimits(total_seconds=1),
    )

    async def execute():
        runner = JobRunner(store, {"pdf-search": build_search_steps(store, Path("unused"))})
        await runner.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = store.load_job_status(job)
                if status and status.status == "failed":
                    return status
                await asyncio.sleep(0.02)
            raise AssertionError("budget control did not settle")
        finally:
            await runner.stop()

    status = asyncio.run(execute())
    assert status.failure_code == "search_time_limit"
    with store.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM job_step_attempts WHERE job_id=? AND step_key='embed-search'",
                (job,),
            ).fetchone()[0]
            == 1
        )
