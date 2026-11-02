"""New E5 arena choice; genuine SQLite jobs preserve saved flags and old work."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_batch_reservations import embed_stage
from test_preparation_accounting import Clock
from test_query import install
from test_storage import seed

from core.jobs.runner import StepContext
from core.persistence.models import AttemptGuard
from core.search import preparation
from core.search.profiles import profiles
from core.search.query import QueryService
from core.search.runtime import vector_bytes

VECTOR = [1.0] + [0.0] * 383


@pytest.mark.parametrize("profile_id", ["bekko-a8m-quint8-avx2", "minilm-en", "multilingual-e5"])
def test_only_calculated_e5_arena_choice_changes(profile_id):
    saved = preparation.effective_settings(profiles()[profile_id], preparation.SearchLimits(), None)
    assert saved["cpu_memory_arena"] is (profile_id != "multilingual-e5")
    assert saved["threads"] == 2 and saved["idle_worker_policy"] == "stop-spinning-after-run-v1"


@pytest.mark.parametrize("legacy_budget", [False, True])
@pytest.mark.parametrize("saved_true", [False, True])
def test_both_preparation_branches_honor_their_saved_arena_without_rewriting_old_work(
    store, monkeypatch, legacy_budget, saved_true
):
    _, source, _, old_index, prior, _ = seed(
        store,
        "rainfall measurements",
        VECTOR,
        profile_id="multilingual-e5",
        runtime_settings={"cpu_memory_arena": True},
    )
    store.finish_job(prior, "completed")
    extraction = store.load_index(old_index).index.extraction_version_id
    with store.database.connect() as connection:
        original = dict(
            connection.execute(
                "SELECT * FROM search_generations WHERE index_version_id=?", (old_index,)
            ).fetchone()
        )

    class Tokenizer:
        def __init__(self, *_):
            pass

        def count(self, text):
            return 1

        def prepare(self, text):
            return text

    clock = Clock()
    monkeypatch.setattr(preparation, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr("core.persistence.store.time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr("core.search.storage.time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(preparation, "InputTokenizer", Tokenizer)
    calculate = preparation.effective_settings

    def saved_settings(*args, **kwargs):
        values = calculate(*args, **kwargs)
        if legacy_budget:
            values.pop("budget_policy", None)
        if saved_true:
            values["cpu_memory_arena"] = True
        return values

    with monkeypatch.context() as scoped:
        scoped.setattr(preparation, "effective_settings", saved_settings)
        job = preparation.create_search_job(
            store,
            source,
            extraction,
            "multilingual-e5",
            limits=preparation.SearchLimits(batch_size=8, total_seconds=20),
        )
    settings = preparation._inputs(store, job)[1]
    assert settings["cpu_memory_arena"] is saved_true
    assert settings["openblas_threads"] == 1
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    assert fingerprint in store.load_job_by_id(job).logical_work_key
    store.claim_job(job, "accounting-tests")
    attempt = store.start_step_attempt(job, preparation.PLAN)
    guard = AttemptGuard("accounting-tests", preparation.PLAN, attempt.attempt_number)
    context = StepContext(
        job, preparation.PLAN, attempt.attempt_number, lambda: False, lambda: False, guard
    )
    steps = preparation.build_search_steps(store, Path("unused"))
    _, _, context, step, sessions, _ = embed_stage(store, (clock, job, context, steps), monkeypatch)
    step.run(context)
    assert sessions and all(session.options["memory_arena"] is saved_true for session in sessions)
    assert all(session.options["openblas_threads"] == 1 and session.closed for session in sessions)
    assert preparation._inputs(store, job)[1] == settings
    with store.database.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM search_batches WHERE complete=1 AND index_version_id<>?",
                (old_index,),
            ).fetchone()[0]
            == 1
        )
        assert (
            dict(
                connection.execute(
                    "SELECT * FROM search_generations WHERE index_version_id=?", (old_index,)
                ).fetchone()
            )
            == original
        )


def test_query_serially_honors_old_true_and_new_false_arena_flags(store):
    options = {"cpu_memory_arena": True, "idle_worker_policy": "stop-spinning-after-run-v1"}
    notebook, first, _, _, _, _ = seed(
        store, "rainfall", VECTOR, profile_id="multilingual-e5", runtime_settings=options
    )
    _, second, _, _, _, _ = seed(
        store,
        "rainfall",
        VECTOR,
        profile_id="multilingual-e5",
        notebook_id=notebook,
        runtime_settings={**options, "cpu_memory_arena": False, "openblas_threads": 1},
    )
    install(store, "multilingual-e5")
    active, seen = 0, []

    class Session:
        def __init__(self, path, profile, **kwargs):
            assert active == 0 and kwargs["stop_spinning"] is True
            self.profile = profile
            seen.append((kwargs["memory_arena"], kwargs.get("openblas_threads")))

        def __enter__(self):
            nonlocal active
            active += 1
            return self

        def embed(self, texts, **kwargs):
            assert active == 1
            return [vector_bytes(VECTOR, 384)]

        def __exit__(self, *_):
            nonlocal active
            active -= 1

    result = QueryService(store, store.database.config.data_dir, session_factory=Session).search(
        notebook, "rainfall", "meaning"
    )
    assert active == 0 and set(seen) == {(True, None), (False, 1)}
    assert {row["source_id"] for row in result["results"]} == {first, second}


def test_real_e5_arena_off_privacy(tmp_path, request, monkeypatch, capfd, caplog):
    import test_privacy

    original = test_privacy.EmbeddingSession

    def session(*args, **kwargs):
        return original(*args, memory_arena=False, openblas_threads=1, **kwargs)

    monkeypatch.setattr(test_privacy, "EmbeddingSession", session)
    test_privacy.test_real_offline_worker_does_not_spill_source_query_or_vectors_to_scratch(
        tmp_path, request, monkeypatch, capfd, caplog, "multilingual-e5"
    )


def test_real_e5_arena_off_inference_and_overflow(request, monkeypatch):
    import test_runtime

    original = test_runtime.EmbeddingRuntime

    def runtime(*args, **kwargs):
        return original(*args, memory_arena=False, **kwargs)

    monkeypatch.setattr(test_runtime, "EmbeddingRuntime", runtime)
    test_runtime.test_real_installed_inference_is_offline_and_does_not_truncate(
        request, monkeypatch, "multilingual-e5"
    )
