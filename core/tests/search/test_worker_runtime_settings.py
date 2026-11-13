"""New workers opt in; immutable legacy inputs and mixed notebook choices survive."""

import hashlib
import json
import sys

import pytest
from test_query import install
from test_storage import seed

from core.search.preparation import (
    BATCH_BUDGET_POLICY,
    SearchLimits,
    _inputs,
    create_search_job,
    effective_settings,
)
from core.search.profiles import profiles
from core.search.query import QueryService, SearchFailure
from core.search.representation import ARABIC, CANONICAL, search_text
from core.search.runtime import vector_bytes
from core.search.supervisor import (
    EmbeddingFailure,
    EmbeddingSession,
    openblas_options,
    worker_environment,
)

VECTOR = [1.0] + [0.0] * 383
RUNTIME = {"cpu_memory_arena": True, "idle_worker_policy": "stop-spinning-after-run-v1"}


@pytest.mark.parametrize("profile_id", ["bekko-a8m-quint8-avx2", "minilm-en", "multilingual-e5"])
def test_real_new_worker_privacy(tmp_path, request, monkeypatch, capfd, caplog, profile_id):
    import test_privacy

    original = test_privacy.EmbeddingSession

    def selected_session(*args, **kwargs):
        return original(*args, openblas_threads=1, **kwargs)

    monkeypatch.setattr(test_privacy, "EmbeddingSession", selected_session)
    test_privacy.test_real_offline_worker_does_not_spill_source_query_or_vectors_to_scratch(
        tmp_path, request, monkeypatch, capfd, caplog, profile_id
    )


@pytest.mark.parametrize("value", [None, True, False, 1.0, "1", 0, 2, -1])
def test_present_unsupported_saved_setting_is_rejected_instead_of_coerced(value):
    with pytest.raises(ValueError, match="^unsupported_worker_settings$"):
        openblas_options({"openblas_threads": value})


@pytest.mark.parametrize("value", [None, True, False, 1.0, "1", 0, 2, -1])
def test_bad_saved_preparation_choice_is_rejected_before_work(store, monkeypatch, value):
    from core.search import preparation

    _, source, _, index, original_job, _ = seed(store, "rainfall", VECTOR, profile_id="minilm-en")
    store.finish_job(original_job, "completed")
    extraction = store.load_index(index).index.extraction_version_id
    original = preparation.effective_settings

    def invalid_settings(*args, **kwargs):
        return {**original(*args, **kwargs), "openblas_threads": value}

    with monkeypatch.context() as scoped:
        scoped.setattr(preparation, "effective_settings", invalid_settings)
        job = create_search_job(store, source, extraction, "minilm-en")
    with pytest.raises(ValueError, match="^unsupported_worker_settings$"):
        _inputs(store, job)


def test_missing_setting_and_environment_do_not_silently_adopt_parent_options(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "31")
    monkeypatch.setenv("OMP_NUM_THREADS", "17")
    monkeypatch.setenv("MKL_NUM_THREADS", "19")
    monkeypatch.setenv("CORPI_TRIAL_UNTRUSTED", "not-forwarded")
    assert openblas_options({}) == {}
    assert openblas_options({"openblas_threads": 1}) == {"openblas_threads": 1}
    legacy = worker_environment()
    selected = worker_environment(openblas_threads=1)
    assert "OPENBLAS_NUM_THREADS" not in legacy
    assert selected == {**legacy, "OPENBLAS_NUM_THREADS": "1"}
    assert not {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "CORPI_TRIAL_UNTRUSTED"} & selected.keys()


@pytest.mark.parametrize("selected", [False, True])
def test_real_child_receives_only_the_explicit_choice(tmp_path, selected, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "31")
    expected = "'1'" if selected else "None"
    body = f"""
import json,os,sys,time
from core.search.provenance import runtime_versions
started = {{'type':'started','manifest':sys.argv[1], 'request':0,'pid':os.getpid()}}
print(json.dumps(started),flush=True)
assert os.environ.get('OPENBLAS_NUM_THREADS') == {expected}
assert 'OMP_NUM_THREADS' not in os.environ and 'MKL_NUM_THREADS' not in os.environ
print(json.dumps({{'type':'ready','manifest':sys.argv[1],'request':0,
                  'runtime':runtime_versions()}}),flush=True)
time.sleep(30)
"""
    profile = profiles()["minilm-en"]
    body = body.replace("sys.argv[1]", repr(profile.identity))
    with EmbeddingSession(
        tmp_path,
        profile,
        command=[sys.executable, "-B", "-c", body],
        load_timeout=10,
        openblas_threads=1 if selected else None,
    ) as session:
        assert session.process.poll() is None and session.worker_pid is not None
    assert session.process.poll() is not None and session._closed


def test_new_enqueue_fingerprints_setting_without_rewriting_old_generation(store):
    _, source, _, index, original_job, _ = seed(store, "rainfall", VECTOR, profile_id="minilm-en")
    store.finish_job(original_job, "completed")
    with store.database.connect() as connection:
        before = dict(
            connection.execute(
                "SELECT * FROM search_generations WHERE index_version_id=?", (index,)
            ).fetchone()
        )
        ready = tuple(
            connection.execute(
                "SELECT * FROM source_readiness_indexes WHERE source_id=?", (source,)
            ).fetchone()
        )
    extraction = store.load_index(index).index.extraction_version_id
    job = create_search_job(store, source, extraction, "minilm-en")
    with store.database.connect() as connection:
        saved = json.loads(
            connection.execute(
                "SELECT settings_json FROM search_job_inputs WHERE job_id=?", (job,)
            ).fetchone()[0]
        )
        assert (
            dict(
                connection.execute(
                    "SELECT * FROM search_generations WHERE index_version_id=?", (index,)
                ).fetchone()
            )
            == before
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT * FROM source_readiness_indexes WHERE source_id=?", (source,)
                ).fetchone()
            )
            == ready
        )
    legacy = effective_settings(
        profiles()["minilm-en"], SearchLimits(), None, budget_policy=BATCH_BUDGET_POLICY
    )
    assert saved == {**legacy, "openblas_threads": 1}
    fingerprint = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
    assert fingerprint in store.load_job_by_id(job).logical_work_key
    assert "openblas_threads" not in json.loads(before["settings_json"])


@pytest.mark.parametrize("choices", ["legacy", "new", "mixed"])
def test_query_uses_saved_options_serially_and_keeps_representations_and_words(store, choices):
    text = "ﻣﻄـﺮ rainfall"
    first_options = {**RUNTIME, **({"openblas_threads": 1} if choices == "new" else {})}
    notebook, first, _, _, _, _ = seed(
        store, text, VECTOR, profile_id="minilm-en", runtime_settings=first_options
    )
    new = {**RUNTIME, **({"openblas_threads": 1} if choices != "legacy" else {})}
    _, second, _, _, _, _ = seed(
        store, text, VECTOR, profile_id="minilm-en", notebook_id=notebook, runtime_settings=new
    )
    _, third, _, _, _, _ = seed(
        store,
        text,
        VECTOR,
        profile_id="minilm-en",
        notebook_id=notebook,
        representation=ARABIC,
        runtime_settings=new,
    )
    install(store, "minilm-en")
    sessions, calls = [], []
    active = 0

    class Session:
        def __init__(self, path, profile, **options):
            nonlocal active
            assert active == 0
            assert options["memory_arena"] is True and options["stop_spinning"] is True
            self.profile, self.options = profile, options
            sessions.append(self)

        def __enter__(self):
            nonlocal active
            active += 1
            return self

        def embed(self, texts, **options):
            assert active == 1 and options == {"query": True, "timeout": 120}
            # A real write borrow is possible during inference; words need no model admission.
            with store._transaction() as connection:
                assert connection.in_transaction
            assert service.search(notebook, "rainfall", "words")["ready_sources"] == 3
            calls.append((self.options.get("openblas_threads"), texts))
            return [vector_bytes(VECTOR, 384)]

        def __exit__(self, *_):
            nonlocal active
            active -= 1

    service = QueryService(store, store.database.config.data_dir, session_factory=Session)
    result = service.search(notebook, text, "meaning")
    assert active == 0 and len(sessions) == (2 if choices == "mixed" else 1)
    assert {row["source_id"] for row in result["results"]} == {first, second, third}
    assert all(
        row["text"] == text and row["citations"][0]["page_number"] == 1 for row in result["results"]
    )
    if choices == "mixed":
        assert (None, [text]) in calls and (1, [text]) in calls
        assert (1, [search_text(text, ARABIC)]) in calls
    else:
        threads = 1 if choices == "new" else None
        assert sorted((thread, texts[0]) for thread, texts in calls) == sorted(
            (threads, search_text(text, representation)) for representation in (CANONICAL, ARABIC)
        )


def test_mixed_query_failure_releases_admission_and_closes_prior_group(store):
    notebook, _, _, _, _, _ = seed(store, "rainfall", VECTOR, profile_id="minilm-en")
    seed(
        store,
        "rainfall",
        VECTOR,
        profile_id="minilm-en",
        notebook_id=notebook,
        runtime_settings={"openblas_threads": 1},
    )
    install(store, "minilm-en")
    started, closed = [], []

    class Session:
        def __init__(self, path, profile, **options):
            started.append(options)
            if len(started) == 2:
                assert len(closed) == 1
                raise EmbeddingFailure("embedding_worker_failed")
            self.profile = profile

        def __enter__(self):
            return self

        def embed(self, texts, **options):
            return [vector_bytes(VECTOR, 384)]

        def __exit__(self, *_):
            closed.append(True)

    service = QueryService(store, store.database.config.data_dir, session_factory=Session)
    with pytest.raises(SearchFailure) as error:
        service.search(notebook, "rainfall", "meaning")
    assert error.value.code == "embedding_worker_failed" and len(closed) == 1
    assert service._meaning_query.acquire(blocking=False)
    service._meaning_query.release()


@pytest.mark.parametrize("value", [None, True, 1.0, "1", 2])
def test_bad_saved_worker_choice_fails_meaning_without_blocking_words(store, value):
    notebook, _, _, _, _, _ = seed(
        store,
        "rainfall",
        VECTOR,
        profile_id="minilm-en",
        runtime_settings={"openblas_threads": value},
    )
    install(store, "minilm-en")
    service = QueryService(
        store,
        store.database.config.data_dir,
        session_factory=lambda *a, **k: pytest.fail("invalid setting reached a worker"),
    )
    with pytest.raises(SearchFailure) as error:
        service.search(notebook, "rainfall", "meaning")
    assert error.value.code == "search_input_changed"
    assert service.search(notebook, "rainfall", "words")["results"]
