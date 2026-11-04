"""Public marker controls for worker diagnostics, durable errors and scratch files.

Controlled workers prove failure boundaries, not model quality or an OS-wide privacy audit.
"""

import hashlib
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_embedding_supervisor import assert_process_stopped, command
from test_storage import seed

from core.api.app import create_app
from core.jobs.runner import JobRunner, StepDefinition, StepOutcome, StepPolicy
from core.persistence.models import new_id
from core.search.handoff import SUPPORTED_PROFILE_IDS
from core.search.preparation import build_search_steps, create_search_job
from core.search.profiles import profiles, verify_installation
from core.search.runtime import InputTokenizer, vector_bytes
from core.search.supervisor import EmbeddingFailure, EmbeddingSession

SOURCE = "PUBLIC_FIXTURE_PRIVATE_SOURCE_MARKER"
QUERY = "PUBLIC_FIXTURE_PRIVATE_QUERY_MARKER"
VECTOR = vector_bytes([1.0] + [0.0] * 383, 384)
DIAGNOSTIC = json.dumps({"source": SOURCE, "query": QUERY, "vector": VECTOR.hex()})


@pytest.fixture(autouse=True)
def capture_all_log_levels(caplog):
    caplog.set_level(logging.DEBUG)


def assert_private(*outputs):
    for output in outputs:
        assert SOURCE not in output and QUERY not in output and VECTOR.hex() not in output


def file_hashes(root):
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(root)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def assert_closed(session):
    assert session._closed
    assert session.process.poll() is not None
    assert session.worker_pid is not None
    assert_process_stopped(session.worker_pid)


@pytest.mark.parametrize("phase", ["load", "encode"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_parent_tokenizer_does_not_propagate_native_diagnostics(
    tmp_path, monkeypatch, capfd, caplog, phase, error_type
):
    calls: list[str] = []

    class NativeTokenizer:
        def no_truncation(self):
            pass

        def no_padding(self):
            pass

        def encode(self, *args, **kwargs):
            calls.append("encode")
            raise error_type(DIAGNOSTIC)

    def load(_path):
        calls.append("load")
        if phase == "load":
            raise error_type(DIAGNOSTIC)
        return NativeTokenizer(), {}

    monkeypatch.setattr("core.search.runtime._load_runtime_tokenizer", load)
    with pytest.raises(ValueError) as failure:
        tokenizer = InputTokenizer(tmp_path, profiles()["minilm-en"])
        tokenizer.count(SOURCE)
    assert calls == (["load"] if phase == "load" else ["load", "encode"])
    assert str(failure.value) == "tokenizer_failed"
    captured = capfd.readouterr()
    assert_private(
        "".join(traceback.format_exception(failure.value)), captured.out, captured.err, caplog.text
    )


def recorded_session(sessions, worker):
    class RecordingSession(EmbeddingSession):
        def __init__(self, path, profile, **kwargs):
            sessions.append(self)
            super().__init__(path, profile, command=worker(profile), **kwargs)

    return RecordingSession


def response_worker(profile, payload, phase="document"):
    body = ""
    if phase != "load":
        ready = {"type": "ready", "manifest": profile.identity, "request": 0}
        body += f"print(json.dumps({ready!r}),flush=True)\n"
        body += "request=json.loads(sys.stdin.readline())\n"
        expected = QUERY if phase == "query" else SOURCE
        body += f"assert request['texts']==[{expected!r}]\n"
        body += f"assert request['query'] is {phase == 'query'}\n"
    body += f"print({DIAGNOSTIC!r},file=sys.stderr,flush=True)\n"
    body += f"print(json.dumps({payload!r}),flush=True)\ntime.sleep(30)"
    return command(profile, body)


@pytest.mark.parametrize("phase", ["load", "document", "query"])
@pytest.mark.parametrize(
    "code", [DIAGNOSTIC, {"native_error": DIAGNOSTIC}, [DIAGNOSTIC]], ids=["text", "object", "list"]
)
def test_parent_rejects_untrusted_error_codes_without_disclosing_payload(
    tmp_path, capfd, caplog, phase, code
):
    sessions = []
    session_type = recorded_session(
        sessions, lambda profile: response_worker(profile, {"type": "error", "code": code}, phase)
    )
    with pytest.raises(EmbeddingFailure) as failure:
        with session_type(tmp_path, profiles()["minilm-en"], load_timeout=5) as session:
            session.embed(
                [QUERY if phase == "query" else SOURCE], query=phase == "query", timeout=5
            )
    assert failure.value.code == "embedding_worker_failed"
    captured = capfd.readouterr()
    assert_private(
        str(failure.value),
        "".join(traceback.format_exception(failure.value)),
        captured.out,
        captured.err,
        caplog.text,
    )
    assert_closed(sessions[0])
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("code", ["model_input_too_long", "model_corrupt", "tokenizer_failed"])
def test_known_worker_failure_codes_remain_actionable(tmp_path, code):
    sessions = []
    session_type = recorded_session(
        sessions, lambda profile: response_worker(profile, {"type": "error", "code": code}, "load")
    )
    with pytest.raises(EmbeddingFailure) as failure:
        session_type(tmp_path, profiles()["minilm-en"], load_timeout=5)
    assert failure.value.code == code
    assert_closed(sessions[0])


@pytest.mark.parametrize(
    "vectors,expected",
    [
        (DIAGNOSTIC, "embedding_protocol_error"),
        ([{"native_error": DIAGNOSTIC}], "embedding_protocol_error"),
        ([DIAGNOSTIC], "embedding_protocol_error"),
        ([], "invalid_vector_count"),
        ([bytes(384 * 4).hex()], "invalid_vector"),
    ],
    ids=["non-list", "non-string", "bad-hex", "missing", "zero-vector"],
)
def test_malformed_vector_payloads_have_only_safe_failure_codes(
    tmp_path, capfd, caplog, vectors, expected
):
    sessions = []
    session_type = recorded_session(
        sessions,
        lambda profile: response_worker(
            profile, {"type": "result", "request": 1, "vectors": vectors}
        ),
    )
    with pytest.raises(EmbeddingFailure) as failure:
        with session_type(tmp_path, profiles()["minilm-en"], load_timeout=5) as session:
            session.embed([SOURCE], timeout=5)
    assert failure.value.code == expected
    captured = capfd.readouterr()
    assert_private(str(failure.value), captured.out, captured.err, caplog.text)
    assert_closed(sessions[0])


@pytest.mark.parametrize("phase", ["load", "document", "query"])
@pytest.mark.parametrize("error_type", ["ValueError", "RuntimeError"])
def test_actual_worker_sanitizes_native_exceptions_and_discards_diagnostics(
    tmp_path, capfd, caplog, phase, error_type
):
    script = f"""
import sys
from core.search import worker
class FailingRuntime:
    def __init__(self, *args):
        if {phase == "load"}:
            print({DIAGNOSTIC!r}, flush=True)
            raise {error_type}({DIAGNOSTIC!r})
    def embed(self, texts, query=False):
        assert texts == [{(QUERY if phase == "query" else SOURCE)!r}]
        assert query is {phase == "query"}
        print({DIAGNOSTIC!r}, flush=True)
        raise {error_type}({DIAGNOSTIC!r})
worker.EmbeddingRuntime = FailingRuntime
raise SystemExit(worker.main())
"""
    sessions = []
    session_type = recorded_session(sessions, lambda _: [sys.executable, "-c", script])
    with pytest.raises(EmbeddingFailure) as failure:
        with session_type(tmp_path, profiles()["minilm-en"], load_timeout=5) as session:
            session.embed(
                [QUERY if phase == "query" else SOURCE], query=phase == "query", timeout=5
            )
    assert failure.value.code == "embedding_worker_failed"
    captured = capfd.readouterr()
    assert_private(str(failure.value), captured.out, captured.err, caplog.text)
    assert_closed(sessions[0])
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"type": "error", "code": DIAGNOSTIC}, "embedding_worker_failed"),
        ({"type": "result", "request": 1, "vectors": [DIAGNOSTIC]}, "embedding_protocol_error"),
        (None, None),
    ],
    ids=["untrusted-error", "bad-vector", "parent-tokenizer"],
)
def test_preparation_failure_records_and_api_never_echo_native_payload(
    store, tmp_path, monkeypatch, capfd, caplog, payload, expected
):
    tokenizer_calls: list[str] = []
    _, source, version, _, prior_job, _ = seed(store, SOURCE, [1.0] + [0.0] * 383)
    store.finish_job(prior_job, "completed")
    extraction = store.load_job_by_id(prior_job).extraction_version_id
    assert extraction is not None
    job = create_search_job(store, source, extraction, "minilm-en")
    following = store.create_job(new_id(), "privacy-control", source, version)
    store.create_job_status(following.id)
    sessions = []
    monkeypatch.setattr(
        "core.search.preparation.EmbeddingSession",
        recorded_session(sessions, lambda profile: response_worker(profile, payload)),
    )
    if payload is None:

        def fail_tokenizer(_path):
            tokenizer_calls.append("load")
            raise RuntimeError(DIAGNOSTIC)

        monkeypatch.setattr("core.search.runtime._load_runtime_tokenizer", fail_tokenizer)
    else:
        monkeypatch.setattr(
            "core.search.preparation.InputTokenizer",
            lambda *args: SimpleNamespace(count=lambda _: 1, prepare=lambda text: text),
        )
    runner = JobRunner(
        store,
        {
            "pdf-search": build_search_steps(store, tmp_path / "models"),
            "privacy-control": [
                StepDefinition("following", lambda _: StepOutcome(1, {"ok": True}))
            ],
        },
        policies={"search_prepare": StepPolicy(10, 1, 0), "index_build": StepPolicy(10, 1, 0)},
    )
    app = create_app("privacy-test-token", store=store, runner=runner)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        deadline = time.monotonic() + 20
        while (
            status := store.load_job_status(following.id)
        ) is None or status.status != "completed":
            assert time.monotonic() < deadline, "following job did not finish"
            time.sleep(0.02)
        headers = {"Authorization": "Bearer privacy-test-token"}
        responses = [
            client.get("/jobs", headers=headers),
            client.get(f"/jobs/{job}", headers=headers),
        ]
    status = store.load_job_status(job)
    assert status is not None and status.status == "failed" and status.failure_code == expected
    assert all(response.status_code == 200 for response in responses)
    attempts = store.load_job_attempts(job)
    assert tokenizer_calls == (["load"] if payload is None else [])
    if payload is None:
        assert "tokenizer_failed" in str(attempts)
    captured = capfd.readouterr()
    assert_private(
        str(status),
        str(attempts),
        *(response.text for response in responses),
        captured.out,
        captured.err,
        caplog.text,
    )
    assert all(step.step_key != "publish-search" for step in store.load_job_steps(job))
    assert len(sessions) == (0 if payload is None else 1)
    for session in sessions:
        assert_closed(session)
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("profile_id", SUPPORTED_PROFILE_IDS)
def test_real_offline_worker_does_not_spill_source_query_or_vectors_to_scratch(
    tmp_path, request, monkeypatch, capfd, caplog, profile_id
):
    directory = request.config.getoption("--models-directory", default=None)
    if directory is None:
        pytest.skip("explicit installed-model directory required; no downloads")
    profile = profiles()[profile_id]
    source = Path(directory) / profile.identity
    verify_installation(source, profile)
    assets = tmp_path / "models" / profile.identity
    shutil.copytree(source, assets)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMP", str(scratch))
    monkeypatch.setenv("TEMP", str(scratch))
    monkeypatch.setenv("CORPI_CORE_TOKEN", "public-fixture-core-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "public-fixture-provider-secret")
    monkeypatch.chdir(scratch)
    before = file_hashes(tmp_path)
    script = """
import os
import socket
assert 'CORPI_CORE_TOKEN' not in os.environ and 'OPENAI_API_KEY' not in os.environ
def deny(*args, **kwargs):
    raise AssertionError('network attempted during offline inference')
socket.create_connection = deny
socket.socket.connect = deny
socket.socket.connect_ex = deny
from core.search.worker import main
raise SystemExit(main())
"""
    with EmbeddingSession(
        assets, profile, command=[sys.executable, "-c", script], load_timeout=30
    ) as session:
        documents = session.embed([SOURCE], timeout=30)
        queries = session.embed([QUERY], query=True, timeout=30)
    assert len(documents) == len(queries) == 1
    captured = capfd.readouterr()
    assert_private(captured.out, captured.err, caplog.text)
    for vector in (*documents, *queries):
        assert vector.hex() not in captured.out + captured.err + caplog.text
    assert file_hashes(tmp_path) == before
    verify_installation(assets, profile)
    assert_closed(session)
