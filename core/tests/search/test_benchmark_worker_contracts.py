"""Exercise the real worker protocol without importing or running native backends."""

import ctypes
import importlib.util
import io
import json
import socket
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from core.search import download, worker

VECTOR = struct.pack("<2f", 1, 0)


@pytest.fixture
def candidates(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setattr(sys, "argv", ["benchmark-worker-contract"])

    def forbidden(*args, **kwargs):
        pytest.fail("protocol control attempted a download, process or native library load")

    monkeypatch.setattr(download, "download_profile", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    # The CLI worker mutates these globals only in its real child. Restore them
    # after this in-process protocol control, including failure paths.
    for name in ("TEMP", "TMP", "HOME", "USERPROFILE"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    tokenizers = ModuleType("tokenizers")
    tokenizers.__dict__["Tokenizer"] = object
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    ort = ModuleType("onnxruntime")
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    def load(name):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    static = load("benchmark_static_multilingual")
    contextual = load("benchmark_contextual_multilingual")
    gguf = load("benchmark_bekko_gguf")
    openvino = load("benchmark_bekko_openvino")
    return SimpleNamespace(
        static=static, contextual=contextual, gguf=gguf, openvino=openvino, ort=ort
    )


@pytest.fixture
def profile(tmp_path):
    (tmp_path / "config.json").write_text('{"pad_token_id":0}', encoding="utf-8")
    return SimpleNamespace(
        id="controlled-benchmark",
        identity="a" * 64,
        model_file="controlled-model",
        pooling="sentence_embedding",
        dimensions=2,
        files=(),
    )


def protocol_result(monkeypatch, profile, directory, factory, flags=("1", "1"), *, entry=None):
    output = io.BytesIO()
    request = {"request": 7, "texts": ["controlled sample"], "query": True}
    with monkeypatch.context() as scoped:
        scoped.setattr(worker, "EmbeddingRuntime", factory)
        scoped.setattr(worker, "profiles", lambda: {profile.id: profile})
        scoped.setattr(
            worker, "runtime_versions", lambda: {"controlled": "stub; no native runtime"}
        )
        scoped.setattr(
            sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(request).encode() + b"\n"))
        )
        scoped.setattr(sys, "stdout", io.TextIOWrapper(output, write_through=True))
        arguments = [profile.id, str(directory), *flags]
        scoped.setattr(
            sys,
            "argv",
            ["benchmark-worker", *arguments]
            if entry is None
            else ["benchmark-worker", "--worker", "--directory", str(directory), "--", *arguments],
        )
        try:
            status = (entry or worker.main)()
        except SystemExit as error:
            status = error.code
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
    return status, messages


def controlled_embed(calls):
    def embed(self, texts, *, query=False):
        calls.append((texts, query))
        return [VECTOR for _ in texts]

    return embed


def assert_round_trip(status, messages, calls):
    assert status == 0
    assert [message["type"] for message in messages] == ["started", "ready", "result"]
    assert messages[-1] == {"type": "result", "request": 7, "vectors": [VECTOR.hex()]}
    assert calls == [(["controlled sample"], True)]


def test_static_constructor_accepts_current_worker_defaults(
    candidates, profile, tmp_path, monkeypatch
):
    candidate = candidates.static
    verified, calls = [], []
    monkeypatch.setattr(candidate, "verify_installation", lambda *args: verified.append(args))
    monkeypatch.setattr(candidate, "StaticTokenizer", lambda *args: object())
    monkeypatch.setattr(candidate, "map_embeddings", lambda *args, **kwargs: object())
    monkeypatch.setattr(candidate.StaticRuntime, "embed", controlled_embed(calls))
    status, messages = protocol_result(monkeypatch, profile, tmp_path, candidate.StaticRuntime)
    assert_round_trip(status, messages, calls)
    assert verified == [(tmp_path, profile)]


def contextual_session(candidates, monkeypatch):
    options = []

    class SessionOptions:
        def __init__(self):
            self.entries = {}
            options.append(self)

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    def session(path, *, providers, sess_options):
        assert providers == ["CPUExecutionProvider"]
        assert sess_options is options[-1]
        return SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="input_ids")],
            get_outputs=lambda: [SimpleNamespace(name="sentence_embedding")],
        )

    monkeypatch.setattr(candidates.ort, "SessionOptions", SessionOptions, raising=False)
    monkeypatch.setattr(candidates.ort, "InferenceSession", session, raising=False)
    monkeypatch.setattr(candidates.contextual, "verify_installation", lambda *args: None)
    monkeypatch.setattr(candidates.contextual, "UTF8InputTokenizer", lambda *args: object())
    return options


@pytest.mark.parametrize(
    "arena,spinning", [(True, True), (True, False), (False, True), (False, False)]
)
def test_contextual_protocol_applies_both_ort_choices(
    candidates, profile, tmp_path, monkeypatch, arena, spinning
):
    options = contextual_session(candidates, monkeypatch)
    calls = []
    monkeypatch.setattr(candidates.contextual.ContextualRuntime, "embed", controlled_embed(calls))
    status, messages = protocol_result(
        monkeypatch,
        profile,
        tmp_path,
        candidates.contextual.ContextualRuntime,
        (str(int(arena)), str(int(spinning))),
    )
    assert_round_trip(status, messages, calls)
    assert options[0].enable_cpu_mem_arena is arena
    assert options[0].intra_op_num_threads == options[0].inter_op_num_threads == 2
    assert options[0].entries == ({"session.force_spinning_stop": "1"} if spinning else {})


def test_direct_contextual_call_preserves_legacy_defaults(
    candidates, profile, tmp_path, monkeypatch
):
    options = contextual_session(candidates, monkeypatch)
    candidates.contextual.ContextualRuntime(tmp_path, profile)
    assert options[0].enable_cpu_mem_arena is False
    assert options[0].entries == {}


@pytest.mark.parametrize("name", ["static", "gguf"])
@pytest.mark.parametrize("flags", [("0", "1"), ("1", "0"), ("0", "0")])
def test_non_ort_protocol_rejects_alternate_ort_policies_before_loading(
    candidates, profile, tmp_path, monkeypatch, name, flags
):
    candidate = getattr(candidates, name)
    factory = candidate.StaticRuntime if name == "static" else candidate.GGufRuntime
    verified = []
    monkeypatch.setattr(candidate, "verify_installation", lambda *args: verified.append(args))
    status, messages = protocol_result(monkeypatch, profile, tmp_path, factory, flags)
    assert status == 1
    assert [message["type"] for message in messages] == ["started", "error"]
    assert messages[-1] == {"type": "error", "code": "runtime_mismatch"}
    assert verified == []


def test_gguf_cli_traced_factory_forwards_current_worker_keywords(
    candidates, profile, tmp_path, monkeypatch
):
    candidate = candidates.gguf
    constructed, calls = [], []

    def initialize(self, path, selected, *, memory_arena, stop_spinning):
        constructed.append((path, selected, memory_arena, stop_spinning))

    monkeypatch.setattr(candidate.GGufRuntime, "__init__", initialize)
    monkeypatch.setattr(candidate.GGufRuntime, "embed", controlled_embed(calls))
    monkeypatch.setattr(candidate, "owned", lambda path: path)
    monkeypatch.setattr(candidate, "load_profile", lambda manifest: profile)
    status, messages = protocol_result(
        monkeypatch, profile, tmp_path, candidate.GGufRuntime, entry=candidate.main
    )
    assert_round_trip(status, messages, calls)
    assert constructed == [(tmp_path, profile, True, True)]


@pytest.mark.parametrize("name", ["gguf", "lealla"])
@pytest.mark.parametrize("flag", ["--download", "--download-child"])
def test_historical_mixed_source_setup_fails_explicitly_before_transfer(
    candidates, profile, tmp_path, monkeypatch, capsys, name, flag
):
    candidate = candidates.gguf if name == "gguf" else candidates.contextual
    destination = tmp_path / "new-models"
    if name == "gguf":
        monkeypatch.setattr(candidate, "load_profile", lambda manifest: profile)
        arguments = [flag, "--models-directory", str(destination)]
    else:
        profile.id = "lealla-small"
        monkeypatch.setattr(candidate, "profiles", lambda: {profile.id: profile})
        arguments = [flag, "--profile", profile.id, "--directory", str(destination)]
    monkeypatch.setattr(sys, "argv", ["benchmark-setup", *arguments])
    with pytest.raises(SystemExit) as error:
        candidate.main()
    assert error.value.code == 2
    message = f"Fresh {'GGUF' if name == 'gguf' else 'LEALLA'} setup is unsupported"
    assert message in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("name", ["gguf", "lealla"])
def test_historical_setup_still_verifies_complete_cached_assets(
    candidates, profile, tmp_path, monkeypatch, name
):
    candidate = candidates.gguf if name == "gguf" else candidates.contextual
    destination = tmp_path / "cached-models"
    target = destination / profile.identity
    target.mkdir(parents=True)
    verified = []
    monkeypatch.setattr(candidate, "verify_installation", lambda *args: verified.append(args))
    if name == "gguf":
        monkeypatch.setattr(candidate, "load_profile", lambda manifest: profile)
        arguments = ["--download", "--models-directory", str(destination)]
    else:
        profile.id = "lealla-small"
        monkeypatch.setattr(candidate, "profiles", lambda: {profile.id: profile})
        arguments = ["--download", "--profile", profile.id, "--directory", str(destination)]
    monkeypatch.setattr(sys, "argv", ["benchmark-setup", *arguments])
    candidate.main()
    assert verified == [(target, profile)]


@pytest.mark.parametrize("mode", ["worker", "screen"])
def test_bekko_openvino_reports_unsupported_protocol_before_entering_old_adapter(
    candidates, tmp_path, monkeypatch, capsys, mode
):
    def unsupported(*args):
        pytest.fail("historical OpenVINO adapter was entered despite its incompatible protocol")

    monkeypatch.setattr(candidates.openvino.backend_tools, "worker", unsupported)
    monkeypatch.setattr(candidates.openvino, "smoke", unsupported)
    arguments = ["--directory", str(tmp_path)]
    if mode == "worker":
        arguments += ["--worker"]
    else:
        arguments += ["--models-directory", str(tmp_path), "--capacity-run", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["benchmark-worker", *arguments])
    with pytest.raises(SystemExit) as error:
        candidates.openvino.main()
    assert error.value.code == 2
    assert "Historical OpenVINO worker replay is unsupported" in capsys.readouterr().err
    assert not (tmp_path / "compiled-cache").exists()
