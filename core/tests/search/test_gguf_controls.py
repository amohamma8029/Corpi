"""Independent ABI, whitespace-ID and sequence-isolation controls without native execution."""

import ctypes as ct
import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "gguf_benchmark_controls", scripts / "benchmark_bekko_gguf.py"
    )
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_native_layout_matches_pinned_windows_header(monkeypatch):
    candidate = module(monkeypatch)
    assert ct.sizeof(candidate.ModelParams) == 80
    assert ct.sizeof(candidate.ContextParams) == 160
    assert ct.sizeof(candidate.Batch) == 56
    assert candidate.ContextParams.n_threads.offset == 28
    assert candidate.ContextParams.pooling_type.offset == 44
    assert candidate.ContextParams.embeddings.offset == 128
    assert candidate.ContextParams.samplers.offset == 136


def test_only_exact_known_whitespace_display_aliases_are_accepted(monkeypatch):
    candidate = module(monkeypatch)
    assert candidate.known_display_alias({"id": 139, "hf": "\u2581\u2581", "gguf": "  "})
    assert candidate.known_display_alias({"id": 226, "hf": "\t", "gguf": "<0x09>"})
    assert not candidate.known_display_alias({"id": 139, "hf": "wrong", "gguf": "  "})
    assert not candidate.known_display_alias({"id": 140, "hf": "\u2581\u2581", "gguf": "  "})
    assert not candidate.known_display_alias({"id": 226, "hf": "\n", "gguf": "<0x09>"})


def test_native_batch_keeps_hf_ids_positions_sequence_mean_and_order(monkeypatch):
    candidate = module(monkeypatch)
    runtime = candidate.GGufRuntime.__new__(candidate.GGufRuntime)
    runtime.physical_batch = 2
    runtime.context = None
    ids = {"one": [2, 123, 1], "two": [2, 456, 457, 1], "oversized": [2] * 513}
    runtime.tokenizer = SimpleNamespace(
        prepare=lambda text, **_: text,
        tokenizer=SimpleNamespace(encode=lambda text: SimpleNamespace(ids=ids[text])),
    )
    storage = [(ct.c_int32 * 1)() for _ in range(1024)]
    runtime.batch = candidate.Batch(
        0,
        (ct.c_int32 * 1024)(),
        None,
        (ct.c_int32 * 1024)(),
        (ct.c_int32 * 1024)(),
        (ct.POINTER(ct.c_int32) * 1024)(*storage),
        (ct.c_int8 * 1024)(),
    )
    runtime.get_memory = lambda _: None
    runtime.clear_memory = lambda *_: pytest.fail("no memory handle should be cleared")
    calls = []

    def process(context, batch):
        calls.append(
            [
                (batch.token[i], batch.pos[i], batch.seq_id[i][0], batch.logits[i])
                for i in range(batch.n_tokens)
            ]
        )
        return 0

    runtime.process = process
    vectors = [(ct.c_float * 384)(1, 0), (ct.c_float * 384)(0, 1)]
    runtime.get_embedding = lambda context, identity: vectors[identity]
    result = runtime.embed(["one", "two"])
    assert calls == [
        [
            (2, 0, 0, 1),
            (123, 1, 0, 1),
            (1, 2, 0, 1),
            (2, 0, 1, 1),
            (456, 1, 1, 1),
            (457, 2, 1, 1),
            (1, 3, 1, 1),
        ]
    ]
    assert struct.unpack("<2f", result[0][:8]) == (1, 0)
    assert struct.unpack("<2f", result[1][:8]) == (0, 1)
    with pytest.raises(ValueError, match="overflow"):
        runtime.embed(["one", "oversized"])
    assert len(calls) == 1  # reject the whole request before partial native computation
