"""Shared inference tokenizer must not weaken overflow or change padding/options."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from core.search.profiles import profiles
from core.search.runtime import EmbeddingRuntime, InputTokenizer


def native_tokenizer(direction="right", length=None, multiple=None):
    native = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "rain": 4, "soil": 5, "water": 6},
            unk_token="[UNK]",
        )
    )
    native.pre_tokenizer = pre_tokenizers.Whitespace()
    native.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    native.enable_padding(
        direction=direction, length=length, pad_to_multiple_of=multiple, pad_id=1, pad_token="[PAD]"
    )
    native.enable_truncation(max_length=4)
    return native


@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize("length,multiple", [(None, None), (None, 8), (12, None), (12, 8)])
def test_shared_counts_ignore_padding_without_reconfiguring_inference(direction, length, multiple):
    native = native_tokenizer(direction, length, multiple)
    padding = native.padding
    profile = replace(profiles()["minilm-en"], input_tokens=8)
    counter = InputTokenizer(Path("never-read"), profile, tokenizer=native)
    assert counter.tokenizer is native and native.truncation is None
    assert native.padding == padding
    assert counter.count("rain soil water") == 3
    assert counter.prepare("rain soil water rain soil water") == "rain soil water rain soil water"
    with pytest.raises(ValueError, match="model_input_too_long"):
        counter.prepare("rain soil water rain soil water rain")
    assert native.padding == padding
    batch = native.encode_batch(["rain", "rain soil water"])
    assert len(batch[0].ids) == len(batch[1].ids)
    assert [sum(item.attention_mask) for item in batch] == [3, 5]
    assert native.padding == padding


@pytest.mark.parametrize("query", [False, True])
def test_shared_special_prefix_counts_keep_the_exact_limit(query):
    profile = replace(profiles()["multilingual-e5"], input_tokens=7)
    native = native_tokenizer(multiple=8)
    counter = InputTokenizer(Path("never-read"), profile, tokenizer=native)
    prefix = profile.query_prefix if query else profile.passage_prefix
    prepared = counter.prepare("rain soil water", query=query)
    assert prepared == prefix + "rain soil water"
    assert sum(native.encode(prepared).attention_mask) == 7
    with pytest.raises(ValueError, match="model_input_too_long"):
        counter.prepare("rain soil water rain", query=query)


def fake_backend(monkeypatch, native):
    import fastembed
    from fastembed.common.onnx_model import OnnxModel
    from fastembed.text import custom_text_embedding

    options: dict[str, Any] = {"events": []}

    def load_tokenizer(path):
        options["events"].append("tokenizer")
        options["tokenizer_path"] = path
        return native, {"[PAD]": 1}

    def load_session(backend, *args, **kwargs):
        options["events"].append("onnx")
        options["loader_args"] = args
        options["loader_kwargs"] = kwargs

    class Custom:
        def _load_onnx_model(self, *args, **kwargs):
            load_session(self, *args, **kwargs)
            self.tokenizer, self.special_token_to_id = load_tokenizer(kwargs["model_dir"])

        @classmethod
        def add_extra_session_options(cls, session_options, extra_options):
            options["extra_options"] = extra_options

    class Text:
        EMBEDDINGS_REGISTRY: list[type[Custom]] = []

        @classmethod
        def list_supported_models(cls):
            return [{"model": "corpi/" + profiles()["minilm-en"].identity}]

        def __init__(self, **kwargs):
            options.update(kwargs)
            self.model = self.EMBEDDINGS_REGISTRY[0]()
            self.model._load_onnx_model(
                model_dir=Path(kwargs["specific_model_path"]),
                model_file="unchanged-model.onnx",
                threads=kwargs["threads"],
                providers=kwargs["providers"],
                cuda=kwargs["cuda"],
                device_id=None,
                extra_session_options={"enable_cpu_mem_arena": kwargs["enable_cpu_mem_arena"]},
            )
            session = SimpleNamespace(
                add_session_config_entry=lambda key, value: options.update({key: value})
            )
            self.EMBEDDINGS_REGISTRY[0].add_extra_session_options(
                session, {"enable_cpu_mem_arena": True}
            )

        def embed(self, prepared, **kwargs):
            options["prepared"] = prepared
            options["embed_options"] = kwargs
            return [np.ones(384, dtype=np.float32) for _ in prepared]

    monkeypatch.setattr(fastembed, "TextEmbedding", Text)
    monkeypatch.setattr(custom_text_embedding, "CustomTextEmbedding", Custom)
    monkeypatch.setattr("core.search.runtime.verify_installation", lambda *args: None)
    monkeypatch.setattr("core.search.runtime.version", lambda name: "0.8.1")

    monkeypatch.setattr(OnnxModel, "_load_onnx_model", load_session)
    monkeypatch.setattr("fastembed.common.preprocessor_utils.load_tokenizer", load_tokenizer)
    monkeypatch.setattr(
        "core.search.runtime._load_runtime_tokenizer", load_tokenizer, raising=False
    )
    return options


def test_runtime_reuses_backend_tokenizer_and_keeps_existing_model_options(monkeypatch):
    native = native_tokenizer(multiple=8)
    padding = native.padding
    options = fake_backend(monkeypatch, native)
    runtime = EmbeddingRuntime(Path("verified-by-control"), profiles()["minilm-en"])
    assert runtime.tokenizer.tokenizer is getattr(runtime.model.model, "tokenizer", None) is native
    assert options["events"] == ["tokenizer", "onnx"]
    assert getattr(runtime.model.model, "special_token_to_id", None) == {"[PAD]": 1}
    assert options["loader_args"] == ()
    assert options["loader_kwargs"] == {
        "model_dir": Path("verified-by-control"),
        "model_file": "unchanged-model.onnx",
        "threads": 2,
        "providers": ["CPUExecutionProvider"],
        "cuda": False,
        "device_id": None,
        "extra_session_options": {"enable_cpu_mem_arena": True},
    }
    assert native.padding == padding and native.truncation is None
    assert options["threads"] == 2 and options["providers"] == ["CPUExecutionProvider"]
    assert options["enable_cpu_mem_arena"] is True and options["cuda"] is False
    assert options["local_files_only"] is True
    assert options["session.force_spinning_stop"] == "1"
    assert len(runtime.embed(["rain", "rain soil water"])) == 2
    assert options["prepared"] == ["rain", "rain soil water"]
    assert options["embed_options"] == {"batch_size": 2, "parallel": None}
    assert native.padding == padding


def test_runtime_rejects_missing_backend_tokenizer_instead_of_loading_a_fallback(monkeypatch):
    fake_backend(monkeypatch, None)
    with pytest.raises(ValueError, match="tokenizer_failed"):
        EmbeddingRuntime(Path("verified-by-control"), profiles()["minilm-en"])


def test_runtime_tokenizer_load_failure_stays_safe_and_precedes_session_allocation(monkeypatch):
    options = fake_backend(monkeypatch, native_tokenizer())

    def fail(path):
        raise RuntimeError("private native payload")

    monkeypatch.setattr("core.search.runtime._load_runtime_tokenizer", fail)
    with pytest.raises(ValueError, match="^tokenizer_failed$"):
        EmbeddingRuntime(Path("verified-by-control"), profiles()["minilm-en"])
    assert options["events"] == []


@pytest.mark.parametrize("profile_id", ["bekko-a8m-quint8-avx2", "minilm-en", "multilingual-e5"])
def test_real_runtime_tokenizer_configuration_matches_pinned_loader(request, profile_id):
    import hashlib

    from fastembed.common.preprocessor_utils import load_tokenizer

    from core.search.runtime import _load_runtime_tokenizer

    directory = request.config.getoption("--models-directory")
    if directory is None:
        pytest.skip("explicit installed-model directory required; no downloads")
    path = Path(directory) / profiles()[profile_id].identity
    actual, actual_special = _load_runtime_tokenizer(path)
    expected, expected_special = load_tokenizer(path)
    assert actual_special == expected_special
    assert actual.padding == expected.padding and actual.truncation == expected.truncation
    assert (
        hashlib.sha256(actual.to_str().encode()).digest()
        == hashlib.sha256(expected.to_str().encode()).digest()
    )


def test_large_tokenizer_parse_precedes_numpy_and_fastembed_imports(tmp_path):
    import json
    import subprocess
    import sys

    native_tokenizer().save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 8, "pad_token": "[PAD]"}), encoding="utf-8"
    )
    script = """
import sys
from pathlib import Path
from core.search.runtime import _load_runtime_tokenizer
assert 'numpy' not in sys.modules and 'fastembed' not in sys.modules
seen = []
def observe(frame, event, function):
    if event == 'c_call' and getattr(function, '__name__', '') == 'from_file':
        assert 'numpy' not in sys.modules and 'fastembed' not in sys.modules
        seen.append(True)
sys.setprofile(observe)
_load_runtime_tokenizer(Path(sys.argv[1]))
sys.setprofile(None)
assert seen == [True]
"""
    subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path)], check=True, timeout=30)


def test_shared_encoder_failure_remains_a_safe_error():
    class Broken:
        def no_truncation(self):
            pass

        def encode(self, *args, **kwargs):
            raise RuntimeError("private native payload must not escape")

    counter = InputTokenizer(
        Path("never-read"), profiles()["minilm-en"], tokenizer=cast(Tokenizer, Broken())
    )
    with pytest.raises(ValueError, match="^tokenizer_failed$"):
        counter.count("rain")
