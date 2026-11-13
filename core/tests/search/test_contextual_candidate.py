"""Controls for faithful trained-head restoration and UTF-8 token counting."""

import importlib.util
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from core.search.profiles import profiles


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_contextual_multilingual.py"
    spec = importlib.util.spec_from_file_location("contextual_candidate", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def checkpoint(path):
    weight = np.eye(128, dtype="<f4") * 2
    weight[0, 1] = 0.5
    bias = np.zeros(128, dtype="<f4")
    bias[0] = 0.125
    data = weight.tobytes() + bias.tobytes()
    header = json.dumps(
        {
            "pooler.dense.weight": {
                "dtype": "F32",
                "shape": [128, 128],
                "data_offsets": [0, weight.nbytes],
            },
            "pooler.dense.bias": {
                "dtype": "F32",
                "shape": [128],
                "data_offsets": [weight.nbytes, len(data)],
            },
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def test_trained_pooler_mapping_preserves_transpose_bias_and_tanh(tmp_path):
    path = tmp_path / "reference.safetensors"
    checkpoint(path)
    weight, bias = module().pooler_tensors(path)
    assert not weight.flags.writeable and not bias.flags.writeable
    cls = np.zeros((1, 128), dtype=np.float32)
    cls[0, :2] = [1, 2]
    result = np.tanh(cls @ weight.T + bias)
    assert result[0, :2] == pytest.approx([math.tanh(3.125), math.tanh(4)], abs=1e-7)
    assert np.count_nonzero(result[0, 2:]) == 0


def test_trained_pooler_rejects_short_tensor_payload(tmp_path):
    path = tmp_path / "reference.safetensors"
    checkpoint(path)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="invalid_pooler_tensor"):
        module().pooler_tensors(path)


def test_compiled_utf8_tokenizer_counts_special_tokens_without_truncation(tmp_path):
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "red": 3, "bleu": 4, "مرحبا": 5}
    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer.enable_truncation(3)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    counter = module().UTF8InputTokenizer(
        tmp_path, replace(profiles()["minilm-en"], input_tokens=5)
    )
    assert counter.count("red bleu مرحبا red") == 4
    assert counter.tokenizer.encode("red bleu مرحبا red").ids == [1, 3, 4, 5, 3, 2]
    with pytest.raises(ValueError, match="model_input_too_long"):
        counter.prepare("red bleu مرحبا red")


def test_sentence_mean_includes_real_tokens_and_ignores_poisoned_padding():
    values = np.asarray(
        [[[1, 3], [3, 5], [1e9, 1e9]], [[2, 6], [-1e9, -1e9], [1e9, 1e9]]], dtype=np.float32
    )
    mask = np.asarray([[1, 1, 0], [1, 0, 0]])
    result = module().masked_mean(values, mask)
    assert result.tolist() == [[2, 4], [2, 6]]
    assert result.dtype == np.float32


@pytest.mark.parametrize("pooling", ("cls", "masked-mean"))
def test_contextual_runtime_uses_nonzero_model_padding_id(tmp_path, pooling):
    candidate = module()
    vocab = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "red": 3,
        "blue": 4,
        "unused": 5,
        "other": 6,
        "[PAD]": 7,
    }
    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    profile = replace(profiles()["minilm-en"], dimensions=2, pooling=pooling)
    runtime = candidate.ContextualRuntime.__new__(candidate.ContextualRuntime)
    runtime.profile = profile
    runtime.pad_id = 7
    runtime.tokenizer = candidate.UTF8InputTokenizer(tmp_path, profile)
    runtime.inputs = {"input_ids", "attention_mask"}
    runtime.output = "last_hidden_state"
    runtime.pooler = runtime.dense = None

    class Encoder:
        def run(self, outputs, inputs):
            assert outputs == ["last_hidden_state"]
            assert inputs["input_ids"].tolist() == [[1, 3, 4, 2], [1, 3, 2, 7]]
            assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
            states = np.zeros((2, 4, 2), dtype=np.float32)
            states[:, 0, :] = [1, 0]
            states[:, 1:, :] = [0, 1]
            states[1, 3, :] = [1e9, -1e9]
            return [states]

    runtime.model = Encoder()
    vectors = runtime.embed(["red blue", "red"])
    actual = [struct.unpack("<2f", value) for value in vectors]
    expected = (
        [(1, 0), (1, 0)]
        if pooling == "cls"
        else [
            (1 / math.sqrt(10), 3 / math.sqrt(10)),
            (1 / math.sqrt(5), 2 / math.sqrt(5)),
        ]
    )
    for value, reference in zip(actual, expected, strict=True):
        assert value == pytest.approx(reference, abs=1e-7)
