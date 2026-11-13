"""Independent small-table controls for the experimental static-model benchmark."""

import hashlib
import importlib.util
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import Unigram, WordPiece
from tokenizers.pre_tokenizers import Whitespace

from core.search.profiles import ModelFile, profiles


def candidate_module():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_static_multilingual.py"
    spec = importlib.util.spec_from_file_location("static_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, kind="wordpiece"):
    header = json.dumps(
        {"embeddings": {"dtype": "F32", "shape": [4, 2], "data_offsets": [0, 32]}}
    ).encode()
    payload = struct.pack("<8f", 7, 7, 1, 0, 0, 1, 0, 0)
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    model = (
        WordPiece({"[UNK]": 0, "red": 1, "blue": 2, "zero": 3}, unk_token="[UNK]")
        if kind == "wordpiece"
        else Unigram([("[UNK]", 0.0), ("red", -1.0), ("blue", -1.0), ("zero", -1.0)], unk_id=0)
    )
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    files = tuple(
        ModelFile(
            name,
            (tmp_path / name).stat().st_size,
            hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
            "https://example.org/" + name,
        )
        for name in ("model.safetensors", "tokenizer.json")
    )
    return replace(
        profiles()["minilm-en"],
        dimensions=2,
        input_tokens=4,
        model_file="model.safetensors",
        files=files,
    )


@pytest.mark.parametrize("kind", ("wordpiece", "unigram"))
def test_static_mean_weights_every_token_and_excludes_unknown_without_truncating(tmp_path, kind):
    module = candidate_module()
    runtime = module.StaticRuntime(tmp_path, fixture(tmp_path, kind))
    values = struct.unpack("<2f", runtime.embed(["red red blue missing"])[0])
    assert values == pytest.approx((2 / math.sqrt(5), 1 / math.sqrt(5)), abs=1e-7)
    assert runtime.embed(["red blue"], query=True) == runtime.embed(["red blue"])
    assert not runtime.embeddings.flags.writeable
    with pytest.raises(ValueError, match="model_input_too_long"):
        runtime.embed(["red red red red red"])
    with pytest.raises(ValueError, match="invalid_embedding_input"):
        runtime.embed(["missing"])
    with pytest.raises(ValueError, match="invalid_vector"):
        runtime.embed(["zero"])


def test_static_table_rejects_truncated_payload_and_dimension_mismatch(tmp_path):
    module = candidate_module()
    fixture(tmp_path)
    target = tmp_path / "model.safetensors"
    with pytest.raises(ValueError, match="invalid_static_table"):
        module.map_embeddings(target, 3)
    target.write_bytes(target.read_bytes()[:-4])
    with pytest.raises(ValueError, match="invalid_static_table"):
        module.map_embeddings(target, 2)


def test_retrieval_static_mean_includes_unknown_and_accepts_trained_weight_key(tmp_path):
    module = candidate_module()
    profile = replace(fixture(tmp_path), pooling="static-mean-include-unk")
    runtime = module.StaticRuntime(tmp_path, profile)
    # UNK contributes [7,7]; excluding it would incorrectly return [1,0].
    values = struct.unpack("<2f", runtime.embed(["red missing"])[0])
    assert values == pytest.approx((8 / math.sqrt(113), 7 / math.sqrt(113)), abs=1e-7)
    assert runtime.embed(["missing"])
    assert not runtime.embeddings.flags.writeable
    with pytest.raises(ValueError, match="model_input_too_long"):
        runtime.embed(["red red red red red"])
    target = tmp_path / "trained-key.safetensors"
    header = json.dumps(
        {"embedding.weight": {"dtype": "F32", "shape": [4, 2], "data_offsets": [0, 32]}}
    ).encode()
    target.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<8f", *range(8)))
    assert module.map_embeddings(target, 2, include_unk=True).tolist() == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]
    with pytest.raises(ValueError, match="invalid_static_table"):
        module.map_embeddings(target, 2)
