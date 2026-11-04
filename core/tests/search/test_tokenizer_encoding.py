"""UTF-8 model metadata must not depend on the Windows process code page."""

import builtins
import io
import json
from pathlib import Path

import pytest
from test_runtime_tokenizer_reuse import native_tokenizer

from core.search.profiles import profiles
from core.search.runtime import InputTokenizer, _load_runtime_tokenizer


@pytest.mark.parametrize("loader", ["planning", "runtime"])
@pytest.mark.parametrize("unicode_file", [None, "config", "tokenizer_config", "special_tokens_map"])
def test_model_json_is_utf8_even_when_default_text_encoding_is_cp1252(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: str, unicode_file: str | None
) -> None:
    native_tokenizer().save(str(tmp_path / "tokenizer.json"))
    metadata: dict[str, dict[str, object]] = {
        "config": {"pad_token_id": 1},
        "tokenizer_config": {"model_max_length": 8, "pad_token": "[PAD]"},
        "special_tokens_map": {},
    }
    if unicode_file == "special_tokens_map":
        metadata[unicode_file] = {"additional_special_tokens": ["ف"]}
    elif unicode_file is not None:
        metadata[unicode_file]["metadata_note"] = "ف"
    for name, value in metadata.items():
        (tmp_path / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )
    normal_open = builtins.open

    def legacy_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        selected = encoding if encoding is not None or "b" in mode else "cp1252"
        return normal_open(file, mode, buffering, selected, *args, **kwargs)

    with monkeypatch.context() as legacy:
        legacy.setattr(builtins, "open", legacy_open)
        legacy.setattr(io, "open", legacy_open)
        if loader == "planning":
            counter = InputTokenizer(tmp_path, profiles()["minilm-en"])
            assert counter.count("rain soil") == 2
            assert counter.prepare("rain soil") == "rain soil"
            assert counter.tokenizer.padding is None and counter.tokenizer.truncation is None
        else:
            tokenizer, special = _load_runtime_tokenizer(tmp_path)
            assert tokenizer.truncation is not None and tokenizer.truncation["max_length"] == 8
            assert tokenizer.padding is not None and tokenizer.padding["pad_id"] == 1
            if unicode_file == "special_tokens_map":
                assert special["ف"] == tokenizer.token_to_id("ف")
