import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from core.search.profiles import ModelFile, profiles, verify_installation
from core.search.runtime import EmbeddingRuntime, InputTokenizer, masked_mean_pool


def test_installation_is_a_hash_proof_not_an_existing_directory(tmp_path: Path) -> None:
    profile = replace(
        profiles()["bge-small-en"],
        files=(ModelFile("asset.bin", 4, hashlib.sha256(b"good").hexdigest(), "unused"),),
    )
    with pytest.raises(ValueError, match="model_corrupt"):
        verify_installation(tmp_path, profile)
    (tmp_path / "asset.bin").write_bytes(b"evil")
    with pytest.raises(ValueError, match="model_corrupt"):
        verify_installation(tmp_path, profile)
    (tmp_path / "asset.bin").write_bytes(b"good")
    verify_installation(tmp_path, profile)


def test_token_count_reuse_preserves_special_token_checks_and_remains_bounded(monkeypatch):
    from types import SimpleNamespace

    class Tokenizer:
        def __init__(self):
            self.calls = 0

        def no_truncation(self):
            pass

        def no_padding(self):
            pass

        def encode(self, text, add_special_tokens=True):
            self.calls += 1
            return SimpleNamespace(ids=range(len(text) + (2 if add_special_tokens else 0)))

    native = Tokenizer()
    monkeypatch.setattr("core.search.runtime._load_runtime_tokenizer", lambda _: (native, {}))
    profile = replace(profiles()["minilm-en"], input_tokens=6)
    tokenizer = InputTokenizer(Path("unused"), profile)
    assert tokenizer.count("rain") == tokenizer.count("rain") == 4
    assert tokenizer.prepare("rain") == "rain"
    assert native.calls == 2  # Plain and special-token counts are separate.
    for _ in range(2):
        with pytest.raises(ValueError, match="model_input_too_long"):
            tokenizer.prepare("rains")
    for number in range(200):
        tokenizer.count(str(number))
    assert len(tokenizer._counts) == 128
    before = native.calls
    tokenizer.count("x" * 16_385)
    tokenizer.count("x" * 16_385)
    assert native.calls == before + 2


def test_bounded_mean_pool_preserves_masked_padding_and_float64_accumulation():
    import numpy as np
    from fastembed.common.utils import mean_pooling

    values = np.random.default_rng(42).normal(size=(4, 17, 384)).astype(np.float32)
    mask = np.ones((4, 17), dtype=np.int64)
    mask[1, 7:] = 0
    mask[2, :5] = 0
    mask[3, :] = 0
    actual = masked_mean_pool(values, mask)
    expected = mean_pooling(values, mask)
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert not actual[3].any()


@pytest.mark.parametrize("profile_id", list(profiles()))
def test_real_installed_inference_is_offline_and_does_not_truncate(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    directory = request.config.getoption("--models-directory")
    if directory is None:
        pytest.skip("explicit installed-model directory required; tests never download weights")
    profile = profiles()[profile_id]

    def deny(*args, **kwargs):
        raise AssertionError("inference attempted network access")

    import socket

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    runtime = EmbeddingRuntime(Path(directory) / profile.identity, profile)
    documents = [
        "The experiment measures rainfall and soil moisture.",
        "L'expérience mesure les précipitations et l'humidité du sol.",
        "تقيس التجربة هطول الأمطار ورطوبة التربة.",
    ]
    vectors = (
        [runtime.embed([text])[0] for text in documents]
        if profile_id == "bekko-a8m-quint8-avx2"
        else runtime.embed(documents)
    )
    assert len(vectors) == 3 and all(len(vector) == 384 * 4 for vector in vectors)
    assert len(runtime.embed(["What does the experiment measure?"], query=True)) == 1
    with pytest.raises(ValueError, match="model_input_too_long"):
        runtime.embed([documents[0] * (profile.input_tokens + 1)])
