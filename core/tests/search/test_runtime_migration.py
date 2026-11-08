"""The installed library changes; saved embedding contracts and evidence do not."""

import json
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from test_controls import queued
from test_query import VECTOR, install
from test_runtime_tokenizer_reuse import fake_backend, native_tokenizer
from test_storage import seed

from core.persistence.database import Database
from core.persistence.models import JSONValue
from core.persistence.store import PersistenceStore
from core.search.controls import PreparationControl
from core.search.preparation import SearchLimits, effective_settings
from core.search.profiles import profiles
from core.search.query import QueryService
from core.search.runtime import EmbeddingRuntime, _legacy_normalized_embeddings, vector_bytes


def legacy_output(embeddings, mask, pooling):
    # Independent reference from FastEmbed 0.8.1 common/utils.py and custom text
    # postprocessing. Do not replace with the new library's pooling/normalization.
    if pooling == "cls":
        pooled = embeddings[:, 0]
    else:
        expanded = np.tile(np.expand_dims(mask, -1).astype(np.int64), (1, 1, embeddings.shape[-1]))
        pooled = np.sum(embeddings * expanded, axis=1) / np.maximum(np.sum(expanded, axis=1), 1e-9)
    return pooled / np.maximum(np.linalg.norm(pooled, ord=2, axis=1, keepdims=True), 1e-12)


@pytest.mark.parametrize("pooling", ["cls", "mean", "masked-mean"])
@pytest.mark.parametrize("left_padding", [False, True])
def test_legacy_pooling_preserves_exact_values_and_final_bytes(pooling, left_padding):
    from fastembed.common.utils import mean_pooling, normalize

    embeddings = np.random.default_rng(193).standard_normal((3, 17, 384)).astype(np.float32)
    mask = np.array([[1] * 17, [1] * 9 + [0] * 8, [1] + [0] * 16], dtype=np.int64)
    if left_padding:
        mask = mask[:, ::-1]
    expected = legacy_output(embeddings, mask, pooling)
    actual = _legacy_normalized_embeddings(embeddings, mask, pooling)
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    encoded = [vector_bytes(row.tolist(), 384) for row in actual]
    assert encoded == [vector_bytes(row.tolist(), 384) for row in expected]
    if pooling != "cls":
        upstream = normalize(mean_pooling(embeddings, mask)).astype(embeddings.dtype)
        # This corpus need not cross a final float32 rounding boundary. It must
        # still distinguish the early cast: the legacy intermediate is float64.
        assert actual.dtype == np.float64 and upstream.dtype == np.float32


def test_production_adapter_uses_legacy_postprocessing(monkeypatch):
    from fastembed.common.onnx_model import OnnxOutputContext

    fake_backend(monkeypatch, native_tokenizer())
    runtime = EmbeddingRuntime(Path("verified-by-control"), profiles()["minilm-en"])
    embeddings = np.random.default_rng(193).standard_normal((2, 17, 384)).astype(np.float32)
    mask = np.ones((2, 17), dtype=np.int64)
    output = OnnxOutputContext(model_output=embeddings, attention_mask=mask)
    actual = cast(Any, runtime.model.model)._post_process_onnx_output(output)
    assert actual.tobytes() == legacy_output(embeddings, mask, "mean").tobytes()


@pytest.mark.parametrize("installed", ["0.8.1", "0.9.0", "0.9.2", "1.0.0"])
def test_unreviewed_installed_runtime_is_rejected_before_tokenizer_or_backend(
    monkeypatch, installed
):
    options = fake_backend(monkeypatch, native_tokenizer())
    monkeypatch.setattr("core.search.runtime.version", lambda name: installed)
    with pytest.raises(ValueError, match="^runtime_mismatch$"):
        EmbeddingRuntime(Path("verified-by-control"), profiles()["minilm-en"])
    assert options["events"] == []


@pytest.mark.parametrize("changes", [{"runtime": "fastembed==0.9.1"}, {"pooling": "last"}])
def test_unreviewed_embedding_contract_is_not_silently_mapped(monkeypatch, changes):
    options = fake_backend(monkeypatch, native_tokenizer())
    with pytest.raises(ValueError, match="^runtime_mismatch$"):
        EmbeddingRuntime(Path("verified-by-control"), replace(profiles()["minilm-en"], **changes))
    assert options["events"] == []


def test_original_model_and_installation_identities_remain_fixed():
    assert {key: profile.identity for key, profile in profiles().items()} == {
        "bge-small-en": "991076419d7ade40b484edf9259d4865066b7949d6863374408f920bbaaff993",
        "multilingual-minilm": "1e488ec46d334997fd52c1b8add45d79b8a15f4ac8d2e5b0945168a48375b66e",
        "multilingual-e5": "f9b59b5565a989f06213b338a6bd295e07abffea2f7ec1a9dda97a674a9a53b7",
        "minilm-en": "4660ee9b8e44b64081521d8633a6da38e597968170d22140e92a689d83d633d3",
        "bekko-a8m-quint8-avx2": "6d9ba5719ef4a550b571676dfbc5c33d9d346650d978e17a8d4cfd03f5da77f3",
    }
    settings = effective_settings(profiles()["minilm-en"], SearchLimits(), None)
    assert settings["fastembed"] == version("fastembed") == "0.9.1"
    assert settings["onnxruntime"] == "1.30.0" and settings["tokenizers"] == "0.23.2"


def test_legacy_tokenizer_does_not_adopt_new_config_direction_precedence(tmp_path):
    from fastembed.common.preprocessor_utils import load_tokenizer

    from core.search.runtime import _load_runtime_tokenizer

    native_tokenizer().save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "model_max_length": 8,
                "pad_token": "[PAD]",
                "padding_side": "left",
                "truncation_side": "left",
            }
        ),
        encoding="utf-8",
    )
    legacy, _ = _load_runtime_tokenizer(tmp_path)
    current, _ = load_tokenizer(tmp_path)
    assert legacy.padding is not None and legacy.truncation is not None
    assert current.padding is not None and current.truncation is not None
    assert legacy.padding["direction"] == legacy.truncation["direction"] == "right"
    assert current.padding["direction"] == current.truncation["direction"] == "left"


def test_old_ready_search_survives_database_reopen_without_relabelling(store):
    settings: dict[str, JSONValue] = {
        "fastembed": "0.8.1",
        "onnxruntime": "1.30.0",
        "tokenizers": "0.23.2",
    }
    notebook, source, _, index, _, _ = seed(
        store, "rainfall", VECTOR, profile_id="minilm-en", runtime_settings=settings
    )
    install(store, "minilm-en")
    with store.database.connect() as connection:
        saved = tuple(
            connection.execute(
                "SELECT * FROM search_generations WHERE index_version_id=?", (index,)
            ).fetchone()
        )
        vector = connection.execute("SELECT embedding FROM search_vectors").fetchone()[0]

    class Session:
        def __init__(self, path, profile, **options):
            assert path.name == profile.identity == profiles()["minilm-en"].identity
            self.profile = profile

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def embed(self, texts, **options):
            assert texts == ["rainfall"] and options == {"query": True, "timeout": 120}
            return [vector_bytes(VECTOR, 384)]

    reopened = PersistenceStore(Database(store.database.config))
    service = QueryService(reopened, Path("unused"), session_factory=Session)
    for mode in ("words", "meaning"):
        hit = service.search(notebook, "rainfall", mode)["results"][0]
        assert hit["source_id"] == source and hit["index_version_id"] == index
        assert hit["text"] == "rainfall" and hit["citations"][0]["page_number"] == 1
    with reopened.database.connect() as connection:
        assert (
            tuple(
                connection.execute(
                    "SELECT * FROM search_generations WHERE index_version_id=?", (index,)
                ).fetchone()
            )
            == saved
        )
        assert connection.execute("SELECT embedding FROM search_vectors").fetchone()[0] == vector


def test_old_resume_keeps_saved_runtime_policy_and_spent_allowance(store, monkeypatch):
    from core.search import preparation

    original = preparation.version
    monkeypatch.setattr(
        preparation, "version", lambda name: "0.8.1" if name == "fastembed" else original(name)
    )
    _, source, job = queued(store)
    with store._transaction() as connection:
        connection.execute(
            "UPDATE search_job_inputs SET spent_seconds=43.25 WHERE job_id=?", (job,)
        )
        before = dict(
            connection.execute("SELECT * FROM search_job_inputs WHERE job_id=?", (job,)).fetchone()
        )
    monkeypatch.setattr(preparation, "version", original)
    control = PreparationControl(store)
    control.cancel(source, job)
    from core.search.handoff import SearchCoordinator

    SearchCoordinator(store, Path("unused")).reconcile()
    resumed = control.resume(source, job, "after-runtime-upgrade")
    with store.database.connect() as connection:
        after = dict(
            connection.execute(
                "SELECT * FROM search_job_inputs WHERE job_id=?", (resumed,)
            ).fetchone()
        )
    assert {key: value for key, value in before.items() if key != "job_id"} == {
        key: value for key, value in after.items() if key != "job_id"
    }
    assert json.loads(after["settings_json"])["fastembed"] == "0.8.1"
    assert version("fastembed") == "0.9.1"
    # Enqueue preservation is proven here, not truthful execution provenance for
    # future batches. That requires an explicit durable execution receipt.
