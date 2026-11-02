"""The timing inspector must reject incomplete/misattributed acceptance evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

from core.search.profiles import profiles


@pytest.fixture
def inspector():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_desktop_readiness.py"
    spec = importlib.util.spec_from_file_location("desktop_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(inspector, monkeypatch):
    source = {
        "source_id": "source",
        "version": {"id": "version", "verified_original_sha256": "original"},
        "extractions": [{"pages": [{}] * 16, "eligible_utf8_bytes": 1234}],
        "readiness": [
            {
                "source_version_id": "version",
                "index_version_id": "index",
                "published_by_job_id": "job",
            }
        ],
        "jobs": [
            {
                "id": "job",
                "kind": "pdf-search",
                "status": "completed",
                "steps": [
                    {"step_key": key}
                    for key in ("plan-search", "embed-search", "build-search", "publish-search")
                ],
            }
        ],
        "generations": [
            {
                "index_version_id": "index",
                "manifest_id": profiles()["minilm-en"].identity,
                "chunks": 10,
                "search_words": 10,
                "search_vectors": 10,
                "complete_batches": 1,
                "incomplete_batches": 0,
                "spent_seconds": 10,
                "settings": {
                    "threads": 2,
                    "batch_size": 32,
                    "total_seconds": 600,
                    "batch_seconds": 120,
                    "build_seconds": 300,
                    "memory_bytes": 1073741824,
                    "max_chunks": 10000,
                    "max_text_bytes": 20_000_000,
                },
            }
        ],
    }
    monkeypatch.setattr(inspector, "inspect_source", lambda _, root: copy.deepcopy(source))
    result = {
        "choice": {"pdf": "bert", "profile": "minilm-en", "repetitions": 1, "cancel": False},
        "observations": [
            {
                "sourceId": "source",
                "readyMs": 12000,
                "cancellation": False,
                "fixture": "bert",
                "sha256": "original",
                "pages": 16,
                "profile": "minilm-en",
                "repetition": 1,
            }
        ],
    }
    return result, {"id": "bert", "sha256": "original", "pages": 16}, source


def test_inspector_accepts_complete_original_and_native_rows(inspector, evidence):
    result, fixture, _ = evidence
    assert len(inspector.inspect_result(result, fixture, Path.cwd())) == 1


@pytest.mark.parametrize(
    "damage", ["no_observation", "no_source", "short_repetitions", "wrong_fixture"]
)
def test_inspector_rejects_missing_or_misattributed_ui_evidence(inspector, evidence, damage):
    result, fixture, _ = evidence
    if damage == "no_observation":
        result["observations"] = []
    elif damage == "no_source":
        result["observations"][0]["sourceId"] = None
    elif damage == "short_repetitions":
        result["choice"]["repetitions"] = 2
    else:
        result["observations"][0]["sha256"] = "different"
    with pytest.raises(AssertionError):
        inspector.inspect_result(result, fixture, Path.cwd())


@pytest.mark.parametrize("damage", ["original", "native_rows", "memory_limit", "readiness_version"])
def test_inspector_rejects_wrong_original_rows_limits_or_version(inspector, evidence, damage):
    result, fixture, source = evidence
    if damage == "original":
        source["version"]["verified_original_sha256"] = "different"
    elif damage == "native_rows":
        source["generations"][0]["search_vectors"] = 9
    elif damage == "memory_limit":
        source["generations"][0]["settings"]["memory_bytes"] *= 2
    else:
        source["readiness"][0]["source_version_id"] = "different"
    with pytest.raises(AssertionError):
        inspector.inspect_result(result, fixture, Path.cwd())


def test_cancellation_requires_an_actual_cancelled_search_job(inspector, evidence):
    result, fixture, source = evidence
    result["choice"]["cancel"] = True
    result["observations"][0].update({"cancellation": True, "readyMs": None})
    source["readiness"] = []
    source["jobs"] = [{"kind": "pdf-extract", "status": "completed"}]
    with pytest.raises(AssertionError):
        inspector.inspect_result(result, fixture, Path.cwd())


@pytest.mark.parametrize("state", ["pending", "declined", "confirmed"])
def test_partial_ready_requires_real_confirmed_review(inspector, evidence, state):
    result, fixture, source = evidence
    source["extractions"][0].update({"outcome": "partial", "reviews": [{"state": state}]})
    if state == "confirmed":
        assert inspector.inspect_result(result, fixture, Path.cwd())
    else:
        with pytest.raises(AssertionError):
            inspector.inspect_result(result, fixture, Path.cwd())


@pytest.mark.parametrize(
    "profile_id,batch_size",
    [
        ("minilm-en", 32),
        ("multilingual-e5", 8),
        ("bekko-a8m-quint8-avx2", 1),
    ],
)
def test_current_recipe_checks_exact_saved_profile_and_policy(
    inspector, evidence, profile_id, batch_size
):
    result, fixture, source = evidence
    result["choice"].update(profile=profile_id, preparation_recipe=inspector.CURRENT_RECIPE)
    result["observations"][0]["profile"] = profile_id
    generation = source["generations"][0]
    generation["manifest_id"] = profiles()[profile_id].identity
    generation["settings"].update(
        total_seconds=1200, batch_size=batch_size, budget_policy="batch-reservation-v1"
    )
    generation["spent_seconds"] = 900
    assert inspector.inspect_result(result, fixture, Path.cwd())


@pytest.mark.parametrize(
    "damage",
    [
        "recipe",
        "budget_version",
        "total_seconds",
        "charge",
        "batch",
        "manifest",
        "index",
        "incomplete",
        "no_batches",
        "chunks_limit",
        "text_limit",
        "not_terminal",
        "false_cancel",
        "publisher",
        "failed_publisher",
        "missing_checkpoint",
    ],
)
def test_current_recipe_rejects_stale_or_incomplete_evidence(inspector, evidence, damage):
    result, fixture, source = evidence
    result["choice"]["preparation_recipe"] = inspector.CURRENT_RECIPE
    generation = source["generations"][0]
    generation["settings"].update(total_seconds=1200, budget_policy="batch-reservation-v1")
    if damage == "recipe":
        result["choice"]["preparation_recipe"] = "unknown"
    elif damage == "budget_version":
        generation["settings"]["budget_policy"] = "unknown"
    elif damage == "total_seconds":
        generation["settings"]["total_seconds"] = 600
    elif damage == "charge":
        generation["spent_seconds"] = 1201
    elif damage == "batch":
        generation["settings"]["batch_size"] = 8
    elif damage == "manifest":
        generation["manifest_id"] = profiles()["multilingual-e5"].identity
    elif damage == "index":
        source["readiness"][0]["index_version_id"] = "other"
    elif damage == "incomplete":
        generation["incomplete_batches"] = 1
    elif damage == "no_batches":
        generation["complete_batches"] = 0
    elif damage == "chunks_limit":
        generation["settings"]["max_chunks"] = 10001
    elif damage == "text_limit":
        generation["settings"]["max_text_bytes"] = 20_000_001
    elif damage == "not_terminal":
        result["observations"][0]["readyMs"] = None
    elif damage == "publisher":
        source["readiness"][0]["published_by_job_id"] = "other"
    elif damage == "failed_publisher":
        source["jobs"][0]["status"] = "failed"
    elif damage == "missing_checkpoint":
        source["jobs"][0]["steps"].pop()
    else:
        result["observations"][0]["cancellation"] = True
    with pytest.raises(AssertionError):
        inspector.inspect_result(result, fixture, Path.cwd())


def test_legacy_receipt_cannot_gain_new_allowance_or_policy(inspector, evidence):
    result, fixture, source = evidence
    source["generations"][0]["settings"]["budget_policy"] = "batch-reservation-v1"
    with pytest.raises(AssertionError):
        inspector.inspect_result(result, fixture, Path.cwd())


def test_native_binary_requires_a_passing_identity_receipt(inspector, tmp_path):
    binary = tmp_path / "test-binary"
    binary.write_bytes(b"retained public control")
    digest = inspector.sha(binary)
    inspector.verify_binary(binary, {"passed": True, "binary_sha256": digest})
    inspector.verify_binary(binary, {"status": "passed", "testBinarySha256": digest})
    for receipt in (
        {"passed": False, "binary_sha256": digest},
        {"passed": True, "binary_sha256": "wrong"},
        {"passed": True},
        {"passed": False, "status": "passed", "binary_sha256": digest},
        {"passed": True, "status": "failed", "binary_sha256": digest},
    ):
        with pytest.raises(AssertionError):
            inspector.verify_binary(binary, receipt)
