"""Backend-independent privacy/ownership controls for the temporary experiment."""

import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_openvino_granite.py"
    spec = importlib.util.spec_from_file_location("openvino_benchmark_controls", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def owned_root(tmp_path):
    root = tmp_path / "corpi-granite-openvino-experiment"
    root.mkdir()
    (root / "ownership.json").write_text(
        json.dumps({"task": "m1-6-local-search"}), encoding="utf-8"
    )
    consent = root / "localappdata/Intel Corporation/openvino_telemetry"
    consent.parent.mkdir(parents=True)
    consent.write_text("0", encoding="utf-8")
    return root, consent


def restore_process_state(monkeypatch):
    for key in (
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_OFFLINE",
    ):
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)


def test_owned_declined_consent_and_python_network_denial(tmp_path, monkeypatch):
    restore_process_state(monkeypatch)
    root, consent = owned_root(tmp_path)
    assert module().isolate(root, "ort") == consent
    assert Path(os.environ["LOCALAPPDATA"]).is_relative_to(root)
    assert Path(os.environ["USERPROFILE"]).is_relative_to(root)
    assert consent.read_text() == "0"
    with socket.socket() as connection:
        with pytest.raises(OSError, match="network connections denied"):
            connection.connect(("127.0.0.1", 9))
        with pytest.raises(OSError, match="network connections denied"):
            connection.connect_ex(("127.0.0.1", 9))


def test_backend_refuses_accepted_telemetry_before_import(tmp_path):
    root, consent = owned_root(tmp_path)
    consent.write_text("1", encoding="utf-8")
    with pytest.raises(ValueError, match="not declined"):
        module().isolate(root, "openvino")


def test_backend_rejects_overlay_outside_owned_root(tmp_path, monkeypatch):
    restore_process_state(monkeypatch)
    root, _ = owned_root(tmp_path)
    (root / "environment.json").write_text(
        json.dumps({"overlay_paths": [str(tmp_path)]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="escaped ownership"):
        module().isolate(root, "openvino")


def test_bekko_root_requires_matching_experiment_marker(tmp_path, monkeypatch):
    restore_process_state(monkeypatch)
    root, _ = owned_root(tmp_path)
    target = root.with_name("corpi-bekko-openvino-experiment")
    root.rename(target)
    candidate = module()
    with pytest.raises(ValueError, match="unowned"):
        candidate.owned_root(target)
    (target / "ownership.json").write_text(
        json.dumps({"task": "m1-6-local-search", "experiment": "bekko-openvino"}),
        encoding="utf-8",
    )
    assert candidate.owned_root(target) == target.resolve()
    assert candidate.isolate(target, "ort").read_text().strip() == "0"
