import json
import os
import platform
import shutil
import sqlite3
import time
from importlib.metadata import version
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.api.app import create_app
from core.persistence.store import PersistenceStore
from core.search.handoff import SUPPORTED_PROFILE_IDS
from core.search.install import ModelSetupService
from core.search.profiles import installation_path, profiles, verify_installation
from core.search.runtime import EmbeddingRuntime
from core.search.supervisor import stop_owned_process_tree
from core.search.switching import NotebookModelService


@pytest.mark.skipif(
    os.environ.get("CORPI_RUN_MODEL_DOWNLOAD") != "1",
    reason="explicit live model download opt-in required",
)
@pytest.mark.parametrize("profile_id", SUPPORTED_PROFILE_IDS)
def test_real_authenticated_setup_downloads_verifies_and_supports_local_inference(
    store: PersistenceStore,
    request: pytest.FixtureRequest,
    profile_id: str,
    monkeypatch: pytest.MonkeyPatch,
):
    reaped = []

    def stop_and_observe(process):
        stop_owned_process_tree(process)
        assert process.poll() is not None
        reaped.append(process.pid)

    monkeypatch.setattr("core.search.install.stop_owned_process_tree", stop_and_observe)
    setup = ModelSetupService(store)
    app = create_app(
        "live-setup-test-token",
        store=store,
        model_setup=setup,
        notebook_models=NotebookModelService(store),
    )
    headers = {"Authorization": "Bearer live-setup-test-token"}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        available = client.get("/models", headers=headers)
        available.raise_for_status()
        model = next(item for item in available.json() if item["profile_id"] == profile_id)
        response = client.post(
            f"/models/{profile_id}/setup",
            headers=headers,
            json={"request_key": "real-download", "generation": model["generation"]},
        )
        response.raise_for_status()
        deadline = time.monotonic() + 620
        state = None
        while time.monotonic() < deadline:
            state = setup.state(profile_id)
            if state["state"] != "downloading":
                break
            time.sleep(0.2)
        assert state is not None and state["state"] == "installed", state
        profile = profiles()[profile_id]
        path = installation_path(setup.data_dir, profile)
        verify_installation(path, profile)
        runtime = EmbeddingRuntime(path, profile)
        assert len(runtime.embed(["Rainfall and soil moisture were measured."], query=True)) == 1
        export = request.config.getoption("--model-setup-export-directory")
        if export is not None:
            destination = Path(export)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(path, destination / profile.identity)
            verify_installation(destination / profile.identity, profile)
            evidence = {
                "profile": profile.id,
                "manifest": profile.identity,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "sqlite": sqlite3.sqlite_version,
                "sqlite_vec": version("sqlite-vec"),
                "fastembed": version("fastembed"),
                "onnxruntime": version("onnxruntime"),
                "received_bytes": state["received_bytes"],
                "setup_seconds": state["spent_seconds"],
                "transfer_attempts": state["transfer_attempts"],
                "files": [
                    {"path": asset.path, "bytes": asset.bytes, "sha256": asset.sha256}
                    for asset in profile.files
                ],
            }
            (destination / f"{profile.id}-setup.json").write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
    assert reaped and (setup._thread is None or not setup._thread.is_alive())
    with store.database.connect() as connection:
        assert connection.execute(
            "SELECT state,cleanup_pending FROM model_setup_attempts WHERE manifest_id=?",
            (profile.identity,),
        ).fetchone()[:] == ("installed", 0)
