from fastapi.testclient import TestClient

from core.api.app import create_app
from core.persistence.database import Database, DatabaseConfig
from core.persistence.models import new_id
from core.persistence.store import PersistenceStore
from core.search.install import ModelSetupService
from core.search.switching import NotebookModelService


def test_model_routes_use_existing_bearer_origin_and_host_controls(tmp_path):
    database = Database(DatabaseConfig(tmp_path / "models-data"))
    database.initialize()
    store = PersistenceStore(database)
    app = create_app(
        "models-test-token",
        store=store,
        model_setup=ModelSetupService(store),
        notebook_models=NotebookModelService(store),
    )
    notebook = store.create_notebook(new_id(), "Models")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/models").status_code == 401
        headers = {"Authorization": "Bearer models-test-token"}
        assert (
            client.get(
                "/models", headers={**headers, "Origin": "https://untrusted.example"}
            ).status_code
            == 403
        )
        assert (
            client.get("/models", headers={**headers, "Host": "untrusted.example"}).status_code
            == 403
        )
        models = client.get("/models", headers=headers)
        assert models.status_code == 200
        assert len(models.json()) == 3 and all(
            model["state"] == "missing" for model in models.json()
        )
        selection = client.get(f"/notebooks/{notebook.id}/model", headers=headers)
        assert selection.status_code == 200
        assert selection.json()["active_profile_id"] == "bekko-a8m-quint8-avx2"
        denied = client.post(
            "/models/arbitrary/setup",
            headers=headers,
            json={"request_key": "invalid", "generation": 0},
        )
        assert denied.status_code == 400
