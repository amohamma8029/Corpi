import json

import httpx
from fastapi.testclient import TestClient

from core.api.app import create_app
from core.persistence.database import Database, DatabaseConfig
from core.persistence.store import PersistenceStore
from core.providers.credentials import InMemoryCredentialStore
from core.providers.gemini import GeminiClient
from core.providers.service import ProviderService

TOKEN = "provider-test-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def make_client(tmp_path, handler) -> tuple[TestClient, ProviderService]:
    database = Database(DatabaseConfig(tmp_path / "provider-api"))
    database.initialize()
    store = PersistenceStore(database)
    service = ProviderService(
        store,
        InMemoryCredentialStore(),
        client_factory=lambda key: GeminiClient(
            key,
            transport=httpx.MockTransport(handler),
        ),
    )
    app = create_app(TOKEN, store=store, provider_service=service)
    return TestClient(app, base_url="http://127.0.0.1"), service


def success_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "api-provider-result",
            "output_text": "OK",
            "usage": {"total_input_tokens": 4, "total_output_tokens": 1, "total_tokens": 5},
        },
    )


def test_provider_routes_use_the_authenticated_boundary(tmp_path) -> None:
    client, _ = make_client(tmp_path, success_handler)
    with client:
        assert client.get("/provider/settings").status_code == 401
        assert client.get("/provider/settings", headers=HEADERS).status_code == 200


def test_provider_settings_consent_and_secret_boundary(tmp_path) -> None:
    client, service = make_client(tmp_path, success_handler)
    with client:
        initial = client.get("/provider/settings", headers=HEADERS)
        assert initial.status_code == 200
        assert initial.json()["credential_configured"] is False
        assert initial.json()["free_tier_acknowledged"] is False

        denied = client.put(
            "/provider/settings",
            headers=HEADERS,
            json={
                "generation_model": "gemini-3.8-flash",
                "checker_model": "gemini-3.5-flash-lite",
                "free_tier_acknowledged": False,
                "api_key": "secret-key",
            },
        )
        assert denied.status_code == 400
        assert denied.json()["detail"]["code"] == "free_tier_acknowledgement_required"

        saved = client.put(
            "/provider/settings",
            headers=HEADERS,
            json={
                "generation_model": "gemini-3.8-flash",
                "checker_model": "gemini-3.5-flash-lite",
                "free_tier_acknowledged": True,
                "output_token_allowance": 1024,
                "safety_margin_tokens": 512,
                "api_key": "secret-key",
            },
        )
        assert saved.status_code == 200
        body = saved.json()
        assert body["credential_configured"] is True
        assert "secret-key" not in json.dumps(body)
        assert "api_key" not in body

        tested = client.post("/provider/settings/test", headers=HEADERS)
        assert tested.status_code == 200
        assert tested.json()["ok"] is True
        assert tested.json()["model"] == "gemini-3.8-flash"

        usage = client.get("/provider/usage", headers=HEADERS)
        assert usage.status_code == 200
        assert usage.json()[0]["call_type"] == "settings_test"
        assert usage.json()[0]["outcome"] == "completed"

        cleared = client.delete("/provider/credential", headers=HEADERS)
        assert cleared.status_code == 200
        assert cleared.json()["credential_configured"] is False
        assert service.usage()[0].call_type == "settings_test"


def test_provider_models_route_returns_the_provider_catalog(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-3.8-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-3.5-flash-lite",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ]
            },
        )

    client, _ = make_client(tmp_path, handler)
    with client:
        saved = client.put(
            "/provider/settings",
            headers=HEADERS,
            json={
                "generation_model": "gemini-3.8-flash",
                "checker_model": "gemini-3.5-flash-lite",
                "free_tier_acknowledged": True,
                "api_key": "secret-key",
            },
        )
        assert saved.status_code == 200
        response = client.get("/provider/models", headers=HEADERS)
        assert response.status_code == 200
        assert response.json()["discovered"] is True
        assert response.json()["generation_models"] == [
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
        ]


def test_provider_rejects_an_unsupported_model(tmp_path) -> None:
    client, _ = make_client(tmp_path, success_handler)
    with client:
        response = client.put(
            "/provider/settings",
            headers=HEADERS,
            json={
                "generation_model": "not-a-gemini-model",
                "checker_model": "gemini-3.5-flash-lite",
                "free_tier_acknowledged": True,
                "api_key": "secret-key",
            },
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "unsupported_model"
