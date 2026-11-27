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
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "OK"}],
                }
            ],
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
        test_body = tested.json()
        assert test_body["ok"] is True
        assert test_body["model"] == "gemini-3.8-flash"
        assert test_body["provider_request_id"] == "api-provider-result"
        assert test_body["request_id"] != test_body["provider_request_id"]

        usage = client.get("/provider/usage", headers=HEADERS)
        assert usage.status_code == 200
        assert usage.json()[0]["call_type"] == "settings_test"
        assert usage.json()[0]["outcome"] == "completed"
        assert usage.json()[0]["request_id"] == test_body["request_id"]
        assert usage.json()[0]["provider_request_id"] == "api-provider-result"
        assert usage.json()[0]["parent_request_id"] is None

        cleared = client.delete("/provider/credential", headers=HEADERS)
        assert cleared.status_code == 200
        assert cleared.json()["credential_configured"] is False
        assert service.usage()[0].call_type == "settings_test"


def test_provider_settings_validation_never_echoes_the_submitted_key(tmp_path) -> None:
    client, _ = make_client(tmp_path, success_handler)
    with client:
        invalid_key = client.put(
            "/provider/settings",
            headers=HEADERS,
            json={
                "generation_model": "gemini-3.8-flash",
                "checker_model": "gemini-3.5-flash-lite",
                "free_tier_acknowledged": True,
                "api_key": ["secret-submitted-key"],
            },
        )
        assert invalid_key.status_code == 422
        assert "secret-submitted-key" not in invalid_key.text
        assert all(set(item) == {"loc", "msg", "type"} for item in invalid_key.json()["detail"])

        invalid_setting = client.put(
            "/provider/settings",
            headers=HEADERS,
            json={
                "generation_model": "gemini-3.8-flash",
                "checker_model": "gemini-3.5-flash-lite",
                "free_tier_acknowledged": True,
                "output_token_allowance": 0,
                "api_key": "secret-submitted-key",
            },
        )
        assert invalid_setting.status_code == 422
        assert "secret-submitted-key" not in invalid_setting.text


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
            "gemini-3.6-flash",
            "gemini-3.8-flash",
            "gemini-4-hydrogen",
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


def test_discovered_model_can_be_selected_saved_and_tested(tmp_path) -> None:
    tested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1beta/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-9-experimental",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-3.8-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                    ]
                },
            )
        tested_models.append(json.loads(request.content)["model"])
        return success_handler(request)

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

        models = client.get("/provider/models", headers=HEADERS)
        assert models.status_code == 200
        assert "gemini-9-experimental" in models.json()["generation_models"]

        selected = client.put(
            "/provider/settings",
            headers=HEADERS,
            json={
                "generation_model": "gemini-9-experimental",
                "checker_model": "gemini-3.5-flash-lite",
                "free_tier_acknowledged": True,
            },
        )
        assert selected.status_code == 200
        assert selected.json()["generation_model"] == "gemini-9-experimental"

        tested = client.post("/provider/settings/test", headers=HEADERS)
        assert tested.status_code == 200
        assert tested.json()["model"] == "gemini-9-experimental"
        assert tested_models == ["gemini-9-experimental"]


def test_provider_error_response_excludes_credentials_and_source_text(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json=[
                {
                    "error": {
                        "code": 400,
                        "message": (
                            "secret-key and private source text appeared in the provider message"
                        ),
                        "status": "INVALID_ARGUMENT",
                        "details": [
                            {
                                "fieldViolations": [
                                    {
                                        "field": "secret-key",
                                        "description": "private source text",
                                    }
                                ]
                            }
                        ],
                    }
                }
            ],
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

        tested = client.post("/provider/settings/test", headers=HEADERS)
        assert tested.status_code == 400
        body = tested.text
        assert "secret-key" not in body
        assert "private source text" not in body
        assert "INVALID_ARGUMENT" not in body
        assert "provider rejected the request" in body.lower()

        usage = client.get("/provider/usage", headers=HEADERS)
        assert usage.status_code == 200
        assert usage.json()[0]["outcome"] == "failed"
        assert usage.json()[0]["error_code"] == "provider_rejected"
