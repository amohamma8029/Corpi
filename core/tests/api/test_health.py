from fastapi.testclient import TestClient

TOKEN = "test-per-launch-token"


def test_health_without_token_is_unauthorized(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 401


def test_health_with_wrong_token_is_unauthorized(client: TestClient) -> None:
    response = client.get("/health", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401


def test_health_with_current_token_succeeds(client: TestClient) -> None:
    response = client.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["app"] == "corpi-core"
    assert payload["status"] == "ok"
    assert payload["version"]
