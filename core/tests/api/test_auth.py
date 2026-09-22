from fastapi.testclient import TestClient

TOKEN = "test-per-launch-token"
ALLOWED_ORIGIN = "http://tauri.localhost"


def test_foreign_origin_is_forbidden(client: TestClient) -> None:
    response = client.get(
        "/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": "http://evil.example"},
    )
    assert response.status_code == 403


def test_allowed_origin_is_accepted(client: TestClient) -> None:
    response = client.get(
        "/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": ALLOWED_ORIGIN},
    )
    assert response.status_code == 200


def test_non_local_host_is_forbidden(client: TestClient) -> None:
    response = client.get(
        "http://metadata.evil.internal/health",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 403


def test_absent_origin_is_allowed_for_non_browser_clients(client: TestClient) -> None:
    response = client.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200


def test_allowed_response_carries_cors_headers(client: TestClient) -> None:
    response = client.get(
        "/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": ALLOWED_ORIGIN},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert response.headers["vary"] == "Origin"


def test_preflight_from_allowed_origin_is_answered_without_token(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "authorization" in response.headers["access-control-allow-headers"]


def test_preflight_from_foreign_origin_is_forbidden(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 403
