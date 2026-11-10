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


def test_macos_and_linux_tauri_origin_is_accepted(client: TestClient) -> None:
    response = client.get(
        "/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": "tauri://localhost"},
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


def test_non_ascii_credentials_are_unauthorized(client: TestClient) -> None:
    response = client.get("/health", headers={b"Authorization": b"Bearer invalid-\xff"})
    assert response.status_code == 401


def test_non_ascii_preflight_headers_are_rejected(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            b"Origin": ALLOWED_ORIGIN.encode("ascii"),
            b"Access-Control-Request-Method": b"GET",
            b"Access-Control-Request-Headers": b"authorization,\xff",
        },
    )
    assert response.status_code == 400


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


def test_preflight_allows_pdf_upload_and_file_history_writes(client: TestClient) -> None:
    response = client.options(
        "/notebooks/example/pdf-acquisitions",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert "PUT" in response.headers["access-control-allow-methods"]
    assert "DELETE" in response.headers["access-control-allow-methods"]
    assert "authorization" in response.headers["access-control-allow-headers"]
