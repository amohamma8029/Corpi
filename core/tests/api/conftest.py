import pytest
from fastapi.testclient import TestClient

from core.api.app import create_app

TEST_TOKEN = "test-per-launch-token"
TEST_ORIGIN = "http://tauri.localhost"


@pytest.fixture()
def client() -> TestClient:
    app = create_app(token=TEST_TOKEN)
    return TestClient(app, base_url="http://127.0.0.1")
