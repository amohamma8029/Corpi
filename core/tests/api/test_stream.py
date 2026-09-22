import json

from fastapi.testclient import TestClient

TOKEN = "test-per-launch-token"


def _data_events(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def test_stream_without_token_is_unauthorized(client: TestClient) -> None:
    response = client.get("/demo/stream")
    assert response.status_code == 401


def test_stream_delivers_authenticated_chunks_in_order(client: TestClient) -> None:
    with client.stream(
        "GET", "/demo/stream", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk.decode() for chunk in response.iter_bytes())

    events = _data_events(body)
    steps = [event["step"] for event in events if "step" in event]
    assert steps == [1, 2, 3, 4, 5]
    assert events[-1] == {"done": True}
