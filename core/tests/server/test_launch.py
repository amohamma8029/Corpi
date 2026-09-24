import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

CORE_ROOT = Path(__file__).resolve().parents[2]
TOKEN = "launch-contract-token"


@pytest.fixture()
def core_server() -> Iterator[int]:
    """Launch the real server entry point and yield its OS-assigned port."""
    env = os.environ.copy()
    env["CORPI_CORE_TOKEN"] = TOKEN
    process = subprocess.Popen(
        [sys.executable, "-m", "core.api.server"],
        cwd=CORE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        stream = process.stdout
        if stream is None:
            raise AssertionError("core server stdout unavailable")
        ready_line = ""
        for _ in range(200):
            line = stream.readline()
            if line.startswith("CORPI_CORE_READY"):
                ready_line = line.strip()
                break
            if line == "":
                raise AssertionError("core server exited before reporting ready")
        if not ready_line:
            raise AssertionError("core server did not report ready")
        fields = dict(part.split("=", 1) for part in ready_line.split()[1:])
        yield int(fields["port"])
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_health_requires_token(core_server: int) -> None:
    response = httpx.get(f"http://127.0.0.1:{core_server}/health")
    assert response.status_code == 401


def test_health_accepts_launch_token(core_server: int) -> None:
    response = httpx.get(
        f"http://127.0.0.1:{core_server}/health",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_non_local_host_is_rejected(core_server: int) -> None:
    response = httpx.get(
        f"http://127.0.0.1:{core_server}/health",
        headers={"Authorization": f"Bearer {TOKEN}", "Host": "metadata.evil.internal"},
    )
    assert response.status_code == 403


def test_ready_line_means_immediately_listening(core_server: int) -> None:
    """The first request right after READY must succeed — the marker may only
    print once the socket is accepting connections (no retry allowed here)."""
    response = httpx.get(
        f"http://127.0.0.1:{core_server}/health",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 200


def test_stream_delivers_chunks_over_live_socket(core_server: int) -> None:
    events: list[dict] = []
    with httpx.stream(
        "GET",
        f"http://127.0.0.1:{core_server}/demo/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    assert [event["step"] for event in events if "step" in event] == [1, 2, 3, 4, 5]
    assert events[-1] == {"done": True}
