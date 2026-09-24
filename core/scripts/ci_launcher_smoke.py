"""Focused launcher smoke test for CI on macOS and Linux.

Starts the core the way the desktop shell does (in its own process group),
waits for the READY marker, checks authenticated and unauthenticated /health,
then stops the whole group and verifies no core process or listener remains.
Windows shutdown is covered by the desktop tests (`just e2e`).

Run with: uv run --directory core python scripts/ci_launcher_smoke.py
"""

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOKEN = "launcher-smoke-token"
CORE_ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> "None":
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def main() -> None:
    env = os.environ.copy()
    env["CORPI_CORE_TOKEN"] = TOKEN
    process = subprocess.Popen(
        [sys.executable, "-m", "core.api.server"],
        cwd=CORE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        port = wait_for_ready(process)
        print(f"core ready on port {port}")

        status = http_status(port, token=None)
        if status != 401:
            fail(f"unauthenticated /health returned {status}, expected 401")
        status = http_status(port, token=TOKEN)
        if status != 200:
            fail(f"authenticated /health returned {status}, expected 200")
        print("health checks passed (401 without token, 200 with token)")
    finally:
        stop_group(process)

    assert_tree_gone(process, port)
    print("shutdown left no core process and the port no longer answers")
    print("launcher smoke test passed")


def wait_for_ready(process: subprocess.Popen) -> int:
    deadline = time.time() + 60
    assert process.stdout is not None
    while time.time() < deadline:
        line = process.stdout.readline()
        if line.startswith("CORPI_CORE_READY"):
            fields = dict(part.split("=", 1) for part in line.split()[1:])
            return int(fields["port"])
        if line == "":
            fail("core exited before reporting ready")
    fail("core did not report ready within 60 seconds")
    raise AssertionError("unreachable")


def http_status(port: int, token: "str | None") -> int:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/health")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def stop_group(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        fail("core process did not exit after the group kill")


def assert_tree_gone(process: subprocess.Popen, port: int) -> None:
    try:
        os.killpg(os.getpgid(process.pid), 0)
    except ProcessLookupError:
        pass
    else:
        fail("core process group still has live members after shutdown")

    try:
        status = http_status(port, token=None)
    except (urllib.error.URLError, OSError):
        return
    fail(f"the core port still answers after shutdown (status {status})")


if __name__ == "__main__":
    main()
