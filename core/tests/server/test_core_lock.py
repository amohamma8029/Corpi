"""The core refuses startup when another core already holds the data directory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2]
TOKEN = "two-core-token"
READY_MARKER = "CORPI_CORE_READY"
ERROR_MARKER = "CORPI_CORE_ERROR"


def start_core(data_dir: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CORPI_CORE_TOKEN"] = TOKEN
    env["CORPI_DATA_DIR"] = str(data_dir)
    return subprocess.Popen(
        [sys.executable, "-m", "core.api.server"],
        cwd=CORE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def read_first_line(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    for _ in range(400):
        line = process.stdout.readline()
        if line:
            return line.strip()
        if process.poll() is not None:
            break
    return ""


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def test_second_core_refuses_startup_while_the_first_holds_the_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "corpi-data"
    first = start_core(data_dir)
    try:
        first_line = read_first_line(first)
        assert first_line.startswith(READY_MARKER), first_line

        second = start_core(data_dir)
        try:
            second_line = read_first_line(second)
            assert second_line == f"{ERROR_MARKER} code=core_already_running", second_line
            assert second.wait(timeout=10) == 2
        finally:
            stop(second)
    finally:
        stop(first)


def test_core_startup_recovers_after_the_first_core_stops(tmp_path: Path) -> None:
    data_dir = tmp_path / "corpi-data"
    first = start_core(data_dir)
    try:
        assert read_first_line(first).startswith(READY_MARKER)
    finally:
        stop(first)

    second = start_core(data_dir)
    try:
        assert read_first_line(second).startswith(READY_MARKER)
    finally:
        stop(second)
