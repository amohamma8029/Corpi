"""The data-directory lock admits exactly one core per directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from core.persistence.corelock import DataDirectoryLock
from core.persistence.errors import CoreAlreadyRunningError, InvalidDataDirectoryError

CORE_ROOT = Path(__file__).resolve().parents[2]


def test_acquire_creates_the_lock_file_in_the_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "corpi-data"
    lock = DataDirectoryLock.acquire(data_dir)
    try:
        assert (data_dir / "core.lock").is_file()
    finally:
        lock.release()


def test_second_acquire_is_rejected_while_the_lock_is_held(tmp_path: Path) -> None:
    lock = DataDirectoryLock.acquire(tmp_path)
    try:
        with pytest.raises(CoreAlreadyRunningError) as caught:
            DataDirectoryLock.acquire(tmp_path)
        assert caught.value.code == "core_already_running"
    finally:
        lock.release()


def test_second_process_is_rejected_and_recovers_after_release(tmp_path: Path) -> None:
    probe = (
        "from pathlib import Path\n"
        "from core.persistence.corelock import DataDirectoryLock\n"
        "from core.persistence.errors import CoreAlreadyRunningError\n"
        f"target = Path({str(tmp_path)!r})\n"
        "try:\n"
        "    DataDirectoryLock.acquire(target)\n"
        "except CoreAlreadyRunningError as error:\n"
        "    print(error.code)\n"
        "else:\n"
        "    print('acquired')\n"
    )

    def run_probe() -> str:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=CORE_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    lock = DataDirectoryLock.acquire(tmp_path)
    try:
        assert run_probe() == "core_already_running"
    finally:
        lock.release()
    assert run_probe() == "acquired"


def test_release_is_idempotent(tmp_path: Path) -> None:
    lock = DataDirectoryLock.acquire(tmp_path)
    lock.release()
    lock.release()


def test_invalid_data_path_reports_the_invalid_directory_code(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file where the data directory should be", encoding="utf-8")
    with pytest.raises(InvalidDataDirectoryError) as caught:
        DataDirectoryLock.acquire(blocker)
    assert caught.value.code == "invalid_data_dir"
