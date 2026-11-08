import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.search.profiles import profiles
from core.search.supervisor import EmbeddingFailure, EmbeddingSession, EmbeddingStopped


def command(profile, body: str) -> list[str]:
    code = (
        "import os,json,sys,time; from core.search.provenance import runtime_versions; "
        f"print(json.dumps({{'type':'started','manifest':{profile.identity!r},"
        "'request':0,'pid':os.getpid()}),flush=True); " + body
    )
    return [sys.executable, "-c", code]


def assert_process_stopped(pid: int) -> None:
    if sys.platform != "win32":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE, no termination rights
    if not handle:
        if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: PID is absent
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        state = kernel.WaitForSingleObject(handle, 0)
        if state == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        assert state == 0, "owned process is still running"
    finally:
        kernel.CloseHandle(handle)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-object control")
def test_stopped_check_distinguishes_live_and_exited_readable_process() -> None:
    from core.search.memory import resident_bytes

    # Use the actual interpreter so the PID identifies the worker, not a venv redirector.
    process = subprocess.Popen(
        [str(Path(sys.base_prefix) / "python.exe"), "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with pytest.raises(AssertionError, match="still running"):
            assert_process_stopped(process.pid)
    finally:
        process.terminate()
        process.wait(timeout=5)
    # Popen retains a kernel handle. Memory remains queryable although exit is signalled.
    assert resident_bytes(process.pid) >= 0
    assert_process_stopped(process.pid)


def test_memory_limit_observes_interpreter_not_windows_venv_redirector(tmp_path: Path) -> None:
    profile = profiles()["minilm-en"]
    worker = command(profile, "allocation=bytearray(64*1024*1024); time.sleep(5)")
    began = time.monotonic()
    with pytest.raises(EmbeddingFailure, match="embedding_memory_limit"):
        EmbeddingSession(
            tmp_path, profile, command=worker, memory_limit=32 * 1024 * 1024, load_timeout=10
        )
    assert time.monotonic() - began < 10


def test_timeout_stops_and_reaps_owned_process_tree(tmp_path: Path) -> None:
    profile = profiles()["minilm-en"]
    marker = tmp_path / "worker.pid"
    worker = command(profile, f"open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(30)")
    with pytest.raises(TimeoutError):
        EmbeddingSession(tmp_path, profile, command=worker, load_timeout=0.75)
    if marker.exists():
        pid = int(marker.read_text())
        assert_process_stopped(pid)


def test_cancelled_session_stops_without_waiting_for_worker(tmp_path: Path) -> None:
    profile = profiles()["minilm-en"]
    with pytest.raises(EmbeddingStopped):
        EmbeddingSession(
            tmp_path,
            profile,
            should_stop=lambda: True,
            command=command(profile, "time.sleep(30)"),
            load_timeout=10,
        )


def test_worker_environment_has_no_core_or_provider_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    profile = profiles()["minilm-en"]
    monkeypatch.setenv("CORPI_CORE_TOKEN", "test-only-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-provider-secret")
    body = (
        "assert 'CORPI_CORE_TOKEN' not in os.environ and 'OPENAI_API_KEY' not in os.environ; "
        "assert os.environ['PYTHONUTF8']=='1' and sys.flags.utf8_mode==1; "
        f"print(json.dumps({{'type':'ready','manifest':{profile.identity!r},"
        "'request':0,'runtime':runtime_versions()}),flush=True); "
        "time.sleep(30)"
    )
    with EmbeddingSession(
        tmp_path, profile, command=command(profile, body), load_timeout=10
    ) as session:
        assert session.worker_pid is not None
        assert session.process.poll() is None
    assert session.process.poll() is not None


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_receives_explicit_saved_arena_and_spinning_flags(tmp_path, enabled):
    profile = profiles()["minilm-en"]
    expected = "1" if enabled else "0"
    body = (
        f"assert sys.argv[3:5]==[{expected!r},{expected!r}]; "
        f"print(json.dumps({{'type':'ready','manifest':{profile.identity!r},"
        "'request':0,'runtime':runtime_versions()}),flush=True); time.sleep(30)"
    )
    with EmbeddingSession(
        tmp_path,
        profile,
        command=command(profile, body),
        load_timeout=10,
        memory_arena=enabled,
        stop_spinning=enabled,
    ) as session:
        pid = session.worker_pid
        assert pid is not None
    assert_process_stopped(pid)
