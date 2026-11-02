"""Observed resident bytes, without another native Python dependency.

Linux uses /proc, Windows uses the documented process working-set API, macOS
uses ps's resident-kilobyte report. Failure is explicit; it is never a zero-byte
measurement. Sampling is a stop-after-observation control, not an allocation cap.
"""

import ctypes
import os
import subprocess
import sys
from functools import cache
from pathlib import Path


@cache
def _windows_api():
    # ctypes permanently caches pointer types: create this pair once, not per sample.
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    return kernel, psapi, Counters


def resident_bytes(pid: int) -> int:
    if sys.platform == "win32":
        kernel, psapi, Counters = _windows_api()
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            raise OSError("process_memory_unavailable")
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                raise OSError("process_memory_unavailable")
            return int(counters.WorkingSetSize)
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        value = Path(f"/proc/{pid}/statm").read_text().split()[1]
        return int(value) * os.sysconf("SC_PAGE_SIZE")
    if sys.platform == "darwin":
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        )
        return int(result.stdout.strip()) * 1024
    raise OSError("process_memory_platform_unsupported")
