import ctypes
import os
import sys

import pytest

from core.search.memory import resident_bytes


def test_sampler_reports_current_process_resident_bytes():
    assert resident_bytes(os.getpid()) > 0


@pytest.mark.parametrize("platform", ["linux", "darwin", "unsupported"])
def test_windows_api_rejects_non_windows_platforms(platform, monkeypatch):
    from core.search.memory import _windows_api

    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(OSError, match="process_memory_platform_unsupported"):
        # A previous Windows sample may have populated the cache already.
        _windows_api.__wrapped__()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ctypes sampler")
def test_windows_sampler_reuses_pointer_types_and_reports_unavailable_process():
    from core.search.memory import _windows_api

    api = _windows_api()
    pointers = vars(ctypes)["_pointer_type_cache"]
    count = len(pointers)
    for _ in range(1000):
        assert resident_bytes(os.getpid()) > 0
        assert _windows_api() is api
    assert len(pointers) == count
    with pytest.raises(OSError, match="process_memory_unavailable"):
        resident_bytes(0)
