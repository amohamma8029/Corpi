import ctypes
import os
import sys

import pytest

from core.search.memory import resident_bytes


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
