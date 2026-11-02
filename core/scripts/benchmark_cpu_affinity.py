"""Windows CPU topology and thread-only affinity for supervised diagnostics.

Do not call pin_calling_thread in the application or benchmark parent. The ORT
worker uses it for its calling thread; ORT pins its one intra-op worker separately.
No process/system affinity or power setting is changed.
"""

import ctypes
import json
import os
import struct
import sys
from ctypes import wintypes


class GroupAffinity(ctypes.Structure):
    _fields_ = [
        ("Mask", ctypes.c_size_t),
        ("Group", wintypes.WORD),
        ("Reserved", wintypes.WORD * 3),
    ]


class ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class PowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", wintypes.BYTE),
        ("BatteryFlag", wintypes.BYTE),
        ("BatteryLifePercent", wintypes.BYTE),
        ("SystemStatusFlag", wintypes.BYTE),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


def kernel():
    if sys.platform != "win32" or ctypes.sizeof(ctypes.c_size_t) != 8:
        raise OSError("affinity diagnostic requires 64-bit Windows")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.GetCurrentThread.restype = wintypes.HANDLE
    api.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    api.GetLogicalProcessorInformationEx.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    api.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(GroupAffinity)]
    api.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    api.SetThreadAffinityMask.restype = ctypes.c_size_t
    api.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    api.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
    api.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
    api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenThread.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.GetSystemPowerStatus.argtypes = [ctypes.POINTER(PowerStatus)]
    return api


def topology():
    api = kernel()
    available, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not api.GetProcessAffinityMask(
        api.GetCurrentProcess(), ctypes.byref(available), ctypes.byref(system)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    size = wintypes.DWORD()
    api.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    if ctypes.get_last_error() != 122 or not size.value:
        raise OSError("processor topology size unavailable")
    buffer = ctypes.create_string_buffer(size.value)
    if not api.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    raw = buffer.raw[: size.value]
    offset, cores = 0, []
    while offset < len(raw):
        relationship, length = struct.unpack_from("<II", raw, offset)
        if relationship != 0 or length < 48 or offset + length > len(raw):
            raise ValueError("unexpected processor-core relationship")
        group_count = struct.unpack_from("<H", raw, offset + 30)[0]
        if group_count != 1 or length != 32 + 16 * group_count:
            raise ValueError("multiple processor groups are not supported by this diagnostic")
        mask, group = struct.unpack_from("<QH", raw, offset + 32)
        if group != 0:
            raise ValueError("only processor group zero is supported")
        cores.append([cpu for cpu in range(64) if mask & available.value & (1 << cpu)])
        offset += length
    usable = [core for core in cores if core]
    if len(usable) < 2:
        raise ValueError("two allowed physical cores are required")
    power = PowerStatus()
    if not api.GetSystemPowerStatus(ctypes.byref(power)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "physical_cores": usable,
        "allowed_process_mask": available.value,
        "selected_cpus": [usable[0][0], usable[1][0]],
        "numbering": "zero-based group-zero logical CPUs; ORT affinity entries are one-based",
        "ac_line_status": int(power.ACLineStatus),
        "battery_percent": int(power.BatteryLifePercent),
    }


def thread_affinities():
    api = kernel()
    snapshot = api.CreateToolhelp32Snapshot(4, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    rows = []
    try:
        entry = ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        more = api.Thread32First(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == os.getpid():
                handle = api.OpenThread(0x0040, False, entry.th32ThreadID)
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    affinity = GroupAffinity()
                    if not api.GetThreadGroupAffinity(handle, ctypes.byref(affinity)):
                        raise ctypes.WinError(ctypes.get_last_error())
                    rows.append(
                        {
                            "thread_id": int(entry.th32ThreadID),
                            "group": int(affinity.Group),
                            "mask": int(affinity.Mask),
                        }
                    )
                finally:
                    api.CloseHandle(handle)
            more = api.Thread32Next(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:
            raise ctypes.WinError(ctypes.get_last_error())
        return sorted(rows, key=lambda row: row["thread_id"])
    finally:
        api.CloseHandle(snapshot)


def pin_calling_thread(cpu):
    layout = topology()
    if not layout["allowed_process_mask"] & (1 << cpu):
        raise ValueError("selected calling CPU is unavailable")
    api = kernel()
    thread = api.GetCurrentThread()
    previous = api.SetThreadAffinityMask(thread, 1 << cpu)
    if not previous:
        raise ctypes.WinError(ctypes.get_last_error())
    current = GroupAffinity()
    if not api.GetThreadGroupAffinity(thread, ctypes.byref(current)):
        raise ctypes.WinError(ctypes.get_last_error())
    if current.Group != 0 or current.Mask != 1 << cpu:
        raise ValueError("calling-thread affinity was not applied")
    return {"thread_id": int(api.GetCurrentThreadId()), "group": 0, "mask": int(current.Mask)}


if __name__ == "__main__":
    print(json.dumps(topology(), indent=2))
