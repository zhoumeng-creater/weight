"""Stable host attestation and cross-platform process resource sampling."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
from typing import Any

import numpy as np


class HostSamplingError(RuntimeError):
    """A required host or live-process resource value is unavailable."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _linux_cpu_model() -> str:
    text = _read_text(Path("/proc/cpuinfo"))
    for line in text.splitlines():
        if line.lower().startswith(("model name", "hardware")):
            return line.split(":", 1)[-1].strip()
    return ""


def _linux_memory_bytes() -> int:
    match = re.search(
        r"^MemTotal:\s+(\d+)\s+kB$",
        _read_text(Path("/proc/meminfo")),
        flags=re.MULTILINE,
    )
    return 0 if match is None else int(match.group(1)) * 1024


def _linux_cgroup_memory_limit_bytes() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        value = _read_text(path).strip()
        if not value or value == "max":
            continue
        try:
            limit = int(value)
        except ValueError:
            continue
        if 0 < limit < (1 << 60):
            return limit
    return None


def _windows_memory_bytes() -> int:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0
    return int(status.ullTotalPhys)


def _visible_cpu_ids() -> tuple[int, ...]:
    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            return tuple(sorted(int(value) for value in affinity(0)))
        except OSError:
            pass
    return tuple(range(int(os.cpu_count() or 0)))


def _linux_numa_nodes() -> tuple[str, ...]:
    root = Path("/sys/devices/system/node")
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            path.name
            for path in root.glob("node[0-9]*")
            if path.is_dir()
        )
    )


def _linux_cpu_quota() -> tuple[str, float | None]:
    value = _read_text(Path("/sys/fs/cgroup/cpu.max")).strip()
    if value:
        parts = value.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota = int(parts[0])
                period = int(parts[1])
            except ValueError:
                return value, None
            if quota > 0 and period > 0:
                return value, quota / period
        return value, None
    quota = _read_text(
        Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    ).strip()
    period = _read_text(
        Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    ).strip()
    value = f"{quota} {period}".strip()
    try:
        quota_value = int(quota)
        period_value = int(period)
    except ValueError:
        return value, None
    if quota_value > 0 and period_value > 0:
        return value, quota_value / period_value
    return value, None


def host_fingerprint() -> dict[str, Any]:
    """Return stable execution-relevant properties of the visible host."""

    visible_cpu_ids = _visible_cpu_ids()
    if os.name == "nt":
        memory_bytes = _windows_memory_bytes()
        host_memory_bytes = memory_bytes
        cgroup_memory_limit_bytes = None
        cpu_model = (
            os.environ.get("PROCESSOR_IDENTIFIER")
            or platform.processor()
        )
        numa_nodes: tuple[str, ...] = ()
        cpu_quota = ""
        cpu_quota_processors = None
        stable_host_material = (
            os.environ.get("COMPUTERNAME") or socket.gethostname()
        )
    elif platform.system() == "Linux":
        host_memory_bytes = _linux_memory_bytes()
        cgroup_memory_limit_bytes = _linux_cgroup_memory_limit_bytes()
        memory_bytes = (
            min(host_memory_bytes, cgroup_memory_limit_bytes)
            if host_memory_bytes > 0
            and cgroup_memory_limit_bytes is not None
            else host_memory_bytes or int(cgroup_memory_limit_bytes or 0)
        )
        cpu_model = _linux_cpu_model() or platform.processor()
        numa_nodes = _linux_numa_nodes()
        cpu_quota, cpu_quota_processors = _linux_cpu_quota()
        stable_host_material = (
            _read_text(Path("/etc/machine-id")).strip()
            or socket.gethostname()
        )
    else:
        host_memory_bytes = 0
        cgroup_memory_limit_bytes = None
        memory_bytes = 0
        cpu_model = platform.processor()
        numa_nodes = ()
        cpu_quota = ""
        cpu_quota_processors = None
        stable_host_material = socket.gethostname()
    quota_limited_processors = (
        len(visible_cpu_ids)
        if cpu_quota_processors is None
        else max(1, math.floor(cpu_quota_processors))
    )
    effective_logical_processors = min(
        len(visible_cpu_ids),
        quota_limited_processors,
    )
    return {
        "schema_version": "WGT-HOST-FINGERPRINT-1.0",
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model.strip(),
        "visible_cpu_ids": list(visible_cpu_ids),
        "visible_logical_processors": len(visible_cpu_ids),
        "effective_logical_processors": effective_logical_processors,
        "os_cpu_count": int(os.cpu_count() or 0),
        "cpu_quota": cpu_quota,
        "cpu_quota_processors": cpu_quota_processors,
        "numa_nodes": list(numa_nodes),
        "memory_bytes": memory_bytes,
        "host_memory_bytes": host_memory_bytes,
        "cgroup_memory_limit_bytes": cgroup_memory_limit_bytes,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "host_instance_sha256": sha256(
            stable_host_material.encode("utf-8")
        ).hexdigest(),
        "byteorder": sys.byteorder,
    }


def host_fingerprint_sha256(value: dict[str, Any] | None = None) -> str:
    return sha256(
        _canonical_json_bytes(host_fingerprint() if value is None else value)
    ).hexdigest()


def _windows_process_handle(process_id: int):
    query_information = 0x0400
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(query_information, False, process_id)
    if not handle:
        raise HostSamplingError(
            f"OpenProcess failed for live process {process_id}"
        )
    return kernel32, handle


def process_rss_bytes(process_id: int) -> int:
    """Return current resident bytes for a live worker process."""

    if os.name != "nt":
        text = _read_text(Path(f"/proc/{process_id}/status"))
        match = re.search(
            r"^VmRSS:\s+(\d+)\s+kB$",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise HostSamplingError(
                f"RSS unavailable for live process {process_id}"
            )
        return int(match.group(1)) * 1024

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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

    kernel32, handle = _windows_process_handle(process_id)
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise HostSamplingError(
                f"GetProcessMemoryInfo failed for live process {process_id}"
            )
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def process_cpu_seconds(process_id: int) -> float:
    """Return accumulated user+kernel CPU seconds for a live worker."""

    if os.name != "nt":
        text = _read_text(Path(f"/proc/{process_id}/stat"))
        if not text:
            raise HostSamplingError(
                f"CPU time unavailable for live process {process_id}"
            )
        try:
            fields = text.rsplit(")", 1)[1].split()
            ticks = int(fields[11]) + int(fields[12])
            ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        except (IndexError, TypeError, ValueError, OSError) as error:
            raise HostSamplingError(
                f"CPU time malformed for live process {process_id}"
            ) from error
        return ticks / ticks_per_second

    kernel32, handle = _windows_process_handle(process_id)
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        kernel32.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        )
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        if not ok:
            raise HostSamplingError(
                f"GetProcessTimes failed for live process {process_id}"
            )

        def ticks(value: wintypes.FILETIME) -> int:
            return (int(value.dwHighDateTime) << 32) | int(
                value.dwLowDateTime
            )

        return (ticks(kernel) + ticks(user)) / 10_000_000.0
    finally:
        kernel32.CloseHandle(handle)
