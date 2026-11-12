"""Thread limits and machine description used by the benchmarks."""

from __future__ import annotations

import os
import platform


def default_threads() -> int:
    env = os.environ.get("VISIONLENS_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, (os.cpu_count() or 4) // 4)


def limit_threads(n: int | None = None) -> int:
    """Cap torch and OpenCV threads (the machine may be shared). Returns the count used."""
    n = n or default_threads()
    import cv2
    import torch

    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    return n


def resolve_device(pref: str | None = None) -> str:
    """'cpu', 'cuda' or 'auto'. Order: argument, VISIONLENS_DEVICE, then CUDA if present else CPU."""
    pref = (pref or os.environ.get("VISIONLENS_DEVICE") or "auto").lower()
    if pref == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    return pref


def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return platform.processor() or platform.machine()


def _cpu_times() -> tuple[int, int] | None:
    """(idle, total) cumulative CPU time, or None when the platform does not say."""
    try:
        if platform.system() == "Windows":
            import ctypes
            from ctypes import wintypes

            idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
            ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            )

            def val(f) -> int:
                return (f.dwHighDateTime << 32) | f.dwLowDateTime

            # kernel time includes idle time
            return val(idle), val(kernel) + val(user)
        with open("/proc/stat", encoding="utf-8") as fh:
            parts = [int(x) for x in fh.readline().split()[1:]]
        return parts[3] + parts[4], sum(parts)
    except Exception:  # noqa: BLE001 - best effort only
        return None


def observed_load(sample_s: float = 1.0) -> dict:
    """What the machine was doing right now: overall CPU use over a short sample, and the GPU's
    utilisation and memory when nvidia-smi is there. Recorded next to every benchmark group."""
    import subprocess
    import time

    out: dict = {}
    a = _cpu_times()
    time.sleep(sample_s)
    b = _cpu_times()
    if a and b and b[1] > a[1]:
        out["cpu_busy_percent"] = round(100 * (1 - (b[0] - a[0]) / (b[1] - a[1])), 1)
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if q.returncode == 0 and q.stdout.strip():
            u, used, total = (int(x) for x in q.stdout.strip().splitlines()[0].split(","))
            out.update(gpu_busy_percent=u, gpu_mem_used_mb=used, gpu_mem_total_mb=total)
    except Exception:  # noqa: BLE001
        pass
    return out
