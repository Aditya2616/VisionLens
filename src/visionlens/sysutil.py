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
