"""OpenCV video I/O.

Findings that shape this module (measured with opencv-python 5.0 on Windows): VP9 in WebM
encodes and plays in Chromium; H.264 encoding is unavailable without an openh264 DLL;
MPEG-4 Part 2 encodes but browsers do not play it; and seeking inside WebM is unreliable.
So reads are sequential and frame times come from container timestamps when they are sane.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".ogv", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int  # 0 when unknown
    duration: float  # 0.0 when unknown


def _sane(fps: float) -> bool:
    return 1.0 < fps <= 120.0


def probe(path: str | Path) -> VideoInfo:
    """Read size, fps and an estimated frame count; uses timestamps for odd containers."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not _sane(fps) or n <= 0 or n / max(fps, 1e-6) > 24 * 3600:
        stamps: list[float] = []
        for _ in range(40):
            ok, _f = cap.read()
            if not ok:
                break
            stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        deltas = np.diff(stamps) if len(stamps) > 2 else np.array([])
        deltas = deltas[deltas > 1e-4]
        fps = float(1.0 / np.median(deltas)) if len(deltas) else 25.0
        if not _sane(fps):
            fps = 25.0
        n = 0
    cap.release()
    if w <= 0 or h <= 0:
        raise ValueError(f"video has no frames: {path}")
    return VideoInfo(w, h, fps, n, n / fps if n else 0.0)


def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_h = max(2, int(round(h * scale)) // 2 * 2)
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def output_size(info: VideoInfo, max_width: int) -> tuple[int, int]:
    if info.width <= max_width:
        return info.width - info.width % 2, info.height - info.height % 2
    h = int(round(info.height * max_width / info.width))
    return max_width - max_width % 2, h - h % 2


def iter_frames(
    path: str | Path,
    *,
    max_width: int = 640,
    target_fps: float = 15.0,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, time, BGR frame), resampled to at most target_fps, read sequentially."""
    info = probe(path)
    out_fps = min(info.fps, target_fps)
    step = 1.0 / out_fps
    cap = cv2.VideoCapture(str(path))
    size = output_size(info, max_width)
    next_t = start_seconds
    k = 0
    i = 0
    last_ts = -1.0
    use_stamps = True
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if use_stamps and (ts <= last_ts or (k > 3 and last_ts <= 0.0)):
                use_stamps = False
            last_ts = ts
            t_src = ts if use_stamps else k / info.fps
            k += 1
            if t_src + 1e-6 < next_t:
                continue
            rel = i * step
            if max_seconds is not None and rel >= max_seconds:
                break
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield i, rel, frame
            i += 1
            next_t = start_seconds + i * step
    finally:
        cap.release()


def effective_fps(info: VideoInfo, target_fps: float) -> float:
    return min(info.fps, target_fps)


class WebmWriter:
    """VP9 WebM writer (plays in Chromium and Firefox)."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = str(path)
        self._w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"VP90"), fps, size)
        if not self._w.isOpened():
            raise RuntimeError("OpenCV could not open a VP9 WebM writer on this machine")

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def close(self) -> None:
        self._w.release()

    def __enter__(self) -> WebmWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_image(path: str | Path, max_width: int = 960) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return resize_to_width(img, max_width)


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXT
