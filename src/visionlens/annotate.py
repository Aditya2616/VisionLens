"""Draw an analysis result onto a video with OpenCV (the CLI `annotate` command)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schemas import FrameResult, Result
from .video import WebmWriter, iter_frames

LIME = (63, 255, 215)  # BGR of #d7ff3f
WHITE = (235, 235, 230)


def draw_frame(
    img: np.ndarray,
    fr: FrameResult,
    trails: dict[int, list[tuple[int, int]]],
    show_dets: bool = False,
) -> np.ndarray:
    if show_dets and fr.dets:
        for d in fr.dets:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 1)
    for t in fr.tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box)
        col = LIME if t.detected else (110, 170, 130)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if t.detected else 1)
        label = f"{t.cls} {t.id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(
            img,
            label,
            (x1 + 3, max(th + 1, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
        trail = trails.setdefault(t.id, [])
        trail.append(((x1 + x2) // 2, y2))
        del trail[:-30]
        for a, b in zip(trail, trail[1:], strict=False):
            cv2.line(img, a, b, col, 1, cv2.LINE_AA)
    return img


def annotate_video(
    video: str | Path, result: Result, out: str | Path, show_dets: bool = False
) -> None:
    trails: dict[int, list[tuple[int, int]]] = {}
    w = None
    by_i = {f.i: f for f in result.frames}
    try:
        for i, _t, frame in iter_frames(
            video, max_width=result.meta.width, target_fps=result.meta.fps
        ):
            if w is None:
                w = WebmWriter(out, result.meta.fps, (frame.shape[1], frame.shape[0]))
            fr = by_i.get(i)
            w.write(draw_frame(frame, fr, trails, show_dets) if fr else frame)
    finally:
        if w is not None:
            w.close()
