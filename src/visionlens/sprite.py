"""Sprite sheet of low-rate thumbnails: the fallback player for browsers without VP9 WebM.

The UI picks it when `canPlayType` says the browser cannot play VP9 WebM (or with ?player=sprite).
It is a grid JPEG; the page draws the frame for the current time on a canvas. It is choppier than
video by design (a few frames per second) but keeps the demo from being blank.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np


def write_sprite(
    frames: Iterable[tuple[float, np.ndarray]],
    path: str | Path,
    *,
    fps: float = 6.0,
    thumb_w: int = 288,
    cols: int = 8,
    quality: int = 55,
) -> dict:
    """frames are (time, BGR). Returns the sheet geometry for the manifest."""
    thumbs: list[np.ndarray] = []
    next_t = 0.0
    for t, f in frames:
        if t + 1e-6 < next_t:
            continue
        h, w = f.shape[:2]
        th = int(round(h * thumb_w / w))
        thumbs.append(cv2.resize(f, (thumb_w, th), interpolation=cv2.INTER_AREA))
        next_t = len(thumbs) / fps
    if not thumbs:
        raise ValueError("no frames for the sprite")
    th = thumbs[0].shape[0]
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * thumb_w, 3), np.uint8)
    for k, im in enumerate(thumbs):
        r, c = divmod(k, cols)
        sheet[r * th : (r + 1) * th, c * thumb_w : (c + 1) * thumb_w] = im
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return {"fps": fps, "cols": cols, "thumb_w": thumb_w, "thumb_h": th, "count": len(thumbs)}
