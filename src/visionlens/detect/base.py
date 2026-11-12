from __future__ import annotations

from typing import Protocol

import numpy as np

from ..schemas import Detection


class Detector(Protocol):
    name: str
    input_size: int

    def detect(self, frame: np.ndarray | None, index: int = 0) -> list[Detection]:
        """Detect objects in a BGR frame. `index` lets test doubles look up per-frame results."""
        ...


def nms(dets: list[Detection], iou_thr: float = 0.7) -> list[Detection]:
    """Class-aware greedy NMS. DETR-family models need little, but nested duplicates occur."""
    from ..track.iou import iou_matrix

    if len(dets) < 2:
        return dets
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    boxes = np.array([d.box for d in dets])
    keep: list[int] = []
    for i in order:
        if all(
            dets[i].cls != dets[k].cls
            or iou_matrix(boxes[i : i + 1], boxes[k : k + 1])[0, 0] < iou_thr
            for k in keep
        ):
            keep.append(i)
    return [dets[i] for i in sorted(keep)]
