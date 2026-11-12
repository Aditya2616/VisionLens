"""IoU matrices and Hungarian association."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of two (N,4) and (M,4) arrays of x1,y1,x2,y2 boxes."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def associate(
    sim: np.ndarray, min_sim: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Maximise total similarity with the Hungarian algorithm, keeping pairs >= min_sim.

    Returns (matches, unmatched_rows, unmatched_cols).
    """
    n, m = sim.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    rows, cols = linear_sum_assignment(-sim)
    matches: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    for r, c in zip(rows, cols, strict=True):
        if sim[r, c] >= min_sim and sim[r, c] > 0:
            matches.append((int(r), int(c)))
            used_r.add(int(r))
            used_c.add(int(c))
    return (
        matches,
        [r for r in range(n) if r not in used_r],
        [c for c in range(m) if c not in used_c],
    )
