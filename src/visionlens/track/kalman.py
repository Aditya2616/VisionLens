"""Constant-velocity Kalman filter over a box, in the SORT parameterisation.

State x = [cx, cy, s, r, vcx, vcy, vs] with s = area and r = aspect ratio (w / h, held
constant). Measurement z = [cx, cy, s, r]. One step equals one video frame.
"""

from __future__ import annotations

import numpy as np


def xyxy_to_z(box: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w * h, w / h])


def x_to_xyxy(x: np.ndarray) -> tuple[float, float, float, float]:
    s = max(float(x[2]), 1e-3)
    r = max(float(x[3]), 1e-3)
    w = float(np.sqrt(s * r))
    h = s / max(w, 1e-6)
    cx, cy = float(x[0]), float(x[1])
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class BoxKalman:
    def __init__(self, box: tuple[float, float, float, float]) -> None:
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
        self.x = np.zeros(7)
        self.x[:4] = xyxy_to_z(box)

    def predict(self) -> tuple[float, float, float, float]:
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_xyxy(self.x)

    def update(self, box: tuple[float, float, float, float] | np.ndarray) -> None:
        z = xyxy_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self) -> tuple[float, float, float, float]:
        return x_to_xyxy(self.x)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()

    def restore(self, snap: tuple[np.ndarray, np.ndarray]) -> None:
        self.x, self.P = snap[0].copy(), snap[1].copy()
