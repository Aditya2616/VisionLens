"""Deterministic synthetic scenes with exact ground truth.

These scenes exist so tracker quality can be measured against truth that I control. They are
simple (flat shapes on a flat background) and easier than real footage; numbers measured here
are not comparable with MOT benchmark results.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .schemas import Detection

Box = tuple[float, float, float, float]


@dataclass
class Agent:
    id: int
    cls: str  # "person" | "car"
    foot: list[tuple[float, float]]  # per-frame foot (bottom centre) position; may be off screen
    alive: list[bool]
    scale: float = 1.0
    tone: int = 0  # picks a colour

    def box(self, i: int, H: int) -> Box:
        x, y = self.foot[i]
        base_h = 0.2 * H * (0.55 + 0.9 * (y / H)) * self.scale
        if self.cls == "person":
            h, w = base_h, base_h * 0.38
        else:
            h, w = base_h * 0.62, base_h * 1.45
        return (x - w / 2, y - h, x + w / 2, y)


@dataclass
class Scene:
    name: str
    width: int
    height: int
    fps: float
    n_frames: int
    agents: list[Agent]
    occluders: list[tuple[int, int, int, int]] = field(default_factory=list)
    seed: int = 0

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    # ------------------------------------------------------------------ ground truth
    def gt(self, i: int) -> list[tuple[int, Box, float, str]]:
        """Visible ground truth for frame i: (id, clipped box, visibility 0..1, class)."""
        W, H = self.width, self.height
        out = []
        boxes = {a.id: a.box(i, H) for a in self.agents if a.alive[i]}
        for a in self.agents:
            if not a.alive[i]:
                continue
            x1, y1, x2, y2 = boxes[a.id]
            area = (x2 - x1) * (y2 - y1)
            cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            carea = (cx2 - cx1) * (cy2 - cy1)
            if carea < 0.4 * area:
                continue
            mask = np.zeros((int(cy2 - cy1) + 1, int(cx2 - cx1) + 1), np.uint8)

            clip = (cx1, cy1, cx2, cy2)
            for o in self.occluders:
                _cover(mask, (o[0], o[1], o[2], o[3]), clip)
            for other in self.agents:
                if other.id != a.id and other.alive[i] and other.foot[i][1] > a.foot[i][1]:
                    _cover(mask, boxes[other.id], clip)
            vis = 1.0 - float(mask.mean())
            out.append((a.id, (cx1, cy1, cx2, cy2), vis, a.cls))
        return out

    def gt_sequence(self, min_visibility: float = 0.25) -> dict[int, list[tuple[int, Box]]]:
        seq: dict[int, list[tuple[int, Box]]] = {}
        for i in range(self.n_frames):
            seq[i] = [(g, b) for g, b, v, _ in self.gt(i) if v >= min_visibility]
        return seq

    # ------------------------------------------------------------------- rendering
    def render(self, i: int, noise: float = 0.0) -> np.ndarray:
        W, H = self.width, self.height
        img = _background(W, H, self.name)
        order = sorted((a for a in self.agents if a.alive[i]), key=lambda a: a.foot[i][1])
        for a in order:
            x1, y1, x2, y2 = (int(round(v)) for v in a.box(i, H))
            if a.cls == "person":
                _draw_person(img, x1, y1, x2, y2, a.tone, phase=i * 0.5 + a.id)
            else:
                _draw_car(img, x1, y1, x2, y2, a.tone, facing_right=_facing_right(a, i))
        for o in self.occluders:
            cv2.rectangle(img, (o[0], o[1]), (o[2], o[3]), (52, 56, 60), -1)
            cv2.rectangle(img, (o[0], o[1]), (o[2], o[3]), (74, 80, 86), 2)
        if noise <= 0:
            return img
        rng = np.random.default_rng(self.seed * 100003 + i)
        grain = rng.normal(0, noise, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def _cover(mask: np.ndarray, r: Box, clip: Box) -> None:
    """Mark the part of rectangle r that falls inside the clipped box (origin clip[:2])."""
    cx1, cy1, cx2, cy2 = clip
    rx1, ry1 = int(max(r[0], cx1) - cx1), int(max(r[1], cy1) - cy1)
    rx2, ry2 = int(min(r[2], cx2) - cx1), int(min(r[3], cy2) - cy1)
    if rx2 > rx1 and ry2 > ry1:
        mask[ry1:ry2, rx1:rx2] = 1


def _facing_right(a: Agent, i: int) -> bool:
    j = min(i + 1, len(a.foot) - 1)
    k = max(j - 2, 0)
    return a.foot[j][0] >= a.foot[k][0]


_TONES = [
    (222, 255, 63),
    (60, 150, 255),
    (230, 230, 235),
    (140, 120, 255),
    (255, 170, 90),
    (120, 200, 170),
]


def _background(W: int, H: int, name: str) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    horizon = int(H * 0.32)
    for y in range(H):
        t = y / H
        img[y, :] = (int(18 + 14 * t), int(16 + 12 * t), int(14 + 10 * t))
    cv2.rectangle(img, (0, 0), (W, horizon), (26, 24, 22), -1)
    # distant buildings
    rng = np.random.default_rng(7)
    x = 0
    while x < W:
        bw = int(rng.integers(30, 80))
        bh = int(rng.integers(int(horizon * 0.35), int(horizon * 0.95)))
        shade = int(rng.integers(34, 52))
        cv2.rectangle(img, (x, horizon - bh), (x + bw, horizon), (shade, shade, shade + 2), -1)
        x += bw + 2
    cv2.rectangle(img, (0, horizon), (W, H), (36, 36, 38), -1)
    if name == "street":
        road_top, road_bot = int(H * 0.50), int(H * 0.74)
        cv2.rectangle(img, (0, road_top), (W, road_bot), (28, 28, 30), -1)
        for x0 in range(0, W, 48):
            cv2.line(
                img,
                (x0, (road_top + road_bot) // 2),
                (x0 + 24, (road_top + road_bot) // 2),
                (92, 92, 90),
                2,
            )
    return img


def _draw_person(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, tone: int, phase: float
) -> None:
    col = _TONES[tone % len(_TONES)]
    h, w = y2 - y1, x2 - x1
    cx = (x1 + x2) // 2
    head_r = max(2, int(h * 0.11))
    cv2.circle(img, (cx, y1 + head_r + 1), head_r, (170, 190, 215), -1)
    body_top = y1 + 2 * head_r + 2
    cv2.rectangle(img, (x1 + 1, body_top), (x2 - 1, body_top + int(h * 0.42)), col, -1)
    leg = int(math.sin(phase) * w * 0.18)
    hip = body_top + int(h * 0.42)
    cv2.line(img, (cx - w // 6, hip), (cx - w // 6 + leg, y2), (70, 76, 92), max(2, w // 4))
    cv2.line(img, (cx + w // 6, hip), (cx + w // 6 - leg, y2), (60, 66, 82), max(2, w // 4))


def _draw_car(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, tone: int, facing_right: bool
) -> None:
    col = _TONES[(tone + 2) % len(_TONES)]
    h, w = y2 - y1, x2 - x1
    body = (x1, y1 + int(h * 0.30), x2, y2 - int(h * 0.12))
    cv2.rectangle(img, body[:2], body[2:], col, -1)
    cab_x1 = x1 + int(w * (0.22 if facing_right else 0.18))
    cab_x2 = x2 - int(w * (0.18 if facing_right else 0.22))
    cv2.rectangle(
        img, (cab_x1, y1), (cab_x2, y1 + int(h * 0.34)), tuple(int(c * 0.75) for c in col), -1
    )
    cv2.rectangle(img, (cab_x1 + 3, y1 + 3), (cab_x2 - 3, y1 + int(h * 0.30)), (150, 168, 180), -1)
    r = max(3, int(h * 0.17))
    for wx in (x1 + int(w * 0.22), x2 - int(w * 0.22)):
        cv2.circle(img, (wx, y2 - r + 1), r, (20, 20, 22), -1)


# ---------------------------------------------------------------------------- scenes
def _agent(
    aid: int,
    cls: str,
    n: int,
    fps: float,
    keys: list[tuple[float, float, float]],
    tone: int,
    scale: float = 1.0,
) -> Agent:
    """Piecewise-linear foot path through (t, x, y) keys; alive between the first and last key."""
    ts = np.array([k[0] for k in keys])
    xs = np.array([k[1] for k in keys])
    ys = np.array([k[2] for k in keys])
    foot, alive = [], []
    for i in range(n):
        t = i / fps
        foot.append((float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys))))
        alive.append(bool(ts[0] <= t <= ts[-1]))
    return Agent(aid, cls, foot, alive, scale, tone)


def _walk(
    aid: int,
    cls: str,
    n: int,
    fps: float,
    rng: np.random.Generator,
    W: int,
    H: int,
    tone: int,
    t_in: float,
    t_out: float,
) -> Agent:
    """Smooth random walk that enters at t_in and leaves at t_out."""
    foot, alive = [], []
    x = float(rng.choice([-10, W + 10])) if rng.random() < 0.6 else float(rng.uniform(0.1, 0.9) * W)
    y = float(rng.uniform(0.62, 0.95) * H)
    vx = (1 if x < W / 2 else -1) * float(rng.uniform(25, 55))
    vy = float(rng.uniform(-6, 6))
    for i in range(n):
        t = i / fps
        if t >= t_in:
            vx += float(rng.normal(0, 4)) / fps * 10
            vy += float(rng.normal(0, 3)) / fps * 10
            vx = float(np.clip(vx, -70, 70))
            vy = float(np.clip(vy, -14, 14))
            x += vx / fps
            y = float(np.clip(y + vy / fps, 0.58 * H, 0.97 * H))
        foot.append((x, y))
        alive.append(t_in <= t <= t_out)
    return Agent(aid, cls, foot, alive, 1.0, tone)


def make_scene(
    name: str,
    seed: int = 0,
    seconds: float = 10.0,
    fps: float = 15.0,
    size: tuple[int, int] = (640, 360),
) -> Scene:
    W, H = size
    n = int(round(seconds * fps))
    rng = np.random.default_rng(seed)
    agents: list[Agent] = []
    occ: list[tuple[int, int, int, int]] = []
    s = seconds
    if name == "crossing":
        # two groups cross in an X; two more walk towards each other along one row
        agents = [
            _agent(1, "person", n, fps, [(0, -20, 0.76 * H), (s, W + 20, 0.76 * H)], 0),
            _agent(2, "person", n, fps, [(0, W + 20, 0.86 * H), (s, -20, 0.86 * H)], 1),
            _agent(3, "person", n, fps, [(0, -20, 0.70 * H), (s, W + 20, 0.92 * H)], 2),
            _agent(4, "person", n, fps, [(0, W + 20, 0.92 * H), (s, -20, 0.70 * H)], 3),
            _agent(5, "person", n, fps, [(0, 0.2 * W, 0.62 * H), (s, 0.8 * W, 0.62 * H)], 4),
            _agent(6, "person", n, fps, [(0, 0.8 * W, 0.68 * H), (s, 0.2 * W, 0.68 * H)], 5),
        ]
    elif name == "occlusion":
        occ = [
            (int(0.30 * W), int(0.30 * H), int(0.37 * W), int(0.98 * H)),
            (int(0.62 * W), int(0.30 * H), int(0.67 * W), int(0.98 * H)),
        ]
        agents = [
            _agent(1, "person", n, fps, [(0, -20, 0.85 * H), (s, W + 20, 0.85 * H)], 0),
            _agent(2, "person", n, fps, [(0, W + 20, 0.78 * H), (s, -20, 0.78 * H)], 1),
            _agent(3, "car", n, fps, [(0, -80, 0.68 * H), (s, W + 80, 0.68 * H)], 2),
            _agent(
                4,
                "person",
                n,
                fps,
                [(0, 0.5 * W, 0.97 * H), (3, 0.5 * W, 0.97 * H), (s, 0.9 * W, 0.7 * H)],
                3,
            ),
        ]
    elif name == "crowd":
        for k in range(9):
            t_in = float(rng.uniform(0, 0.4 * s))
            t_out = float(rng.uniform(0.7 * s, s + 1))
            agents.append(_walk(k + 1, "person", n, fps, rng, W, H, k, t_in, t_out))
    elif name == "street":
        agents = [
            _agent(1, "car", n, fps, [(0, -90, 0.60 * H), (s, W + 90, 0.60 * H)], 0),
            _agent(2, "car", n, fps, [(1.0, W + 90, 0.68 * H), (s, -90, 0.68 * H)], 1),
            _agent(3, "car", n, fps, [(0, 0.78 * W, 0.99 * H), (s, 0.78 * W, 0.99 * H)], 2, 0.8),
            _agent(4, "person", n, fps, [(0, -20, 0.88 * H), (s, W + 20, 0.88 * H)], 3),
            _agent(5, "person", n, fps, [(0, -45, 0.88 * H), (s, W + 5, 0.88 * H)], 4),
            _agent(
                6,
                "person",
                n,
                fps,
                [(0.5, 0.45 * W, 0.78 * H), (4, 0.45 * W, 0.78 * H), (s, 0.2 * W, 0.80 * H)],
                5,
            ),
            _agent(7, "person", n, fps, [(2, W + 20, 0.92 * H), (s - 1, 0.35 * W, 0.92 * H)], 1),
        ]
    else:
        raise ValueError(f"unknown scene: {name}")
    return Scene(name, W, H, fps, n, agents, occ, seed)


SCENES = ["crossing", "occlusion", "crowd", "street"]


# ------------------------------------------------------------------ noisy oracle detector
@dataclass
class NoiseConfig:
    p_miss: float = 0.05  # random missed detections
    jitter: float = 0.04  # box corner noise as a fraction of box size
    p_low: float = 0.12  # chance a detection gets a low confidence (partial-occlusion look)
    fp_per_frame: float = 0.15  # mean false positives per detector call
    seed: int = 0

    @staticmethod
    def level(k: int, seed: int = 0) -> NoiseConfig:
        presets = [
            NoiseConfig(0.0, 0.0, 0.0, 0.0, seed),
            NoiseConfig(0.05, 0.03, 0.10, 0.10, seed),
            NoiseConfig(0.12, 0.06, 0.20, 0.30, seed),
            NoiseConfig(0.25, 0.10, 0.30, 0.60, seed),
        ]
        return presets[k]


class NoisyOracle:
    """Detector stand-in: ground truth plus controlled noise. Used for tracker evaluation."""

    name = "oracle"

    def __init__(self, scene: Scene, noise: NoiseConfig) -> None:
        self.scene = scene
        self.noise = noise
        self.rng = np.random.default_rng(noise.seed * 7919 + scene.seed + 13)

    def detect(self, frame: np.ndarray | None, index: int = 0) -> list[Detection]:
        sc, nz, rng = self.scene, self.noise, self.rng
        out: list[Detection] = []
        for _id, box, vis, cls in sc.gt(index):
            if vis < 0.4 or rng.random() < nz.p_miss:
                continue
            w, h = box[2] - box[0], box[3] - box[1]
            jit = rng.normal(0, nz.jitter, 4) * np.array([w, h, w, h])
            b = (box[0] + jit[0], box[1] + jit[1], box[2] + jit[2], box[3] + jit[3])
            if b[2] - b[0] < 4 or b[3] - b[1] < 4:
                continue
            if rng.random() < nz.p_low:
                score = float(rng.uniform(0.2, 0.44))
            else:
                score = float(np.clip(rng.normal(0.78, 0.1), 0.46, 0.99))
            out.append(Detection(box=b, score=score, cls=cls))
        for _ in range(rng.poisson(nz.fp_per_frame)):
            cx, cy = rng.uniform(0, sc.width), rng.uniform(0.3 * sc.height, sc.height)
            fw, fh = rng.uniform(15, 60), rng.uniform(25, 90)
            out.append(
                Detection(
                    box=(cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2),
                    score=float(rng.uniform(0.15, 0.6)),
                    cls=str(rng.choice(["person", "car"])),
                )
            )
        return out


def render_video(scene: Scene, path: str, progress: Callable[[int], None] | None = None) -> None:
    from .video import WebmWriter

    with WebmWriter(path, scene.fps, (scene.width, scene.height)) as w:
        for i in range(scene.n_frames):
            w.write(scene.render(i))
            if progress:
                progress(i)
