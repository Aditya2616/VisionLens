"""Turn per-frame tracks into typed events: enter, leave, move, linger, count_change, proximity.

Everything here is deterministic and is the source of truth for the description: the template
describer prints these events, and the language model is only allowed to rephrase them.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .schemas import Event, FrameResult

Box = tuple[float, float, float, float]


@dataclass
class EventConfig:
    edge_margin: float = (
        0.12  # box this close to an edge (fraction of the dimension) counts as edge
    )
    min_life_s: float = 0.8  # tracks shorter than this are ignored (flicker)
    move_min_disp: float = 0.06  # net displacement (fraction of frame width) to call it moving
    linger_s: float = 1.5  # stay slow this long to count as lingering
    linger_speed: float = 0.02  # slower than this (frame widths per second) is "slow"
    count_stable_s: float = 0.8  # a count must hold this long before it is reported
    prox_gap: float = 0.3  # boxes closer than this fraction of the smaller box size are "close"
    prox_feet: float = 1.0  # and their feet (box bottoms) closer than this many box heights
    prox_s: float = 0.6  # ... for at least this long
    prox_merge_s: float = 0.6  # merge close periods separated by less than this


def _center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _gap(a: Box, b: Box) -> float:
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def _edge(box: Box, W: int, H: int, margin: float) -> str | None:
    d = {
        "left": box[0] / W,
        "right": (W - box[2]) / W,
        "top": box[1] / H,
        "bottom": (H - box[3]) / H,
    }
    side, val = min(d.items(), key=lambda kv: kv[1])
    return side if val <= margin else None


def direction_name(dx: float, dy: float) -> str:
    if abs(dx) >= 1.2 * abs(dy):
        return "right" if dx > 0 else "left"
    if abs(dy) >= 1.2 * abs(dx):
        return "down" if dy > 0 else "up"
    return ("down-" if dy > 0 else "up-") + ("right" if dx > 0 else "left")


def extract_events(
    frames: list[FrameResult], width: int, height: int, fps: float, cfg: EventConfig | None = None
) -> list[Event]:
    cfg = cfg or EventConfig()
    if len(frames) < 2:
        return []
    n = len(frames)
    series: dict[int, list[tuple[int, Box, str, bool]]] = defaultdict(list)
    for fr in frames:
        for t in fr.tracks:
            series[t.id].append((fr.i, t.box, t.cls, t.detected))
    events: list[Event] = []

    def add(
        kind: str,
        t: float,
        ids: list[int],
        classes: list[str],
        pos: tuple[float, float] | None,
        t_end: float | None = None,
        **data: object,
    ) -> None:
        events.append(
            Event(
                id="",
                t=round(t, 2),
                kind=kind,
                ids=ids,
                classes=classes,  # type: ignore[arg-type]
                pos=(round(pos[0], 1), round(pos[1], 1)) if pos else None,
                t_end=round(t_end, 2) if t_end is not None else None,
                data=dict(data),
            )
        )

    life: dict[int, tuple[int, int]] = {}
    cls_of: dict[int, str] = {}
    for tid, pts in series.items():
        det_frames = [p[0] for p in pts if p[3]]
        if not det_frames:
            continue
        i0, i1 = det_frames[0], det_frames[-1]
        if (i1 - i0) / fps < cfg.min_life_s and not (i0 == 0 and i1 >= n - 2):
            continue
        life[tid] = (i0, i1)
        cls_of[tid] = Counter(p[2] for p in pts).most_common(1)[0][0]

    for tid, (i0, i1) in life.items():
        pts = {p[0]: p for p in series[tid]}
        first_box, last_box = pts[i0][1], pts[i1][1]
        cls = cls_of[tid]
        t0, t1 = frames[i0].t, frames[i1].t
        (cx0, cy0), (cx1, cy1) = _center(first_box), _center(last_box)
        disp = math.hypot(cx1 - cx0, cy1 - cy0) / width
        tol = max(1, int(round(0.8 * fps)))
        from_start = i0 <= 1 or (i0 <= tol and disp < cfg.move_min_disp)
        to_end = i1 >= n - 2
        if not from_start:
            add(
                "enter",
                t0,
                [tid],
                [cls],
                _center(first_box),
                edge=_edge(first_box, width, height, cfg.edge_margin),
            )
        if not to_end:
            add(
                "leave",
                t1,
                [tid],
                [cls],
                _center(last_box),
                edge=_edge(last_box, width, height, cfg.edge_margin),
            )
        # movement summary over the whole stay
        dur = max(t1 - t0, 1e-6)
        speed = disp / dur
        if disp >= cfg.move_min_disp and dur >= cfg.min_life_s:
            bucket = "slowly" if speed < 0.05 else ("at a steady pace" if speed < 0.15 else "fast")
            add(
                "move",
                t0 if not from_start else t0,
                [tid],
                [cls],
                (cx0, cy0),
                t_end=t1,
                direction=direction_name(cx1 - cx0, cy1 - cy0),
                speed=bucket,
                speed_value=round(speed, 3),
                from_start=from_start,
            )
        # lingering: slow for a sustained time, using a ~0.7 s window
        win = max(2, int(round(0.7 * fps)))
        idxs = sorted(pts)
        slow = []
        for k, i in enumerate(idxs):
            j = idxs[max(0, k - win)]
            if i == j:
                slow.append(False)
                continue
            (ax, ay), (bx, by) = _center(pts[j][1]), _center(pts[i][1])
            v = math.hypot(bx - ax, by - ay) / width / ((i - j) / fps)
            slow.append(v < cfg.linger_speed)
        gap_fill = max(1, int(round(0.8 * fps)))
        k = 0
        while k < len(idxs):  # bridge short fast gaps (detector jitter) between slow runs
            if not slow[k]:
                m = k
                while m < len(idxs) and not slow[m]:
                    m += 1
                if 0 < k and m < len(idxs) and idxs[m] - idxs[k] <= gap_fill:
                    for q in range(k, m):
                        slow[q] = True
                k = m
            else:
                k += 1
        k = 0
        while k < len(idxs):
            if not slow[k]:
                k += 1
                continue
            m = k
            while m + 1 < len(idxs) and slow[m + 1]:
                m += 1
            ta, tb = frames[idxs[k]].t, frames[idxs[m]].t
            if tb - ta >= cfg.linger_s:
                whole = (tb - ta) >= 0.9 * max(t1 - t0, 1e-6)
                add(
                    "linger",
                    ta,
                    [tid],
                    [cls],
                    _center(pts[idxs[k]][1]),
                    t_end=tb,
                    duration=round(tb - ta, 1),
                    whole_stay=whole,
                )
            k = m + 1

    # count changes (debounced)
    counts: list[tuple[int, Counter]] = []
    for fr in frames:
        c = Counter(t.cls for t in fr.tracks if t.id in life)
        counts.append((sum(c.values()), c))
    need = max(1, int(round(cfg.count_stable_s * fps)))
    reported = counts[0][0]
    i = 1
    while i < n:
        v = counts[i][0]
        if v != reported:
            j = i
            while j < n and counts[j][0] == v:
                j += 1
            if j - i >= need or j >= n:
                add(
                    "count_change",
                    frames[i].t,
                    [],
                    [],
                    None,
                    previous=reported,
                    count=v,
                    by_class=dict(counts[i][1]),
                )
                reported = v
            i = j
        else:
            i += 1

    # proximity between pairs
    ids = sorted(life)
    for a_i, a in enumerate(ids):
        pa = {p[0]: p[1] for p in series[a]}
        for b in ids[a_i + 1 :]:
            pb = {p[0]: p[1] for p in series[b]}
            common = sorted(set(pa) & set(pb))
            if not common:
                continue
            close = []
            for f in common:
                size = min(
                    min(pa[f][2] - pa[f][0], pa[f][3] - pa[f][1]),
                    min(pb[f][2] - pb[f][0], pb[f][3] - pb[f][1]),
                )
                fa, fb = (
                    ((pa[f][0] + pa[f][2]) / 2, pa[f][3]),
                    ((pb[f][0] + pb[f][2]) / 2, pb[f][3]),
                )
                feet = math.hypot(fa[0] - fb[0], fa[1] - fb[1])
                hmin = min(pa[f][3] - pa[f][1], pb[f][3] - pb[f][1])
                close.append(
                    _gap(pa[f], pb[f]) <= cfg.prox_gap * max(size, 1.0)
                    and feet <= cfg.prox_feet * max(hmin, 1.0)
                )
            periods: list[list[int]] = []
            for f, c in zip(common, close, strict=True):
                if c:
                    if periods and (f - periods[-1][1]) / fps <= cfg.prox_merge_s:
                        periods[-1][1] = f
                    else:
                        periods.append([f, f])
            for f0, f1 in periods:
                if (f1 - f0) / fps >= cfg.prox_s:
                    ca, cb = _center(pa[f0]), _center(pb[f0])
                    add(
                        "proximity",
                        frames[f0].t,
                        [a, b],
                        [cls_of[a], cls_of[b]],
                        ((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2),
                        t_end=frames[f1].t,
                        duration=round(frames[f1].t - frames[f0].t, 1),
                    )

    events.sort(key=lambda e: (e.t, e.kind))
    for k, e in enumerate(events):
        e.id = f"e{k}"
    return events


def summarize(frames: list[FrameResult], fps: float, cfg: EventConfig | None = None) -> dict:
    """Track counts used by the describers and the UI (same life filter as the events)."""
    cfg = cfg or EventConfig()
    span: dict[int, list[int]] = {}
    votes: dict[int, Counter] = defaultdict(Counter)
    n = len(frames)
    peak = 0
    for fr in frames:
        for t in fr.tracks:
            votes[t.id][t.cls] += 1
            if t.detected:
                s = span.setdefault(t.id, [fr.i, fr.i])
                s[1] = fr.i
    by_class: Counter = Counter()
    kept = set()
    for tid, (a, b) in span.items():
        if (b - a) / fps >= cfg.min_life_s or (a == 0 and b >= n - 2):
            kept.add(tid)
            by_class[votes[tid].most_common(1)[0][0]] += 1
    for fr in frames:
        peak = max(peak, sum(1 for t in fr.tracks if t.id in kept))
    return {"tracks": len(kept), "by_class": dict(by_class), "peak": peak}
