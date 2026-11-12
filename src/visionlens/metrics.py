"""Multi-object tracking metrics: CLEAR MOT (MOTA, ID switches, fragmentation), IDF1, HOTA.

Inputs are dictionaries frame -> list of (track id, (x1, y1, x2, y2)). Implemented from the
definitions in Bernardin and Stiefelhagen (2008), Ristani et al. (2016) and Luiten et al. (2021);
hand-computed cases are in tests/test_metrics.py.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .track.iou import iou_matrix

Seq = dict[int, list[tuple[int, tuple[float, float, float, float]]]]


def _frames(gt: Seq, pred: Seq) -> list[int]:
    return sorted(set(gt) | set(pred))


def _arrays(items: list[tuple[int, tuple]]) -> tuple[list[int], np.ndarray]:
    return [i for i, _ in items], np.array([b for _, b in items], dtype=float).reshape(-1, 4)


def clear_mot(gt: Seq, pred: Seq, iou_thr: float = 0.5) -> dict[str, float]:
    """MOTA with the usual 'keep the previous match if it still overlaps' rule."""
    fn = fp = idsw = n_gt = matches_total = 0
    prev: dict[int, int] = {}  # gt id -> pred id of the last match
    last_tracked: dict[int, bool] = {}
    frag = 0
    traj_started: set[int] = set()
    for f in _frames(gt, pred):
        g_ids, g_boxes = _arrays(gt.get(f, []))
        p_ids, p_boxes = _arrays(pred.get(f, []))
        n_gt += len(g_ids)
        iou = iou_matrix(g_boxes, p_boxes)
        matched: dict[int, int] = {}
        used_p: set[int] = set()
        for gi, g in enumerate(g_ids):  # keep consistent matches first
            pid = prev.get(g)
            if pid is not None and pid in p_ids:
                pj = p_ids.index(pid)
                if iou[gi, pj] >= iou_thr and pj not in used_p:
                    matched[gi] = pj
                    used_p.add(pj)
        rest_g = [gi for gi in range(len(g_ids)) if gi not in matched]
        rest_p = [pj for pj in range(len(p_ids)) if pj not in used_p]
        if rest_g and rest_p:
            sub = iou[np.ix_(rest_g, rest_p)]
            rows, cols = linear_sum_assignment(-sub)
            for r, c in zip(rows, cols, strict=True):
                if sub[r, c] >= iou_thr:
                    matched[rest_g[r]] = rest_p[c]
                    used_p.add(rest_p[c])
        for gi, pj in matched.items():
            g, p = g_ids[gi], p_ids[pj]
            if g in prev and prev[g] != p:
                idsw += 1
            prev[g] = p
        matches_total += len(matched)
        fn += len(g_ids) - len(matched)
        fp += len(p_ids) - len(matched)
        for gi, g in enumerate(g_ids):
            tracked = gi in matched
            if g in traj_started and tracked and not last_tracked.get(g, False):
                frag += 1
            if tracked:
                traj_started.add(g)
            last_tracked[g] = tracked
    mota = 1.0 - (fn + fp + idsw) / n_gt if n_gt else 0.0
    return {
        "MOTA": mota,
        "IDsw": idsw,
        "FN": fn,
        "FP": fp,
        "Frag": frag,
        "GT": n_gt,
        "TP": matches_total,
    }


def _lengths_and_overlap(gt: Seq, pred: Seq, iou_thr: float) -> tuple[dict, dict, dict]:
    g_len: dict[int, int] = defaultdict(int)
    p_len: dict[int, int] = defaultdict(int)
    tp: dict[tuple[int, int], int] = defaultdict(int)
    for f in _frames(gt, pred):
        g_ids, g_boxes = _arrays(gt.get(f, []))
        p_ids, p_boxes = _arrays(pred.get(f, []))
        for g in g_ids:
            g_len[g] += 1
        for p in p_ids:
            p_len[p] += 1
        if g_ids and p_ids:
            iou = iou_matrix(g_boxes, p_boxes)
            for gi, pj in zip(*np.nonzero(iou >= iou_thr), strict=True):
                tp[(g_ids[gi], p_ids[pj])] += 1
    return g_len, p_len, tp


def idf1(gt: Seq, pred: Seq, iou_thr: float = 0.5) -> dict[str, float]:
    """Identity F1 from the single best one-to-one assignment of whole trajectories."""
    g_len, p_len, tp = _lengths_and_overlap(gt, pred, iou_thr)
    g_ids, p_ids = sorted(g_len), sorted(p_len)
    n_gt, n_pr = sum(g_len.values()), sum(p_len.values())
    idtp = 0
    if g_ids and p_ids:
        m = np.zeros((len(g_ids), len(p_ids)))
        gi = {g: i for i, g in enumerate(g_ids)}
        pi = {p: j for j, p in enumerate(p_ids)}
        for (g, p), v in tp.items():
            m[gi[g], pi[p]] = v
        rows, cols = linear_sum_assignment(-m)
        idtp = int(m[rows, cols].sum())
    idfn, idfp = n_gt - idtp, n_pr - idtp
    denom = 2 * idtp + idfp + idfn
    return {
        "IDF1": 2 * idtp / denom if denom else 0.0,
        "IDP": idtp / n_pr if n_pr else 0.0,
        "IDR": idtp / n_gt if n_gt else 0.0,
    }


def hota(gt: Seq, pred: Seq) -> dict[str, float]:
    """HOTA averaged over IoU thresholds 0.05 ... 0.95, following the TrackEval formulation."""
    frames = _frames(gt, pred)
    g_ids_all = sorted({g for f in frames for g, _ in gt.get(f, [])})
    p_ids_all = sorted({p for f in frames for p, _ in pred.get(f, [])})
    if not g_ids_all or not p_ids_all:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0}
    gi = {g: i for i, g in enumerate(g_ids_all)}
    pi = {p: j for j, p in enumerate(p_ids_all)}
    g_count = np.zeros(len(g_ids_all))
    p_count = np.zeros(len(p_ids_all))
    sims: list[tuple[list[int], list[int], np.ndarray]] = []
    potential = np.zeros((len(g_ids_all), len(p_ids_all)))
    for f in frames:
        g_ids, g_boxes = _arrays(gt.get(f, []))
        p_ids, p_boxes = _arrays(pred.get(f, []))
        gi_idx = [gi[g] for g in g_ids]
        pi_idx = [pi[p] for p in p_ids]
        for i in gi_idx:
            g_count[i] += 1
        for j in pi_idx:
            p_count[j] += 1
        s = iou_matrix(g_boxes, p_boxes)
        sims.append((gi_idx, pi_idx, s))
        if s.size:
            denom = s.sum(0, keepdims=True) + s.sum(1, keepdims=True) - s
            ratio = np.where(denom > 1e-12, s / np.maximum(denom, 1e-12), 0.0)
            for a, ga in enumerate(gi_idx):
                for b, pb in enumerate(pi_idx):
                    potential[ga, pb] += ratio[a, b]
    align = potential / np.maximum(g_count[:, None] + p_count[None, :] - potential, 1e-12)
    n_gt, n_pr = g_count.sum(), p_count.sum()
    alphas = np.arange(0.05, 0.96, 0.05)
    det_a, ass_a = [], []
    for alpha in alphas:
        counts = np.zeros_like(potential)
        tp = 0
        for gi_idx, pi_idx, s in sims:
            if not gi_idx or not pi_idx:
                continue
            score = align[np.ix_(gi_idx, pi_idx)] * s
            rows, cols = linear_sum_assignment(-score)
            for r, c in zip(rows, cols, strict=True):
                if s[r, c] >= alpha - 1e-10:
                    counts[gi_idx[r], pi_idx[c]] += 1
                    tp += 1
        fn, fp = n_gt - tp, n_pr - tp
        det_a.append(tp / max(tp + fn + fp, 1e-12))
        if tp:
            ass = counts / np.maximum(g_count[:, None] + p_count[None, :] - counts, 1e-12)
            ass_a.append(float((counts * ass).sum() / tp))
        else:
            ass_a.append(0.0)
    det, ass = float(np.mean(det_a)), float(np.mean(ass_a))
    return {
        "HOTA": float(np.mean(np.sqrt(np.array(det_a) * np.array(ass_a)))),
        "DetA": det,
        "AssA": ass,
    }


def evaluate(gt: Seq, pred: Seq) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(clear_mot(gt, pred))
    out.update(idf1(gt, pred))
    out.update(hota(gt, pred))
    return out
