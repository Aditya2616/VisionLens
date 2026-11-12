"""Reproducible measurements. One command writes every number the README quotes:

    visionlens bench --out results/results.json

Three groups: detector speed on this CPU, tracker quality on synthetic scenes with exact ground
truth (NOT a MOT-benchmark result), and end-to-end pipeline timing on a real clip. A fourth,
`describer`, measures how often the local language model's rewrite is rejected by the guard.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import time
from pathlib import Path

import numpy as np

from .metrics import evaluate
from .schemas import Detection
from .synth import SCENES, NoiseConfig, NoisyOracle, make_scene
from .sysutil import cpu_name, limit_threads
from .track import ByteTracker, TrackerConfig
from .track.iou import associate, iou_matrix

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "docs" / "demo" / "intersection.webm"

METRIC_KEYS = ("MOTA", "IDF1", "HOTA", "DetA", "AssA", "IDsw", "Frag")


# ------------------------------------------------------------------------------ machine
def machine_info(threads: int, note: str) -> dict:
    import cv2
    import torch
    import transformers

    return {
        "cpu": cpu_name(),
        "logical_cores": __import__("os").cpu_count(),
        "threads_used": threads,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "opencv": cv2.__version__,
        "os": platform.platform(),
        "date": dt.date.today().isoformat(),
        "load_note": note,
    }


# ----------------------------------------------------------------------------- detectors
def bench_detectors(
    models: list[tuple[str, int]], n_frames: int = 40, warmup: int = 3
) -> list[dict]:
    from .detect.hf import HFDetector
    from .video import iter_frames

    frames = [
        f
        for _, _, f in iter_frames(
            CLIP, max_width=640, target_fps=15, max_seconds=n_frames / 15 + 1
        )
    ]
    frames = frames[:n_frames]
    rows = []
    for model, size in models:
        det = HFDetector(model, input_size=size)
        params = sum(p.numel() for p in det.model.parameters()) / 1e6
        for f in frames[:warmup]:
            det.detect(f)
        ms = []
        n_dets = 0
        for f in frames:
            t0 = time.perf_counter()
            d = det.detect(f)
            ms.append((time.perf_counter() - t0) * 1000)
            n_dets += len(d)
        a = np.array(ms)
        rows.append(
            {
                "model": model,
                "input_size": size,
                "params_m": round(params, 1),
                "ms_median": round(float(np.median(a)), 1),
                "ms_p95": round(float(np.percentile(a, 95)), 1),
                "fps_median": round(1000 / float(np.median(a)), 2),
                "frames": len(a),
                "detections_per_frame": round(n_dets / len(a), 2),
            }
        )
        print("detector", rows[-1], flush=True)
    return rows


# --------------------------------------------------------------------------- tracking
class NaiveIoUTracker:
    """Baseline: match each detection to the previous frame's box by IoU, no motion model, no
    confirmation delay, no low-score pass. Born on the first sighting, dropped after 1 miss."""

    def __init__(self, thr: float = 0.3) -> None:
        self.thr = thr
        self.tracks: list[tuple[int, tuple, str]] = []
        self.next_id = 1

    def step(self, dets: list[Detection] | None, frame_idx: int = 0):
        from .schemas import TrackOut

        if dets is None:
            return [
                TrackOut(id=i, box=b, cls=c, score=1.0, detected=False) for i, b, c in self.tracks
            ]
        dets = [d for d in dets if d.score >= 0.3]
        sim = iou_matrix(np.array([t[1] for t in self.tracks]), np.array([d.box for d in dets]))
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(dets):
                if d.cls != t[2]:
                    sim[i, j] = 0
        m, _, un_d = associate(sim, self.thr)
        new = []
        for ti, di in m:
            new.append((self.tracks[ti][0], dets[di].box, dets[di].cls))
        for j in un_d:
            new.append((self.next_id, dets[j].box, dets[j].cls))
            self.next_id += 1
        self.tracks = new
        return [TrackOut(id=i, box=b, cls=c, score=1.0, detected=True) for i, b, c in new]


def run_tracker(scene, level: int, stride: int, tracker) -> dict:
    orc = NoisyOracle(scene, NoiseConfig.level(level, seed=scene.seed))
    pred = {}
    for i in range(scene.n_frames):
        dets = orc.detect(None, i) if i % stride == 0 else None
        pred[i] = [(t.id, t.box) for t in tracker.step(dets, i)]
    return evaluate(scene.gt_sequence(), pred)


def _mean(rows: list[dict]) -> dict:
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in METRIC_KEYS}


def bench_tracker(seeds: int = 3, seconds: float = 10.0) -> dict:
    scenes = [make_scene(n, s, seconds=seconds) for n in SCENES for s in range(seeds)]
    gt_boxes = sum(len(v) for sc in scenes for v in sc.gt_sequence().values())
    out: dict = {
        "scenes": SCENES,
        "seeds": seeds,
        "seconds_per_scene": seconds,
        "gt_boxes_total": gt_boxes,
        "fps": 15.0,
        "noise_levels": {str(k): vars(NoiseConfig.level(k)) for k in range(4)},
    }

    def agg(make, level, stride):
        rows = [run_tracker(sc, level, stride, make()) for sc in scenes]
        return _mean(rows)

    out["noise_sweep"] = [
        {"level": lv, "stride": 3, **agg(lambda: ByteTracker(), lv, 3)} for lv in range(4)
    ]
    out["stride_sweep"] = [
        {"level": 2, "stride": st, **agg(lambda: ByteTracker(), 2, st)} for st in (1, 2, 3, 5, 8)
    ]
    variants = {
        "ByteTrack-style (default)": lambda: ByteTracker(TrackerConfig()),
        "no low-score second pass": lambda: ByteTracker(TrackerConfig(two_stage=False)),
        "+ observation-centric re-update": lambda: ByteTracker(TrackerConfig(ocr_update=True)),
        "no class gating": lambda: ByteTracker(TrackerConfig(class_gate=False)),
        "no confirmation delay (min_hits=1)": lambda: ByteTracker(TrackerConfig(min_hits=1)),
        "naive IoU baseline (no motion model)": lambda: NaiveIoUTracker(),
    }
    out["ablation"] = [
        {"variant": name, "level": 2, "stride": 3, **agg(mk, 2, 3)} for name, mk in variants.items()
    ]
    out["per_scene"] = []
    for name in SCENES:
        rows = [run_tracker(sc, 2, 3, ByteTracker()) for sc in scenes if sc.name == name]
        out["per_scene"].append({"scene": name, "level": 2, "stride": 3, **_mean(rows)})
    for r in out["noise_sweep"]:
        print("tracker noise", r, flush=True)
    return out


# --------------------------------------------------------------------------- end to end
def bench_pipeline(stride: int = 3, model: str = "dfine-nano", input_size: int = 640) -> dict:
    from .detect.hf import HFDetector
    from .pipeline import PipelineConfig
    from .runner import analyze_file

    det = HFDetector(model, input_size=input_size)
    res = analyze_file(CLIP, det, PipelineConfig(stride=stride), threads=None)
    t = res.timings
    return {
        "clip": CLIP.name,
        "frames": res.meta.n_frames,
        "fps_video": res.meta.fps,
        "resolution": f"{res.meta.width}x{res.meta.height}",
        "detector": model,
        "input_size": input_size,
        "stride": stride,
        "detect_calls": t.detect_calls,
        "detect_ms_mean": t.detect_ms_mean,
        "detect_ms_p95": t.detect_ms_p95,
        "track_ms_mean": t.track_ms_mean,
        "wall_s": t.wall_s,
        "fps_end_to_end": t.fps_end_to_end,
        "tracks": res.summary["tracks"],
        "events": len(res.events),
    }


# ----------------------------------------------------------------------------- describer
def bench_describer(model_id: str | None = None) -> dict:
    """Run the local LLM rewrite on several fact sets and count guard rejections."""
    from .describe import Facts, TemplateDescriber, build_description
    from .describe.llm import LocalLLMDescriber
    from .events import extract_events, summarize
    from .pipeline import Pipeline, PipelineConfig

    llm = LocalLLMDescriber(model_id)
    cases = []
    for name in SCENES:
        for seed in (0, 1):
            sc = make_scene(name, seed)
            p = Pipeline(NoisyOracle(sc, NoiseConfig.level(1, seed)), PipelineConfig(stride=2))
            for _ in p.process((i, i / sc.fps, None) for i in range(sc.n_frames)):
                pass
            ev = extract_events(p.frames, sc.width, sc.height, sc.fps)
            cases.append(
                (
                    f"synthetic-{name}-{seed}",
                    Facts(sc.duration, sc.width, sc.height, ev, summarize(p.frames, sc.fps)),
                )
            )
    rows = []
    for name, facts in cases:
        tpl = TemplateDescriber().describe(facts)
        t0 = time.perf_counter()
        d = build_description(facts, tpl, llm)
        mv = d.model
        assert mv is not None
        rows.append(
            {
                "case": name,
                "accepted": mv.accepted,
                "reason": mv.rejected_reason,
                "seconds": round(time.perf_counter() - t0, 1),
                "chars": len(mv.text),
                "text": mv.text,
            }
        )
        print("describer", name, mv.accepted, mv.rejected_reason, flush=True)
    rej = sum(1 for r in rows if not r["accepted"])
    return {
        "model": llm.model_id,
        "cases": len(rows),
        "rejected": rej,
        "accepted": len(rows) - rej,
        "rejection_rate": round(rej / len(rows), 3),
        "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 1),
        "rows": rows,
    }


# ---------------------------------------------------------------------------------- main
def run_all(
    out: str | Path, threads: int | None, note: str, quick: bool = False, skip: tuple[str, ...] = ()
) -> dict:
    n_threads = limit_threads(threads)
    path = Path(out)
    results: dict = json.loads(path.read_text("utf-8")) if path.exists() else {}
    results["machine"] = machine_info(n_threads, note)

    def save() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), "utf-8")

    if "detectors" not in skip:
        models = [
            ("dfine-nano", 640),
            ("dfine-nano", 480),
            ("dfine-nano", 320),
            ("dfine-small", 640),
            ("rtdetr-r18", 640),
        ]
        results["detectors"] = bench_detectors(
            models[:2] if quick else models, n_frames=15 if quick else 40
        )
        save()
    if "tracker" not in skip:
        results["tracker"] = bench_tracker(seeds=1 if quick else 3)
        save()
    if "pipeline" not in skip:
        results["pipeline"] = bench_pipeline()
        save()
    if "describer" not in skip:
        results["describer"] = bench_describer()
        save()
    save()
    return results
