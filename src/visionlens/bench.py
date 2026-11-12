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
from typing import TYPE_CHECKING

import numpy as np

from .metrics import evaluate
from .schemas import Detection
from .synth import SCENES, NoiseConfig, NoisyOracle, make_scene
from .sysutil import cpu_name, gpu_name, limit_threads, observed_load, resolve_device
from .track import ByteTracker, TrackerConfig
from .track.iou import associate, iou_matrix

if TYPE_CHECKING:
    from .describe import Facts

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
        "gpu": gpu_name(),
    }


# ----------------------------------------------------------------------------- detectors
def bench_detectors(
    models: list[tuple[str, int]], n_frames: int = 40, warmup: int = 3, device: str = "cpu"
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
        det = HFDetector(model, input_size=size, device=device)
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
                "device": det.device,
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
def bench_pipeline(
    stride: int = 3, model: str = "dfine-nano", input_size: int = 640, device: str = "cpu"
) -> dict:
    from .detect.hf import HFDetector
    from .pipeline import PipelineConfig
    from .runner import analyze_file

    det = HFDetector(model, input_size=input_size, device=device)
    res = analyze_file(CLIP, det, PipelineConfig(stride=stride), threads=None)
    t = res.timings
    return {
        "device": det.device,
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
DESCRIBER_MODELS = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")


def _describer_cases() -> list[tuple[str, Facts]]:
    """All 12 synthetic scenes (4 types x 3 seeds, simulated detector) and the 3 recorded clips."""
    from .describe import Facts, TemplateDescriber
    from .events import extract_events, summarize
    from .pipeline import Pipeline, PipelineConfig
    from .schemas import Result

    cases = []
    for name in SCENES:
        for seed in range(3):
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
    for clip in ("intersection", "traffic", "synthetic"):
        path = ROOT / "docs" / "demo" / f"{clip}.json"
        if path.exists():
            r = Result.model_validate_json(path.read_text("utf-8"))
            cases.append(
                (
                    f"recorded-{clip}",
                    Facts(r.meta.duration, r.meta.width, r.meta.height, r.events, r.summary),
                )
            )
    for _, facts in cases:
        TemplateDescriber().describe(facts)
    return cases


def _run_model(model_id: str, device: str | None, cases, label: str) -> dict:
    from .describe import build_description
    from .describe.llm import LocalLLMDescriber
    from .describe.template import TemplateDescriber

    llm = LocalLLMDescriber(model_id, device=device)
    rows = []
    for name, facts in cases:
        tpl = TemplateDescriber().describe(facts)
        d = build_description(facts, tpl, llm)
        mv = d.model
        assert mv is not None
        rows.append(
            {
                "case": name,
                "accepted": mv.accepted,
                "reason": mv.rejected_reason,
                "seconds": mv.seconds,
                "chars": len(mv.text),
                "text": mv.text,
                "template": tpl.text,
            }
        )
        print(label, model_id, name, mv.accepted, mv.rejected_reason, mv.seconds, flush=True)
    acc = sum(1 for r in rows if r["accepted"])
    return {
        "model": model_id,
        "device": llm.device,
        "dtype": llm.dtype,
        "cases": len(rows),
        "accepted": acc,
        "rejected": len(rows) - acc,
        "acceptance_rate": round(acc / len(rows), 3),
        "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 2),
        "rows": rows,
    }


def bench_describer(device: str | None = None, cpu_cases: int = 3) -> dict:
    """Rewrite the grouped template facts with each local model and count guard rejections.

    The full set runs on the GPU in float16 when there is one. A few cases are also timed on the
    CPU in float32 (threads as limited by the caller) because that is what a CPU-only user gets.
    """
    cases = _describer_cases()
    out: dict = {"cases": [n for n, _ in cases], "models": []}
    for model_id in DESCRIBER_MODELS:
        r = _run_model(model_id, device, cases, "describer")
        if r["device"] != "cpu":
            sub = _run_model(model_id, "cpu", cases[:cpu_cases], "describer-cpu")
            r["cpu"] = {k: sub[k] for k in ("device", "dtype", "cases", "mean_seconds")}
            r["cpu"]["load"] = observed_load()
        out["models"].append(r)
    return out


# ---------------------------------------------------------------------------------- main
def run_all(
    out: str | Path,
    threads: int | None,
    note: str,
    quick: bool = False,
    skip: tuple[str, ...] = (),
    gpu_device: str = "auto",
) -> dict:
    n_threads = limit_threads(threads)
    gpu = resolve_device(gpu_device)
    path = Path(out)
    results: dict = json.loads(path.read_text("utf-8")) if path.exists() else {}
    if "machine" not in results or {"detectors", "pipeline"} - set(skip):
        results["machine"] = machine_info(n_threads, note)
    loads: dict = results.setdefault("load_observed", {})

    def track_load(group: str, before: dict | None, after: dict | None = None) -> None:
        loads[group] = {"before": before, "after": after or observed_load(0.5)}

    def save() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), "utf-8")

    if "detectors" not in skip:
        before = observed_load()
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
        if gpu != "cpu":
            # A second, labelled column: the same models on the GPU (CPU threads kept low).
            results["detectors_gpu"] = bench_detectors(
                models[:2] if quick else models, n_frames=15 if quick else 40, device=gpu
            )
        track_load("detectors", before)
        save()
    if "tracker" not in skip:
        results["tracker"] = bench_tracker(seeds=1 if quick else 3)
        save()
    if "pipeline" not in skip:
        before = observed_load()
        results["pipeline"] = bench_pipeline(device="cpu")
        if gpu != "cpu":
            results["pipeline_gpu"] = bench_pipeline(device=gpu)
        track_load("pipeline", before)
        save()
    if "describer" not in skip:
        before = observed_load()
        results["describer"] = bench_describer(device=gpu_device)
        track_load("describer", before)
        save()
    save()
    return results
