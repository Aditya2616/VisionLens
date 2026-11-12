"""Refresh the measured numbers in README.md from results/results.json.

    visionlens bench --out results/results.json     # measure (one command)
    python scripts/update_readme.py                  # rewrite the tables in the README

Only the blocks between `<!-- RESULTS:name:start -->` and `<!-- RESULTS:name:end -->` change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def blocks(res: dict) -> dict[str, str]:
    m = res["machine"]
    out: dict[str, str] = {}
    out["machine"] = (
        f"Measured on {m['cpu']} ({m['logical_cores']} logical cores; {m['threads_used']} threads used), "
        f"PyTorch {m['torch']} (CPU rows forced to the CPU), transformers {m['transformers']}, OpenCV {m['opencv']}, "
        f"Python {m['python']}, {m['date']}. Load: {m['load_note'] or 'not recorded'}"
    )
    lo = res.get("load_observed", {})

    def busy(group: str) -> str:
        v = lo.get(group) or {}
        a = (v.get("before") or {}).get("cpu_busy_percent")
        b = (v.get("after") or {}).get("cpu_busy_percent")
        if a is None or b is None:
            return "CPU load not recorded for this group"
        return (
            f"measured with the CPU about {(a + b) / 2:.0f}% busy with other work "
            f"(sampled {a:.0f}% before, {b:.0f}% after); an idle machine will be faster"
        )

    if lo:

        def fmt(x: dict | None) -> str:
            x = x or {}
            bits = []
            if "cpu_busy_percent" in x:
                bits.append(f"CPU {x['cpu_busy_percent']:.0f}% busy")
            if "gpu_busy_percent" in x:
                bits.append(
                    f"GPU {x['gpu_busy_percent']}% busy, {x['gpu_mem_used_mb']} of {x['gpu_mem_total_mb']} MB used"
                )
            return ", ".join(bits) or "not recorded"

        out["load"] = "; ".join(
            f"{g}: {fmt(v.get('before'))} before, {fmt(v.get('after'))} after"
            for g, v in lo.items()
        )

    def gpu_busy(group: str) -> str:
        v = lo.get(group) or {}
        a = (v.get("before") or {}).get("gpu_busy_percent")
        b = (v.get("after") or {}).get("gpu_busy_percent")
        if a is None or b is None:
            return "GPU load not recorded"
        return f"GPU about {(a + b) / 2:.0f}% busy with other work"

    if "detectors" in res:
        gpu = {(d["model"], d["input_size"]): d for d in res.get("detectors_gpu", [])}
        rows = []
        for d in res["detectors"]:
            g = gpu.get((d["model"], d["input_size"]))
            rows.append(
                [
                    d["model"],
                    d["input_size"],
                    d["params_m"],
                    d["ms_median"],
                    d["ms_p95"],
                    d["fps_median"],
                    f"{g['ms_median']} ms / {g['fps_median']}" if g else "not run",
                    d["detections_per_frame"],
                ]
            )
        out["detectors"] = (
            table(
                [
                    "Detector",
                    "Input px",
                    "Params (M)",
                    "CPU median ms",
                    "CPU p95 ms",
                    "CPU FPS",
                    "GPU median ms / FPS",
                    "Dets/frame",
                ],
                rows,
            )
            + "\n\n"
            + f"One detector call on a 480x270 frame (resized to the input size), batch 1, {res['detectors'][0]['frames']} frames, per-call time includes pre- and post-processing. "
            f"CPU columns: {res['machine']['threads_used']} threads, float32, {busy('detectors')}. GPU column: {res['machine'].get('gpu') or 'n/a'}, float32, {gpu_busy('detectors')}."
        )
    if "pipeline" in res:
        prow = []
        for label, p in (("CPU", res["pipeline"]), ("GPU", res.get("pipeline_gpu"))):
            if not p:
                continue
            prow.append(
                [
                    label,
                    p["clip"],
                    p["frames"],
                    p["detector"],
                    f"{p['input_size']}px",
                    p["stride"],
                    p["detect_calls"],
                    f"{p['detect_ms_mean']} / {p['detect_ms_p95']}",
                    p["track_ms_mean"],
                    p["fps_end_to_end"],
                ]
            )
        out["pipeline"] = (
            f"CPU row: {res['machine']['threads_used']} threads, {busy('pipeline')}. "
            f"GPU row: float32 detector, {gpu_busy('pipeline')}.\n\n"
        ) + table(
            [
                "Detector on",
                "Clip",
                "Frames",
                "Detector",
                "Input",
                "Stride",
                "Detector calls",
                "Detect ms (mean / p95)",
                "Track ms (mean)",
                "End-to-end FPS",
            ],
            prow,
        )
    if "tracker" in res:
        t = res["tracker"]
        keys = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "IDsw"]
        out["tracker_setup"] = (
            f"{len(t['scenes'])} scene types x {t['seeds']} seeds, {t['seconds_per_scene']:.0f} s at "
            f"{t['fps']:.0f} fps, {t['gt_boxes_total']} ground-truth boxes in total. "
            "Detections are the exact ground truth plus injected noise (see `NoiseConfig`); "
            "these are **synthetic-scene results, not MOT-benchmark results**."
        )
        out["noise_sweep"] = table(
            ["Noise level", "Stride"] + keys,
            [[r["level"], r["stride"]] + [r[k] for k in keys] for r in t["noise_sweep"]],
        )
        out["stride_sweep"] = table(
            ["Noise level", "Detector stride"] + keys,
            [[r["level"], r["stride"]] + [r[k] for k in keys] for r in t["stride_sweep"]],
        )
        out["ablation"] = table(
            ["Variant", "Noise level", "Stride"] + keys,
            [[r["variant"], r["level"], r["stride"]] + [r[k] for k in keys] for r in t["ablation"]],
        )
        out["per_scene"] = table(
            ["Scene", "Noise level", "Stride"] + keys,
            [[r["scene"], r["level"], r["stride"]] + [r[k] for k in keys] for r in t["per_scene"]],
        )
    if "describer" in res:
        d = res["describer"]
        rows = []
        for m in d["models"]:
            cpu = m.get("cpu")
            load = (cpu or {}).get("load", {}).get("cpu_busy_percent")
            rows.append(
                [
                    m["model"],
                    f"{m['accepted']} / {m['cases']} ({m['acceptance_rate']:.0%})",
                    f"{m['mean_seconds']} s ({m['device']}, {m['dtype']})",
                    f"{cpu['mean_seconds']} s ({cpu['device']}, {cpu['dtype']}, {cpu['cases']} cases, "
                    f"CPU {load}% busy)"
                    if cpu
                    else "not run",
                ]
            )
        out["describer"] = (
            f"{len(d['cases'])} cases: all 12 synthetic scenes (4 types x 3 seeds, simulated detector) "
            "plus the 3 recorded clips, rewritten from the grouped template facts with greedy decoding.\n\n"
            + table(
                [
                    "Model",
                    "Guard accepted",
                    "Mean time per rewrite, GPU",
                    "Mean time per rewrite, CPU",
                ],
                rows,
            )
        )
        reasons: dict[str, int] = {}

        def category(why: str) -> str:
            if "count" in why or why.startswith("number"):
                return "wrong or invented count"
            if "disagrees" in why:
                return "direction contradicts the events"
            if "not a tracked object" in why:
                return "object that was not tracked"
            if why.startswith("time") or "duration" in why:
                return "time or duration not in the events"
            if why.startswith("claims"):
                return "event verb not supported by the events"
            return why

        for m in d["models"]:
            for r in m["rows"]:
                if not r["accepted"]:
                    key = category(r["reason"] or "")
                    reasons[key] = reasons.get(key, 0) + 1
        out["describer_reasons"] = ", ".join(
            f"{n} x {k}" for k, n in sorted(reasons.items(), key=lambda kv: -kv[1])
        )
    return out


def main() -> None:
    res = json.loads((ROOT / "results" / "results.json").read_text("utf-8"))
    path = ROOT / "README.md"
    text = path.read_text("utf-8")
    for name, body in blocks(res).items():
        pat = re.compile(
            rf"(<!-- RESULTS:{name}:start -->)(.*?)(<!-- RESULTS:{name}:end -->)", re.S
        )
        if not pat.search(text):
            print("no marker for", name)
            continue
        text = pat.sub(lambda m, b=body: f"{m.group(1)}\n{b}\n{m.group(3)}", text)
    path.write_text(text, "utf-8")
    print("README updated")


if __name__ == "__main__":
    main()
