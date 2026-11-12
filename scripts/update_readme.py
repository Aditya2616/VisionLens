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
        f"CPU-only PyTorch {m['torch']}, transformers {m['transformers']}, OpenCV {m['opencv']}, "
        f"Python {m['python']}, {m['date']}. Load: {m['load_note'] or 'not recorded'}"
    )
    if "detectors" in res:
        rows = [
            [
                d["model"],
                d["input_size"],
                d["params_m"],
                d["ms_median"],
                d["ms_p95"],
                d["fps_median"],
                d["detections_per_frame"],
            ]
            for d in res["detectors"]
        ]
        out["detectors"] = (
            table(
                ["Detector", "Input px", "Params (M)", "Median ms", "p95 ms", "FPS", "Dets/frame"],
                rows,
            )
            + f"\n\nOne detector call on a 480x270 frame (resized to the input size), batch 1, {res['detectors'][0]['frames']} frames, per-call time includes pre- and post-processing."
        )
    if "pipeline" in res:
        p = res["pipeline"]
        out["pipeline"] = table(
            [
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
            [
                [
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
            ],
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
        out["describer"] = (
            f"`{d['model']}` rewrote the template facts of {d['cases']} synthetic scenes "
            f"(greedy decoding, {d['mean_seconds']} s each on average). The faithfulness guard "
            f"accepted {d['accepted']} and rejected {d['rejected']} "
            f"(rejection rate {d['rejection_rate']:.0%}). Rejected rewrites fall back to the "
            "template text."
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
