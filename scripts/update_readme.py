"""Keep the numbers in README.md identical to the files they come from.

    visionlens bench --out results/results.json     # measure (one command)
    python scripts/update_readme.py                  # rewrite the generated blocks of the README
    python scripts/update_readme.py --check          # exit 1 if the README is out of date (also a test)

Every number the README quotes lives between `<!-- RESULTS:name:start -->` and
`<!-- RESULTS:name:end -->` markers and is generated from one of:

- results/results.json            the benchmark run (tables, load, tracker and rewrite figures)
- data/facts.json                 model licences, parameters, download sizes, published AP
                                  (written by scripts/fetch_facts.py)
- data/samples.json               the sample-clip credits
- docs/demo/index.json + <clip>.json   the recorded demo the web page replays
- the code (api.Settings limits, the cli fetch-models list)

`--check` fails when a marker is missing, when a block differs from what the sources say, or when
a block's source is absent. tests/test_readme_facts.py runs it and adds checks for the facts that
sit in the prose (commands, flags, versions, file names).
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LICENCE_NAMES = {"apache-2.0": "Apache-2.0", "mit": "MIT"}


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def size_text(n_bytes: float) -> str:
    mb = n_bytes / 1e6
    return f"{mb / 1000:.1f} GB" if mb >= 995 else f"{mb:.0f} MB"


def _busy(lo: dict, group: str, key: str) -> tuple[float, float] | None:
    v = lo.get(group) or {}
    a = (v.get("before") or {}).get(key)
    b = (v.get("after") or {}).get(key)
    return None if a is None or b is None else (a, b)


def cpu_busy_text(lo: dict, group: str) -> str:
    ab = _busy(lo, group, "cpu_busy_percent")
    if not ab:
        return "CPU load not recorded for this group"
    return (
        f"measured with the CPU about {sum(ab) / 2:.0f}% busy with other work "
        f"(sampled {ab[0]:.0f}% before, {ab[1]:.0f}% after); an idle machine will be faster"
    )


def gpu_busy_text(lo: dict, group: str) -> str:
    ab = _busy(lo, group, "gpu_busy_percent")
    if not ab:
        return "GPU load not recorded"
    return f"GPU about {sum(ab) / 2:.0f}% busy with other work"


def _category(why: str) -> str:
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


def best_model(res: dict) -> dict:
    """The rewrite model chosen by evidence: highest guard acceptance after retries, then fewest
    attempts, then the lower CPU time."""
    models = res["describer"]["models"]
    return max(
        models,
        key=lambda m: (
            m["acceptance_rate"],
            -m["mean_attempts"],
            -(m.get("cpu") or {}).get("mean_seconds", 1e9),
        ),
    )


ACCEPT_MIN = 0.9  # a rewrite model must pass the guard on at least this share of cases


def decide_default(res: dict) -> dict:
    """Which describer is the default, by the evidence in results.json.

    The rewrite is the default (where a GPU is available) only if the best model is accepted by the
    guard on at least 90% of the cases after retries AND its shown text reads at least as well as
    the template on the objective measures: no more repeated openings and no more words. Otherwise
    the template stays the default and the rewrite is an option.
    """
    d = res["describer"]
    win = best_model(res)
    tpl = d["template"]["readability"]
    got = win["readability_shown"]
    checks = {
        "accepted": win["acceptance_rate"] >= ACCEPT_MIN,
        "openings": got["repeated_openings"] <= tpl["repeated_openings"],
        "words": got["words"] <= tpl["words"],
    }
    return {
        "default": "auto" if all(checks.values()) else "template",
        "model": win,
        "checks": checks,
    }


def blocks(root: Path = ROOT) -> dict[str, str]:
    res = json.loads((root / "results" / "results.json").read_text("utf-8"))
    facts = json.loads((root / "data" / "facts.json").read_text("utf-8"))
    samples = json.loads((root / "data" / "samples.json").read_text("utf-8"))
    index = json.loads((root / "docs" / "demo" / "index.json").read_text("utf-8"))
    m = res["machine"]
    lo = res.get("load_observed", {})
    out: dict[str, str] = {}

    out["machine"] = (
        f"{m['cpu']}, {m['logical_cores']} logical cores ({m['threads_used']} threads used for the CPU rows). "
        f"PyTorch {m['torch']} (the CPU rows are forced onto the CPU), transformers {m['transformers']}, "
        f"OpenCV {m['opencv']}, Python {m['python']}. GPU: {m.get('gpu') or 'none'}. Measured on {m['date']}."
    )

    def fmt(x: dict | None) -> str:
        x = x or {}
        bits = []
        if "cpu_busy_percent" in x:
            bits.append(f"CPU {x['cpu_busy_percent']:.0f}% busy")
        if "gpu_busy_percent" in x:
            bits.append(
                f"GPU {x['gpu_busy_percent']}% busy, {x['gpu_mem_used_mb']} of "
                f"{x['gpu_mem_total_mb']} MB of its memory in use"
            )
        return ", ".join(bits) or "not recorded"

    out["load"] = table(
        ["Group", "Before", "After"],
        [[g, fmt(v.get("before")), fmt(v.get("after"))] for g, v in lo.items()],
    )

    # ------------------------------------------------------------------ models and downloads
    fm = facts["models"]
    det_rows = []
    for key in ("dfine-nano", "dfine-small", "rtdetr-r18"):
        f = fm[key]
        det_rows.append(
            [
                f"`{f['repo']}`",
                LICENCE_NAMES.get(f["licence"], f["licence"]),
                f"{f['params'] / 1e6:.1f}",
                size_text(f["download_bytes"]),
                f"{f['published_ap']} ([source]({f['ap_source']}))",
            ]
        )
    for key in ("qwen-0.5b", "qwen-1.5b"):
        f = fm[key]
        det_rows.append(
            [
                f"`{f['repo']}`",
                LICENCE_NAMES.get(f["licence"], f["licence"]),
                f"{f['params'] / 1e6:.0f}",
                size_text(f["download_bytes"]),
                "n/a",
            ]
        )
    cap = fm["smolvlm-256m"]
    det_rows.append(
        [
            f"`{cap['repo']}`",
            LICENCE_NAMES.get(cap["licence"], cap["licence"]),
            f"{cap['params'] / 1e6:.0f}",
            size_text(cap["download_bytes"]),
            "n/a",
        ]
    )
    out["models"] = (
        table(
            ["Hub id", "Licence", "Parameters (M)", "Download", "Published COCO AP (val)"],
            det_rows,
        )
        + f"\n\nLicence, parameters and download size are read from the Hugging Face hub on {facts['fetched_on']} "
        "(`scripts/fetch_facts.py`). The AP figures are quoted from the sources linked, not measured here."
    )
    from visionlens.describe.llm import DEFAULT_LLM
    from visionlens.models import fetch_plan

    repo_key = {m["repo"]: k for k, m in fm.items()}

    def plan_bytes(**kw) -> int:
        return sum(fm[repo_key[r]]["download_bytes"] for r, _ in fetch_plan(**kw))

    base = plan_bytes()
    out["downloads"] = (
        f"`visionlens fetch-models` downloads the default detector ({size_text(base)}). `--llm` adds the "
        f"optional rewrite model, {DEFAULT_LLM.split('/')[-1]} ({size_text(plan_bytes(llm=True) - base)}), "
        f"`--caption` adds the optional scene captioner, SmolVLM-256M-Instruct "
        f"({size_text(plan_bytes(caption=True) - base)}; only the weights and tokenizer files, not the "
        f"repository's ONNX exports), and `--all` fetches everything in the models table "
        f"({size_text(plan_bytes(all_models=True))} in total)."
    )

    # ------------------------------------------------------------------ detector speed
    if "detectors" in res:
        gpu = {(d["model"], d["input_size"]): d for d in res.get("detectors_gpu", [])}
        rows = []
        for d in res["detectors"]:
            g = gpu.get((d["model"], d["input_size"]))
            rows.append(
                [
                    d["model"],
                    d["input_size"],
                    f"{fm[d['model']]['params'] / 1e6:.1f}",
                    d["ms_median"],
                    d["ms_p95"],
                    d["fps_median"],
                    f"{g['ms_median']} ms / {g['fps_median']}" if g else "not run",
                    d["detections_per_frame"],
                ]
            )
        w, h = res["pipeline"]["resolution"].split("x")
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
            + f"One detector call on a {w}x{h} frame of the intersection clip (resized to the input size), "
            f"batch 1, {res['detectors'][0]['frames']} frames, per-call time includes pre- and "
            f"post-processing. CPU columns: {m['threads_used']} threads, float32, "
            f"{cpu_busy_text(lo, 'detectors')}. GPU column: {m.get('gpu') or 'n/a'}, float32, "
            f"{gpu_busy_text(lo, 'detectors')}."
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
            f"CPU row: {m['threads_used']} threads, {cpu_busy_text(lo, 'pipeline')}. "
            f"GPU row: float32 detector, {gpu_busy_text(lo, 'pipeline')}.\n\n"
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

    # ------------------------------------------------------------------ tracking
    if "tracker" in res:
        t = res["tracker"]
        keys = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "IDsw"]
        out["tracker_setup"] = (
            f"{len(t['scenes'])} scene types x {t['seeds']} seeds, {t['seconds_per_scene']:.0f} s at "
            f"{t['fps']:.0f} fps, {t['gt_boxes_total']} ground-truth boxes in total. "
            "Detections are the exact ground truth plus injected noise (see `NoiseConfig`); "
            "these are **synthetic-scene results, not MOT-benchmark results**. IDsw is the mean number of "
            "identity switches per scene."
        )
        ours = next(r for r in t["ablation"] if r["variant"].startswith("ByteTrack-style"))
        naive = next(r for r in t["ablation"] if r["variant"].startswith("naive"))
        single = next(r for r in t["ablation"] if r["variant"].startswith("no low-score"))
        out["tracker_headline"] = (
            f"At noise level {ours['level']} and detector stride {ours['stride']} the tracker scores IDF1 "
            f"{ours['IDF1']} and HOTA {ours['HOTA']}, against IDF1 {naive['IDF1']} and HOTA {naive['HOTA']} "
            f"for the naive per-frame IoU baseline ({ours['IDsw']} identity switches per scene against "
            f"{naive['IDsw']}). Without the low-score second pass it scores IDF1 {single['IDF1']}."
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

    # ------------------------------------------------------------------ rewrite and guard
    if "describer" in res:
        d = res["describer"]
        n_syn = sum(1 for c in d["cases"] if c.startswith("synthetic-"))
        n_rec = sum(1 for c in d["cases"] if c.startswith("recorded-"))
        rows = []
        for mod in d["models"]:
            cpu = mod.get("cpu")
            load = ((cpu or {}).get("load") or {}).get("cpu_busy_percent")
            rows.append(
                [
                    mod["model"],
                    f"{mod['accepted_first_try']} / {mod['cases']} ({mod['first_try_rate']:.0%})",
                    f"{mod['accepted']} / {mod['cases']} ({mod['acceptance_rate']:.0%})",
                    mod["mean_attempts"],
                    f"{mod['mean_seconds']} s ({mod['device']}, {mod['dtype']})",
                    f"{cpu['mean_seconds']} s, {cpu['mean_attempts']} attempts ({cpu['device']}, "
                    f"{cpu['dtype']}, {cpu['cases']} cases, CPU {load}% busy)"
                    if cpu
                    else "not run",
                ]
            )
        out["describer"] = (
            f"{len(d['cases'])} cases: {n_syn} synthetic scenes (every scene type x 3 seeds, simulated "
            f"detector) and the {n_rec} recorded clips. The facts go to the model as a compact structured "
            "list; a rewrite the guard rejects is regenerated, up to "
            f"{d['models'][0]['max_attempts']} attempts in all, with the guard's reason and the offending "
            "sentence fed back. Greedy decoding.\n\n"
            + table(
                [
                    "Model",
                    "Accepted on the first try",
                    "Accepted after retries",
                    "Mean attempts",
                    "Mean time per clip, GPU",
                    "Mean time per clip, CPU",
                ],
                rows,
            )
        )
        tpl = d["template"]["readability"]
        rrows = [
            [
                "Template",
                tpl["sentences"],
                tpl["words"],
                tpl["words_per_sentence"],
                tpl["repeated_openings"],
            ]
        ]
        for mod in d["models"]:
            for label, key in (
                ("text shown (accepted rewrite, else the template)", "readability_shown"),
                ("accepted rewrites only", "readability_accepted_rewrites"),
            ):
                v = mod[key]
                if v:
                    rrows.append(
                        [
                            f"`{mod['model'].split('/')[-1]}`, {label}",
                            v["sentences"],
                            v["words"],
                            v["words_per_sentence"],
                            v["repeated_openings"],
                        ]
                    )
        out["describer_readability"] = (
            "Means over the same cases. Repeated openings counts neighbouring sentences that start "
            "with the same word.\n\n"
            + table(
                ["Text", "Sentences", "Words", "Words per sentence", "Repeated openings"], rrows
            )
        )
        reasons: dict[str, int] = {}
        for mod in d["models"]:
            for r in mod["rows"]:
                for why in r["attempt_log"]:
                    key = _category(why or "")
                    reasons[key] = reasons.get(key, 0) + 1
        out["describer_reasons"] = (
            ", ".join(f"{n} x {k}" for k, n in sorted(reasons.items(), key=lambda kv: -kv[1]))
            or "none"
        )
        # what the retries did, stated from the rows
        parts = []
        for mod in d["models"]:
            name = mod["model"].split("/")[-1]
            same = sum(
                1
                for r in mod["rows"]
                if not r["accepted"] and len(set(r["attempt_log"])) == 1 and r["attempts"] > 1
            )
            parts.append(
                f"for `{name}` retries moved acceptance from {mod['accepted_first_try']} of {mod['cases']} "
                f"cases on the first try to {mod['accepted']} of {mod['cases']} "
                f"(mean {mod['mean_attempts']} attempts); in {same} of its {mod['rejected']} final "
                "rejections every attempt failed for the same reason"
            )
        out["retry_effect"] = "Effect of the retries: " + "; ".join(parts) + "."
        dec = decide_default(res)
        win = dec["model"]
        got = win["readability_shown"]
        verdict = (
            "The rewrite passes all three tests, so it is the default describer where a GPU is available "
            "(`--describer auto`) and the template is the default on a CPU-only machine."
            if dec["default"] == "auto"
            else "The rewrite does not pass all three tests, so the template stays the default describer "
            "and the rewrite is an option."
        )
        out["describer_choice"] = (
            f"**Choice by evidence.** The best model is `{win['model']}`: accepted on "
            f"{win['accepted']} of {win['cases']} cases after retries (threshold {ACCEPT_MIN:.0%}: "
            f"{'met' if dec['checks']['accepted'] else 'not met'}), repeated openings "
            f"{got['repeated_openings']} against the template's {tpl['repeated_openings']} "
            f"({'met' if dec['checks']['openings'] else 'not met'}), words {got['words']} against "
            f"{tpl['words']} ({'met' if dec['checks']['words'] else 'not met'}). {verdict}"
        )

    # ------------------------------------------------------------------ scene captioner
    if "captioner" in res:
        from visionlens.bench import SETTING_WORDS

        c = res["captioner"]
        rows = []
        for p in c["prompts"]:
            real = [
                r
                for r in p["rows"]
                if r["case"] in ("recorded-intersection", "recorded-traffic", "recorded-still")
            ]
            rows.append(
                [
                    p["key"] + (" (chosen)" if p["key"] == c["chosen_prompt"] else ""),
                    f"{p['setting_ok']} / {p['cases']}",
                    f"{sum(r['setting_ok'] for r in real)} / {len(real)}",
                    f"{p['names_unseen']} / {p['cases']}",
                    f"{p['one_sentence']} / {p['cases']}",
                    f"{p['mean_seconds']} s ({p['device']}, {p['dtype']})",
                ]
            )
        chosen = next(p for p in c["prompts"] if p["key"] == c["chosen_prompt"])
        cpu = c.get("cpu")
        out["captioner"] = (
            f"{len(c['cases'])} cases (the 3 recorded clips, the recorded image and all 12 synthetic scenes), "
            "three key frames per clip (10%, 50% and 90% of its length); the shown caption is the middle "
            "frame's, cut at its first full stop. Rubric, applied automatically: (1) the caption names the "
            "setting correctly (it contains a word from the list for that case, below), (2) it names an object "
            "class (person, car, bus, ...) "
            "that the detector never saw in that clip, (3) it is one complete sentence of at least four words."
            "\n\n"
            + table(
                [
                    "Prompt",
                    "Setting named correctly (all)",
                    "Setting named correctly (3 real captures)",
                    "Names an unseen class",
                    "One complete sentence",
                    "Mean time per clip",
                ],
                rows,
            )
            + "\n\nWord lists for rubric (1) (`SETTING_WORDS` in `src/visionlens/bench.py`):\n"
            + "\n".join(f"- {k}: {', '.join(v)}" for k, v in SETTING_WORDS.items())
            + "\n\nPrompts:\n"
            + "\n".join(f"- **{p['key']}**: \u201c{p['prompt']}\u201d" for p in c["prompts"])
            + (
                f"\n\nTime per clip (three key frames) for prompt {chosen['key']}: {chosen['mean_seconds']} s on "
                f"{chosen['device']} ({chosen['dtype']}) over {chosen['cases']} cases"
                + (
                    f", {cpu['mean_seconds']} s on the {cpu['device']} ({cpu['dtype']}) over {cpu['cases']} cases "
                    f"with the CPU {cpu['load'].get('cpu_busy_percent', 'n/a')}% busy"
                    if cpu
                    else ""
                )
                + "."
            )
        )
        ex = []
        for r in chosen["rows"]:
            if r["case"].startswith("recorded-"):
                ex.append(
                    [
                        r["case"].replace("recorded-", ""),
                        r["shown"],
                        "yes" if r["setting_ok"] else "no",
                    ]
                )
        real_names = ("recorded-intersection", "recorded-traffic", "recorded-still")
        real = [r for r in chosen["rows"] if r["case"] in real_names]
        rendered = [r for r in chosen["rows"] if r["case"].startswith("synthetic-")]
        plaza = [r for r in rendered if not r["case"].startswith("synthetic-street")]
        quotes = sorted({r["shown"] for r in plaza})
        out["caption_trust"] = (
            f"**How far to trust it.** On the {len(real)} real captures (two clips and the image) it named the "
            f"kind of place correctly in {sum(r['setting_ok'] for r in real)} of {len(real)}; over all "
            f"{chosen['cases']} cases it named a class the detector never saw in {chosen['names_unseen']}. "
            f"On the {len(rendered)} rendered scenes it passed the setting test in "
            f"{sum(r['setting_ok'] for r in rendered)}. For the {len(plaza)} plaza-like scenes (dark open "
            f"ground with building silhouettes) it wrote {len(quotes)} different captions: "
            + "; ".join(f"\u201c{q}\u201d" for q in quotes)
            + "."
        )
        out["caption_examples"] = table(
            ["Recorded clip", "Caption shown", "Setting named correctly"], ex
        )

    # ------------------------------------------------------------------ recorded demo
    rows = []
    for clip in index["clips"]:
        r = json.loads((root / "docs" / "demo" / clip["json"]).read_text("utf-8"))
        meta, tm, ds = r["meta"], r["timings"], r.get("description") or {}
        model = ds.get("model")
        if model is None:
            rewrite = "none"
        elif model["accepted"]:
            rewrite = "accepted"
        else:
            rewrite = f"rejected ({model['rejected_reason']}), template shown"
        sc = r.get("scene")
        if sc is None:
            caption = "none"
        else:
            caption = f"\u201c{sc['text']}\u201d" + (
                f" (flagged: names {', '.join(sc['unseen'])})" if sc["flagged"] else ""
            )
        where = {"cuda": "GPU", "cpu": "CPU"}.get(tm.get("device", ""), "simulated detector")
        files = [root / "docs" / "demo" / clip["json"], root / "docs" / "demo" / clip["video"]]
        size = sum(f.stat().st_size for f in files)
        rows.append(
            [
                clip["label"],
                "image"
                if meta["kind"] == "image"
                else f"{meta['n_frames']} frames, {meta['duration']:.0f} s",
                r["summary"].get("tracks", 0) if meta["kind"] == "video" else "n/a",
                len(r["events"]) if meta["kind"] == "video" else "n/a",
                f"{tm['detect_ms_mean']} ms ({where})",
                rewrite,
                caption,
                f"{size / 1e6:.2f} MB",
            ]
        )
    out["demo"] = (
        f"Recorded on {index['recorded_on']} with {index['detector']} at stride {index['stride']}; "
        f"detector device: {index['device']}"
        + (f", language model in {index['llm_dtype']}" if index.get("llm_dtype") else "")
        + (f", captioner in {index['caption_dtype']}" if index.get("caption_dtype") else "")
        + ".\n\n"
        + table(
            [
                "Clip",
                "Length",
                "Tracks",
                "Events",
                "Detect, mean per call",
                "Model rewrite",
                "Scene caption (vision-language model)",
                "Video + JSON",
            ],
            rows,
        )
    )

    # ------------------------------------------------------------------ limits and credits
    from visionlens.api import Settings

    s = Settings()
    out["limits"] = (
        f"Uploads are capped at {s.max_upload_mb} MB; only the first {s.max_seconds:.0f} s are analysed, "
        f"at up to {s.max_width} px wide. They are written to a temporary folder and deleted when processing ends."
    )
    lines = []
    for sm in samples.values():
        src = sm["source"]
        lines.append(
            f"  - {sm['label']}: [{src['title']}]({src['page']}) by {src['author']}, {src['license']}; "
            f"an excerpt of {src['seconds']:.0f} s starting at {src['trimmed_from'].rstrip('s')} s, resized to "
            f"{sm['width']} px wide and re-encoded as VP9 WebM."
        )
    out["credits"] = "\n".join(lines)
    return out


def render(text: str, generated: dict[str, str]) -> tuple[str, list[str]]:
    problems: list[str] = []
    for name, body in generated.items():
        pat = re.compile(
            rf"(<!-- RESULTS:{name}:start -->)(.*?)(<!-- RESULTS:{name}:end -->)", re.S
        )
        if not pat.search(text):
            problems.append(f"README has no marker for '{name}'")
            continue
        text = pat.sub(lambda mm, b=body: f"{mm.group(1)}\n{b}\n{mm.group(3)}", text)
    used = set(re.findall(r"<!-- RESULTS:(\w+):start -->", text))
    for name in sorted(used - set(generated)):
        problems.append(f"README marker '{name}' has no generator")
    return text, problems


def check(root: Path = ROOT) -> list[str]:
    """Problems that make the README disagree with its sources (empty when it is in sync)."""
    path = root / "README.md"
    current = path.read_text("utf-8")
    new, problems = render(current, blocks(root))
    if new != current:
        diff = difflib.unified_diff(
            current.splitlines(), new.splitlines(), "README.md", "regenerated", lineterm="", n=0
        )
        problems.append("README.md is out of date:\n" + "\n".join(list(diff)[:40]))
    return problems


def main() -> int:
    if "--check" in sys.argv:
        problems = check()
        for p in problems:
            print("PROBLEM:", p)
        print("README is in sync" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    path = ROOT / "README.md"
    new, problems = render(path.read_text("utf-8"), blocks())
    for p in problems:
        print("PROBLEM:", p)
    path.write_text(new, "utf-8")
    print("README updated")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
