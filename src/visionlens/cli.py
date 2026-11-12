"""Command line: run, annotate, bench, serve, fetch-models, record-demo, make-synthetic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "docs" / "demo"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--detector", default="dfine-nano", help="dfine-nano | dfine-small | rtdetr-r18")
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--stride", type=int, default=3, help="run the detector every N frames")
    p.add_argument(
        "--describer", default="template", help="template (default) | llm | openai | anthropic"
    )
    p.add_argument(
        "--threads", type=int, default=None, help="torch/OpenCV threads (default: 1/4 of cores)"
    )
    p.add_argument("--device", default=None, help="auto (default) | cpu | cuda")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--fps", type=float, default=15.0, help="analysis frame rate cap")


def default_output(input_path: str | Path, suffix: str) -> Path:
    """Where results go unless --out says otherwise: ./outputs/<name><suffix> (git-ignored)."""
    return Path("outputs") / f"{Path(input_path).stem}{suffix}"


def _apply_device(args: argparse.Namespace) -> None:
    import os

    if getattr(args, "device", None):
        os.environ["VISIONLENS_DEVICE"] = args.device


def _make_pipeline(args: argparse.Namespace):
    from .describe import make_describer
    from .detect import load_detector
    from .pipeline import PipelineConfig
    from .sysutil import limit_threads

    limit_threads(args.threads)
    _apply_device(args)
    det = load_detector(args.detector, args.input_size)
    return det, PipelineConfig(stride=args.stride), make_describer(args.describer)


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import analyze_file

    det, cfg, desc = _make_pipeline(args)

    def progress(fr, total):
        if fr.i % 15 == 0:
            print(f"\rframe {fr.i}" + (f"/{total}" if total else ""), end="", file=sys.stderr)

    res = analyze_file(
        args.input,
        det,
        cfg,
        desc,
        playback_path=args.video_out,
        max_seconds=args.max_seconds,
        target_fps=args.fps,
        on_frame=progress,
        threads=args.threads,
    )
    print(file=sys.stderr)
    out = Path(args.out) if args.out else default_output(args.input, ".visionlens.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(res.model_dump_json(), "utf-8")
    d = res.description
    print(d.template.text if d else "")
    if d and d.model:
        verdict = "accepted" if d.model.accepted else f"rejected ({d.model.rejected_reason})"
        print(f"\n[{d.model.source} rewrite {verdict}]\n{d.model.text}")
    t = res.timings
    print(
        f"\n{res.meta.n_frames} frames, {res.summary['tracks']} tracks, {len(res.events)} events; "
        f"detector {t.detect_ms_mean} ms/call, {t.fps_end_to_end} fps end to end "
        f"({t.threads} threads). Wrote {out}",
        file=sys.stderr,
    )
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    from .annotate import annotate_video
    from .schemas import Result

    res = Result.model_validate_json(Path(args.result).read_text("utf-8"))
    out = Path(args.out) if args.out else default_output(args.input, ".annotated.webm")
    out.parent.mkdir(parents=True, exist_ok=True)
    annotate_video(args.input, res, out, show_dets=args.show_dets)
    print(f"wrote {out}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_all

    run_all(
        args.out,
        args.threads,
        args.note,
        quick=args.quick,
        skip=tuple(args.skip),
        gpu_device=args.gpu_device,
    )
    print(f"wrote {args.out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .sysutil import limit_threads

    limit_threads(args.threads)
    _apply_device(args)
    uvicorn.run(
        "visionlens.api:create_app", factory=True, host=args.host, port=args.port, log_level="info"
    )
    return 0


def cmd_fetch_models(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    from .describe.llm import DEFAULT_LLM
    from .detect.hf import MODELS

    wanted = [MODELS["dfine-nano"]]
    if args.all:
        wanted += [MODELS["dfine-small"], MODELS["rtdetr-r18"]]
    if args.llm or args.all:
        wanted.append(DEFAULT_LLM)
    for repo in wanted:
        print("downloading", repo)
        snapshot_download(repo)
    print("done")
    return 0


def cmd_make_synthetic(args: argparse.Namespace) -> int:
    from .synth import make_scene, render_video

    sc = make_scene(args.scene, args.seed, seconds=args.seconds)
    out = (
        Path(args.out) if args.out else Path("outputs") / f"synthetic-{args.scene}-{args.seed}.webm"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    render_video(sc, str(out))
    print(f"wrote {out}")
    return 0


def cmd_record_demo(args: argparse.Namespace) -> int:
    from .demo import record_demo

    _apply_device(args)
    record_demo(
        DEMO_DIR,
        args.detector,
        args.stride,
        args.threads,
        args.describer,
        names=args.names or None,
        machine_note=args.note,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="visionlens", description="Detect, track and describe.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="analyse a video or image")
    p.add_argument("input")
    p.add_argument("--out", help="result JSON path (default: outputs/<name>.visionlens.json)")
    p.add_argument("--video-out", help="also write the analysed frames as VP9 WebM")
    _common(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("annotate", help="draw a result onto a video")
    p.add_argument("input")
    p.add_argument("result")
    p.add_argument("--out", help="output WebM (default: outputs/<name>.annotated.webm)")
    p.add_argument("--show-dets", action="store_true")
    p.set_defaults(fn=cmd_annotate)

    p = sub.add_parser("bench", help="measure everything the README quotes")
    p.add_argument("--out", default=str(ROOT / "results" / "results.json"))
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--note", default="", help="free text, e.g. whether the machine was busy")
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--gpu-device",
        default="auto",
        help="device for the GPU rows, the pipeline GPU row and the describer (auto | cuda | cpu)",
    )
    p.add_argument(
        "--skip", nargs="*", default=[], choices=["detectors", "tracker", "pipeline", "describer"]
    )
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("serve", help="run the web app and API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8106)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default=None, help="auto (default) | cpu | cuda")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("fetch-models", help="download the models (one time)")
    p.add_argument("--llm", action="store_true", help="also the local describer model")
    p.add_argument("--all", action="store_true", help="every detector plus the LLM")
    p.set_defaults(fn=cmd_fetch_models)

    p = sub.add_parser("make-synthetic", help="render a synthetic scene to WebM")
    p.add_argument("scene", choices=["crossing", "occlusion", "crowd", "street"])
    p.add_argument("--out", help="output WebM (default: outputs/synthetic-<scene>-<seed>.webm)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.set_defaults(fn=cmd_make_synthetic)

    p = sub.add_parser("record-demo", help="record the static demo's results from real runs")
    p.add_argument("names", nargs="*")
    p.add_argument("--note", default="")
    _common(p)
    p.set_defaults(fn=cmd_record_demo)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
