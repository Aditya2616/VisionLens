"""Detect -> track -> events -> describe, as one generator-friendly pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .describe import Describer, Facts, TemplateDescriber, build_description, describe_image
from .detect.base import Detector
from .events import EventConfig, extract_events, summarize
from .schemas import Description, FrameResult, Meta, Result, Timings, TrackOut
from .sysutil import cpu_name
from .track import ByteTracker, TrackerConfig

if TYPE_CHECKING:
    from .caption import SceneCaptioner

STILL_CONFIDENCE = TrackerConfig().high_thr  # a still image reports confident detections only


@dataclass
class PipelineConfig:
    stride: int = 3  # run the detector every Nth frame; the tracker predicts in between
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    events: EventConfig = field(default_factory=EventConfig)
    keep_keyframes: bool = False  # keep a frame every half second for the optional captioner


class Pipeline:
    def __init__(self, detector: Detector, config: PipelineConfig | None = None) -> None:
        self.detector = detector
        self.cfg = config or PipelineConfig()
        self.reset()

    def reset(self) -> None:
        self.tracker = ByteTracker(self.cfg.tracker)
        self.frames: list[FrameResult] = []
        self._detect_ms: list[float] = []
        self._track_ms: list[float] = []
        self._t_start = time.perf_counter()
        self._thumbs: list[tuple[int, float, np.ndarray]] = []
        self._next_keep = 0.0

    def process(self, frames: Iterable[tuple[int, float, np.ndarray]]) -> Iterator[FrameResult]:
        """Run the first two stages frame by frame, yielding each FrameResult as it is ready."""
        self.reset()
        for i, t, frame in frames:
            if self.cfg.keep_keyframes and frame is not None and t >= self._next_keep:
                self._thumbs.append((i, t, frame.copy()))
                self._next_keep = t + 0.5
            dets = None
            if i % self.cfg.stride == 0:
                t0 = time.perf_counter()
                dets = self.detector.detect(frame, i)
                self._detect_ms.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            tracks = self.tracker.step(dets, i)
            self._track_ms.append((time.perf_counter() - t0) * 1000)
            fr = FrameResult(
                i=i,
                t=round(t, 3),
                dets=[
                    d.model_copy(
                        update={
                            "box": tuple(round(v, 1) for v in d.box),
                            "score": round(d.score, 3),
                        }
                    )
                    for d in dets
                ]
                if dets is not None
                else None,
                tracks=tracks,
            )
            self.frames.append(fr)
            yield fr

    def finish(
        self,
        name: str,
        width: int,
        height: int,
        fps: float,
        describer: Describer | None = None,
        on_token: Callable[[str], None] | None = None,
        on_model_token: Callable[[str], None] | None = None,
        threads: int = 0,
        truncated: bool = False,
        machine_note: str = "",
        captioner: SceneCaptioner | None = None,
    ) -> Result:
        """Stage 3 and packaging, after `process` has been exhausted."""
        wall = time.perf_counter() - self._t_start
        n = len(self.frames)
        events = extract_events(self.frames, width, height, fps, self.cfg.events)
        summary = summarize(self.frames, fps, self.cfg.events)
        facts = Facts(
            duration=n / fps if fps else 0.0,
            width=width,
            height=height,
            events=events,
            summary=summary,
        )
        template = TemplateDescriber().describe(facts, on_token)
        description: Description = build_description(facts, template, describer, on_model_token)
        scene, scene_note = None, ""
        if captioner is not None and self._thumbs:
            from .caption import pick_keyframes

            seen = {t.cls for f in self.frames for t in f.tracks} | {
                d.cls for f in self.frames for d in (f.dets or [])
            }
            try:
                scene = captioner.caption(pick_keyframes(self._thumbs), seen)
            except Exception as exc:  # noqa: BLE001 - an optional stage must not break the run
                scene_note = f"scene caption failed: {exc}"
        dm = np.array(self._detect_ms) if self._detect_ms else np.array([0.0])
        timings = Timings(
            detect_ms_mean=round(float(dm.mean()), 1),
            detect_ms_p95=round(float(np.percentile(dm, 95)), 1),
            track_ms_mean=round(float(np.mean(self._track_ms or [0.0])), 3),
            detect_calls=len(self._detect_ms),
            caption_s=scene.seconds if scene else 0.0,
            wall_s=round(wall, 2),
            fps_end_to_end=round(n / wall, 2) if wall > 0 else 0.0,
            threads=threads,
            cpu=cpu_name(),
            device=getattr(self.detector, "device", ""),
            machine_note=machine_note,
        )
        meta = Meta(
            name=name,
            width=width,
            height=height,
            fps=round(fps, 3),
            n_frames=n,
            duration=round(n / fps, 3) if fps else 0.0,
            stride=self.cfg.stride,
            detector=getattr(self.detector, "name", "detector"),
            input_size=getattr(self.detector, "input_size", 0),
            tracker=self.cfg.tracker.as_dict(),
            truncated=truncated,
            note=scene_note,
        )
        return Result(
            meta=meta,
            frames=self.frames,
            events=events,
            description=description,
            scene=scene,
            timings=timings,
            summary=summary,
        )


def analyze_image(
    detector: Detector,
    image: np.ndarray,
    name: str = "image",
    captioner: SceneCaptioner | None = None,
) -> Result:
    """Still image: detection plus a counts-and-positions description. No tracking."""
    detector.detect(image, 0)  # warm-up, so the reported time is a steady-state call
    t0 = time.perf_counter()
    dets = detector.detect(image, 0)
    ms = (time.perf_counter() - t0) * 1000
    h, w = image.shape[:2]
    # Only confident detections are reported (the tracker's "high score" threshold); the raw
    # low-score boxes stay in `dets` for the raw-detections layer.
    sure = [d for d in dets if d.score >= STILL_CONFIDENCE]
    variant = describe_image(sure, w)
    from collections import Counter

    by_class = dict(Counter(d.cls for d in sure))
    meta = Meta(
        name=name,
        kind="image",
        width=w,
        height=h,
        fps=1.0,
        n_frames=1,
        duration=0.0,
        stride=1,
        detector=getattr(detector, "name", "detector"),
        input_size=getattr(detector, "input_size", 0),
    )
    frame = FrameResult(
        i=0,
        t=0.0,
        dets=[d.model_copy(update={"box": tuple(round(v, 1) for v in d.box)}) for d in dets],
        tracks=[
            TrackOut(
                id=k + 1,
                box=tuple(round(v, 1) for v in d.box),
                cls=d.cls,
                score=d.score,
                detected=True,
            )
            for k, d in enumerate(sure)
        ],
    )
    scene = None
    if captioner is not None:
        try:
            scene = captioner.caption([(0.0, image)], {d.cls for d in dets})
        except Exception as exc:  # noqa: BLE001
            meta.note = f"scene caption failed: {exc}"
    return Result(
        meta=meta,
        frames=[frame],
        events=[],
        description=Description(template=variant, used="template"),
        scene=scene,
        timings=Timings(
            detect_ms_mean=round(ms, 1),
            detect_ms_p95=round(ms, 1),
            detect_calls=1,
            wall_s=round(ms / 1000, 2),
            cpu=cpu_name(),
            device=getattr(detector, "device", ""),
        ),
        summary={"tracks": len(sure), "by_class": by_class, "peak": len(sure)},
    )
