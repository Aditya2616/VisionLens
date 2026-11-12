"""High-level helpers: analyse a video or image file, optionally writing the playback video."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .describe import Describer
from .detect.base import Detector
from .pipeline import Pipeline, PipelineConfig, analyze_image
from .schemas import FrameResult, Result
from .sysutil import default_threads

if TYPE_CHECKING:
    from .caption import SceneCaptioner

from .video import (
    WebmWriter,
    effective_fps,
    is_image,
    iter_frames,
    output_size,
    probe,
    read_image,
)


def analyze_file(
    path: str | Path,
    detector: Detector,
    config: PipelineConfig | None = None,
    describer: Describer | None = None,
    *,
    playback_path: str | Path | None = None,
    max_width: int = 640,
    target_fps: float = 15.0,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
    on_frame: Callable[[FrameResult, int | None], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    on_model_token: Callable[[str], None] | None = None,
    threads: int | None = None,
    machine_note: str = "",
    name: str | None = None,
    captioner: SceneCaptioner | None = None,
) -> Result:
    """Run the whole pipeline on a file. If playback_path is given, the resized frames are
    also written there as VP9 WebM so the browser can play exactly what was analysed."""
    path = Path(path)
    name = name or path.stem
    if is_image(path):
        return analyze_image(detector, read_image(path), name, captioner)
    info = probe(path)
    fps = effective_fps(info, target_fps)
    width, height = output_size(info, max_width)
    config = config or PipelineConfig()
    if captioner is not None:
        config.keep_keyframes = True
    pipe = Pipeline(detector, config)
    expected = int(min(info.duration or 1e9, max_seconds or 1e9) * fps) or None
    writer = WebmWriter(playback_path, fps, (width, height)) if playback_path else None
    n = 0

    def frames() -> Iterator[tuple[int, float, object]]:
        nonlocal n
        for i, t, frame in iter_frames(
            path,
            max_width=max_width,
            target_fps=target_fps,
            max_seconds=max_seconds,
            start_seconds=start_seconds,
        ):
            if writer is not None:
                writer.write(frame)
            n += 1
            yield i, t, frame

    try:
        for fr in pipe.process(frames()):  # type: ignore[arg-type]
            if on_frame:
                on_frame(fr, expected)
    finally:
        if writer is not None:
            writer.close()
    truncated = bool(max_seconds and n >= int(max_seconds * fps))
    result = pipe.finish(
        name,
        width,
        height,
        fps,
        describer,
        on_token,
        on_model_token,
        threads=threads or default_threads(),
        truncated=truncated,
        machine_note=machine_note,
        captioner=captioner,
    )
    return result
