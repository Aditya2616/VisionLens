"""Real-model tests. Deselected by default; run with `pytest -m slow` after `visionlens fetch-models --llm`."""

import cv2
import pytest

from visionlens.describe import Facts, TemplateDescriber, build_description
from visionlens.events import extract_events, summarize
from visionlens.synth import NoiseConfig, NoisyOracle, make_scene

pytestmark = pytest.mark.slow


def test_dfine_finds_cars_in_a_real_frame():
    from visionlens.detect.hf import HFDetector
    from visionlens.video import iter_frames

    clip = "docs/demo/traffic.webm"
    frame = next(
        f for _, _, f in iter_frames(clip, max_width=480, target_fps=10, start_seconds=3.0)
    )
    dets = HFDetector("dfine-nano").detect(frame)
    assert any(d.cls in {"car", "bus", "truck"} and d.score > 0.5 for d in dets)
    assert cv2 is not None


def test_local_llm_rewrite_goes_through_the_guard():
    from visionlens.describe.llm import LocalLLMDescriber
    from visionlens.pipeline import Pipeline, PipelineConfig

    sc = make_scene("occlusion", 0)
    p = Pipeline(NoisyOracle(sc, NoiseConfig.level(1, 0)), PipelineConfig(stride=2))
    for _ in p.process((i, i / sc.fps, None) for i in range(sc.n_frames)):
        pass
    ev = extract_events(p.frames, sc.width, sc.height, sc.fps)
    facts = Facts(sc.duration, sc.width, sc.height, ev, summarize(p.frames, sc.fps))
    d = build_description(facts, TemplateDescriber().describe(facts), LocalLLMDescriber())
    assert d.model is not None and d.used in {"model", "template"}
    assert d.used == "template" or d.model.accepted
