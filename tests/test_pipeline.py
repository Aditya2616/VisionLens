from __future__ import annotations

import pytest

from visionlens.bench import NaiveIoUTracker, run_tracker
from visionlens.describe import make_describer
from visionlens.metrics import evaluate
from visionlens.pipeline import Pipeline, PipelineConfig
from visionlens.runner import analyze_file
from visionlens.schemas import Result
from visionlens.synth import NoiseConfig, make_scene
from visionlens.track import ByteTracker


def test_pipeline_on_oracle_gives_tracks_events_and_description(small_scene, oracle):
    pipe = Pipeline(oracle, PipelineConfig(stride=1))
    for _ in pipe.process((i, i / small_scene.fps, None) for i in range(small_scene.n_frames)):
        pass
    res = pipe.finish("t", small_scene.width, small_scene.height, small_scene.fps)
    assert res.meta.n_frames == small_scene.n_frames and res.timings.detect_calls == 60
    assert res.summary["tracks"] >= 4
    assert {e.kind for e in res.events} >= {"enter", "leave"}
    assert res.description.template.text and res.description.used == "template"
    Result.model_validate_json(res.model_dump_json())  # the JSON contract round-trips


def test_stride_skips_the_detector_between_calls(small_scene, oracle):
    pipe = Pipeline(oracle, PipelineConfig(stride=3))
    frames = list(pipe.process((i, i / 10, None) for i in range(12)))
    assert [f.dets is not None for f in frames] == [True, False, False] * 4


def test_analyze_file_reads_video_writes_playback_and_streams_tokens(scene_video, oracle, tmp_path):
    tokens, seen = [], []
    res = analyze_file(
        scene_video,
        oracle,
        PipelineConfig(stride=2),
        None,
        playback_path=tmp_path / "play.webm",
        target_fps=10,
        max_width=320,
        on_frame=lambda fr, total: seen.append(fr.i),
        on_token=tokens.append,
        name="clip",
    )
    assert (tmp_path / "play.webm").stat().st_size > 1000
    assert len(seen) == res.meta.n_frames > 20
    assert "".join(tokens).strip() == res.description.template.text


def test_tracker_beats_naive_baseline_on_a_noisy_scene():
    scene = make_scene("street", 2)
    ours = run_tracker(scene, 2, 3, ByteTracker())
    naive = run_tracker(scene, 2, 3, NaiveIoUTracker())
    assert ours["IDF1"] > naive["IDF1"] and ours["HOTA"] > naive["HOTA"]


def test_perfect_detector_gives_high_scores_on_an_easy_scene():
    scene = make_scene("street", 1)
    m = run_tracker(scene, 0, 1, ByteTracker())
    assert m["MOTA"] > 0.9 and m["IDF1"] > 0.9 and m["IDsw"] <= 1
    assert evaluate(scene.gt_sequence(), scene.gt_sequence())["HOTA"] > 0.99


def test_make_describer_names():
    assert make_describer("template") is None
    assert make_describer("openai").name == "openai"
    with pytest.raises(ValueError):
        make_describer("nope")


def test_noise_config_levels_are_ordered():
    p = [NoiseConfig.level(k) for k in range(4)]
    assert p[0].p_miss == 0 and p[3].p_miss > p[1].p_miss


def test_still_image_reports_only_confident_detections():
    import numpy as np

    from visionlens.pipeline import analyze_image
    from visionlens.schemas import Detection

    class Fake:
        name = "fake"

        def detect(self, frame, index=0):
            return [
                Detection(box=(10, 10, 50, 90), score=0.9, cls="car"),
                Detection(box=(60, 10, 90, 90), score=0.2, cls="person"),  # below the floor
            ]

    res = analyze_image(Fake(), np.zeros((100, 120, 3), np.uint8))
    assert res.summary == {"tracks": 1, "by_class": {"car": 1}, "peak": 1}
    assert [t.cls for t in res.frames[0].tracks] == ["car"]
    assert len(res.frames[0].dets) == 2  # the raw boxes are still there for the raw layer
    assert "person" not in res.description.template.text
