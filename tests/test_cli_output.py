from __future__ import annotations

import json

from visionlens import cli
from visionlens.pipeline import PipelineConfig


def test_run_writes_under_outputs_not_next_to_the_input(
    scene_video, oracle, tmp_path, monkeypatch, capsys
):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(
        cli, "_make_pipeline", lambda args: (oracle, PipelineConfig(stride=2), None)
    )
    assert cli.main(["run", str(scene_video), "--fps", "10"]) == 0
    out = work / "outputs" / "scene.visionlens.json"
    assert out.exists() and "meta" in json.loads(out.read_text("utf-8"))
    assert not list(scene_video.parent.glob("*.visionlens.json"))  # the input folder stays clean


def test_out_overrides_the_default(scene_video, oracle, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli, "_make_pipeline", lambda args: (oracle, PipelineConfig(stride=2), None)
    )
    target = tmp_path / "elsewhere" / "r.json"
    assert cli.main(["run", str(scene_video), "--fps", "10", "--out", str(target)]) == 0
    assert target.exists() and not (tmp_path / "outputs").exists()


def test_default_output_name():
    assert str(cli.default_output("a/b/clip.mp4", ".x.json")).replace("\\", "/") == (
        "outputs/clip.x.json"
    )
