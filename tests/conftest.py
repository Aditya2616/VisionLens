from __future__ import annotations

import pytest

from visionlens.schemas import Detection
from visionlens.synth import NoiseConfig, NoisyOracle, make_scene, render_video


def det(box, score=0.9, cls="person"):
    return Detection(box=tuple(float(v) for v in box), score=score, cls=cls)


@pytest.fixture(scope="session")
def small_scene():
    return make_scene("crossing", seed=3, seconds=6.0, fps=10.0, size=(480, 270))


@pytest.fixture(scope="session")
def scene_video(small_scene, tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "scene.webm"
    render_video(small_scene, str(path))
    return path


@pytest.fixture()
def oracle(small_scene):
    return NoisyOracle(small_scene, NoiseConfig.level(0))
