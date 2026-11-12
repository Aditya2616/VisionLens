from __future__ import annotations

import cv2
import numpy as np

from visionlens.sprite import write_sprite
from visionlens.synth import NoiseConfig, NoisyOracle, make_scene
from visionlens.video import WebmWriter, iter_frames, probe


def test_scene_is_deterministic_and_has_ground_truth():
    a, b = make_scene("crowd", 5, seconds=3), make_scene("crowd", 5, seconds=3)
    assert a.gt(10) == b.gt(10)
    assert np.array_equal(a.render(10), b.render(10))
    assert make_scene("crowd", 6, seconds=3).gt(20) != a.gt(20)
    seq = make_scene("street", 1).gt_sequence()
    assert len(seq) == 150 and any(seq[i] for i in seq)


def test_occluders_lower_visibility():
    sc = make_scene("occlusion", 0, seconds=10)
    vis = [v for i in range(sc.n_frames) for _g, _b, v, _c in sc.gt(i)]
    assert min(vis) < 0.5 and max(vis) == 1.0


def test_noisy_oracle_levels_and_reproducibility():
    sc = make_scene("street", 1, seconds=3)
    perfect = NoisyOracle(sc, NoiseConfig.level(0))
    gt = [g for g in sc.gt(20) if g[2] >= 0.4]
    assert len(perfect.detect(None, 20)) == len(gt)
    harsh1 = NoisyOracle(sc, NoiseConfig.level(3)).detect(None, 20)
    harsh2 = NoisyOracle(sc, NoiseConfig.level(3)).detect(None, 20)
    assert [d.box for d in harsh1] == [d.box for d in harsh2]


def test_vp9_webm_roundtrip_and_resampling(tmp_path):
    path = tmp_path / "t.webm"
    with WebmWriter(path, 30.0, (160, 96)) as w:
        for i in range(60):
            w.write(np.full((96, 160, 3), i * 4, np.uint8))
    info = probe(path)
    assert (info.width, info.height) == (160, 96) and abs(info.fps - 30) < 1
    frames = list(iter_frames(path, max_width=80, target_fps=10))
    assert 18 <= len(frames) <= 22  # 2 s at 10 fps
    assert frames[0][2].shape[:2] == (48, 80)
    assert abs(frames[5][1] - 0.5) < 1e-6
    assert len(list(iter_frames(path, max_seconds=0.5, target_fps=10))) == 5


def test_sprite_sheet_geometry(tmp_path):
    frames = [(i / 10, np.full((90, 160, 3), i, np.uint8)) for i in range(30)]
    info = write_sprite(frames, tmp_path / "s.jpg", fps=5.0, thumb_w=80, cols=4)
    assert info["count"] == 15 and info["thumb_h"] == 45
    img = cv2.imread(str(tmp_path / "s.jpg"))
    assert img.shape[:2] == (4 * 45, 4 * 80)
