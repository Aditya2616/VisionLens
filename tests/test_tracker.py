from __future__ import annotations

import numpy as np

from visionlens.track import BoxKalman, ByteTracker, TrackerConfig, associate, iou_matrix

from .conftest import det


def box_at(x, y=100.0, w=40.0, h=80.0):
    return (x, y, x + w, y + h)


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_associate_picks_global_optimum_and_respects_threshold():
    sim = np.array([[0.9, 0.8], [0.85, 0.1]])
    matches, ua, ub = associate(sim, 0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]  # greedy would take (0,0) and leave row 1 with 0.1
    matches, ua, ub = associate(np.array([[0.2]]), 0.3)
    assert matches == [] and ua == [0] and ub == [0]


def test_kalman_learns_constant_velocity():
    kf = BoxKalman(box_at(0))
    for k in range(1, 30):
        kf.predict()
        kf.update(box_at(5.0 * k))
    pred = kf.predict()
    assert abs(pred[0] - 5.0 * 30) < 2.0  # predicts the next position
    assert abs(kf.x[4] - 5.0) < 0.5  # learned the x velocity


def test_steady_object_keeps_one_id():
    tr = ByteTracker()
    ids = set()
    first = None
    for f in range(30):
        out = tr.step([det(box_at(5 * f))], f)
        if out and first is None:
            first = f
        ids |= {t.id for t in out}
    assert ids == {1}
    assert first == 1  # confirmed on the second hit


def test_id_survives_short_occlusion():
    tr = ByteTracker()
    seen = []
    for f in range(40):
        dets = None if 12 <= f < 17 else [det(box_at(5 * f))]
        seen += [(f, t.id, t.detected) for t in tr.step(dets, f)]
    assert {i for _, i, _ in seen} == {1}
    gap = [d for f, _, d in seen if 12 <= f < 17]
    assert gap and not any(gap)  # coasted on the prediction, no detections


def test_track_is_dropped_after_max_misses():
    tr = ByteTracker(TrackerConfig(max_misses=3))
    for f in range(5):
        tr.step([det(box_at(5 * f))], f)
    assert tr.n_confirmed == 1
    for f in range(5, 5 + 5):
        tr.step([], f)
    assert tr.n_confirmed == 0


def test_crossing_objects_keep_their_ids():
    tr = ByteTracker()
    seq = {1: [], 2: []}
    for f in range(40):
        a = box_at(6 * f, y=100)  # moves right
        b = box_at(240 - 6 * f, y=112)  # moves left, overlaps A around f=20
        out = tr.step([det(a), det(b)], f)
        for t in out:
            cx = (t.box[0] + t.box[2]) / 2
            if f >= 30:
                seq[1 if cx > 120 else 2].append(t.id)  # after the cross: A is on the right
    assert set(seq[1]) == {1} and set(seq[2]) == {2}


def test_class_gate_blocks_cross_class_match():
    tr = ByteTracker()
    for f in range(4):
        tr.step([det(box_at(3 * f), cls="person")], f)
    for f in range(4, 8):
        tr.step([det(box_at(3 * f), cls="car")], f)
    classes = {(t.id, t.cls) for t in tr.step([det(box_at(24), cls="car")], 8)}
    assert all(c == "car" for _, c in classes)
    assert 1 not in {i for i, _ in classes}  # the car did not inherit the person's id


def test_low_score_second_pass_keeps_track_detected():
    def run(two_stage: bool) -> list[bool]:
        tr = ByteTracker(TrackerConfig(two_stage=two_stage))
        flags = []
        for f in range(20):
            score = 0.9 if f < 10 else 0.3  # occlusion-like drop in confidence
            out = tr.step([det(box_at(4 * f), score=score)], f)
            if f >= 10:
                flags.append(bool(out and out[0].detected))
        return flags

    assert all(run(True))
    assert not any(run(False))


def test_new_tracks_need_a_confident_detection():
    tr = ByteTracker()
    for f in range(10):
        assert tr.step([det(box_at(2 * f), score=0.3)], f) == []


def test_predict_only_frames_follow_velocity():
    tr = ByteTracker()
    for f in range(10):
        tr.step([det(box_at(5 * f))], f)
    xs = [tr.step(None, 10 + k)[0].box[0] for k in range(3)]
    assert xs[0] < xs[1] < xs[2]
    assert abs((xs[1] - xs[0]) - 5.0) < 1.5


def test_observation_centric_update_runs_and_keeps_id():
    tr = ByteTracker(TrackerConfig(ocr_update=True))
    ids = set()
    for f in range(40):
        dets = None if 10 <= f < 16 else [det(box_at(4 * f))]
        if f == 15:
            dets = [det(box_at(4 * f))]
        ids |= {t.id for t in tr.step(dets, f)}
    assert ids == {1}
