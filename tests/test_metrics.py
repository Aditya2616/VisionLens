from __future__ import annotations

import pytest

from visionlens.metrics import clear_mot, evaluate, hota, idf1

B = (0.0, 0.0, 10.0, 20.0)


def seq(ids_by_frame):
    return {f: [(i, B) for i in ids] for f, ids in enumerate(ids_by_frame)}


def test_perfect_tracking_scores_one():
    gt = seq([[1]] * 10)
    m = evaluate(gt, seq([[7]] * 10))
    assert m["MOTA"] == 1 and m["IDF1"] == 1 and m["IDsw"] == 0
    assert m["HOTA"] == pytest.approx(1.0) and m["DetA"] == pytest.approx(1.0)


def test_single_id_switch_hand_computed():
    gt = seq([[1]] * 10)
    pred = seq([[10]] * 5 + [[11]] * 5)
    c = clear_mot(gt, pred)
    assert c["IDsw"] == 1 and c["MOTA"] == pytest.approx(0.9)
    assert idf1(gt, pred)["IDF1"] == pytest.approx(2 * 5 / (2 * 5 + 5 + 5))  # 0.5
    h = hota(gt, pred)
    assert h["DetA"] == pytest.approx(1.0)
    assert h["AssA"] == pytest.approx(0.5)  # each of the two pairs: 5 / (10 + 5 - 5)
    assert h["HOTA"] == pytest.approx(0.5**0.5)


def test_missed_half_and_false_positives():
    gt = seq([[1]] * 10)
    pred = seq([[1]] * 5 + [[]] * 5)
    c = clear_mot(gt, pred)
    assert c["FN"] == 5 and c["FP"] == 0 and c["MOTA"] == pytest.approx(0.5)
    assert idf1(gt, pred)["IDF1"] == pytest.approx(2 * 5 / (2 * 5 + 0 + 5))
    pred2 = seq([[1, 2]] * 10)
    c2 = clear_mot(gt, pred2)
    assert c2["FP"] == 10 and c2["MOTA"] == pytest.approx(0.0)


def test_fragmentation_counts_gaps():
    gt = seq([[1]] * 9)
    pred = seq([[1], [1], [], [1], [1], [], [1], [1], [1]])
    assert clear_mot(gt, pred)["Frag"] == 2


def test_non_overlapping_boxes_do_not_match():
    gt = {0: [(1, (0.0, 0.0, 10.0, 10.0))]}
    pred = {0: [(1, (50.0, 50.0, 60.0, 60.0))]}
    c = clear_mot(gt, pred)
    assert c["TP"] == 0 and c["FN"] == 1 and c["FP"] == 1
