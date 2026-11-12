from __future__ import annotations

from visionlens.events import EventConfig, extract_events, summarize
from visionlens.schemas import FrameResult, TrackOut

W, H, FPS = 200, 100, 10.0


def frames_from(tracks_by_frame, n):
    out = []
    for f in range(n):
        tr = [
            TrackOut(id=i, box=b, cls=c, score=0.9, detected=True)
            for i, b, c in tracks_by_frame.get(f, [])
        ]
        out.append(FrameResult(i=f, t=f / FPS, tracks=tr))
    return out


def kinds(events):
    return [e.kind for e in events]


def test_enter_move_leave_with_edges():
    data = {
        f: [(1, (6.0 * (f - 5), 40.0, 6.0 * (f - 5) + 20, 80.0), "person")] for f in range(5, 36)
    }
    ev = extract_events(frames_from(data, 50), W, H, FPS)
    enter = next(e for e in ev if e.kind == "enter")
    leave = next(e for e in ev if e.kind == "leave")
    move = next(e for e in ev if e.kind == "move")
    assert enter.t == 0.5 and enter.data["edge"] == "left"
    assert leave.t == 3.5 and leave.data["edge"] == "right"
    assert move.data["direction"] == "right" and move.data["speed"] == "fast"
    assert {e.id for e in ev} == {f"e{i}" for i in range(len(ev))}


def test_stationary_object_from_the_start_lingers_for_the_whole_stay():
    data = {f: [(1, (80.0 + (f % 2) * 0.5, 40.0, 100.0, 80.0), "car")] for f in range(0, 50)}
    ev = extract_events(frames_from(data, 50), W, H, FPS)
    assert "enter" not in kinds(ev) and "leave" not in kinds(ev)
    ling = next(e for e in ev if e.kind == "linger")
    assert ling.data["whole_stay"] is True and ling.data["duration"] >= 3


def test_proximity_requires_closeness_for_long_enough():
    a = (50.0, 40.0, 70.0, 80.0)
    near = (72.0, 40.0, 92.0, 80.0)
    far = (150.0, 40.0, 170.0, 80.0)
    data = {f: [(1, a, "person"), (2, near if f < 30 else far, "person")] for f in range(50)}
    ev = extract_events(frames_from(data, 50), W, H, FPS)
    prox = [e for e in ev if e.kind == "proximity"]
    assert len(prox) == 1 and prox[0].ids == [1, 2] and prox[0].t_end is not None
    assert prox[0].t_end < 3.2
    # the same boxes close for only 0.3 s are too brief to report
    data = {f: [(1, a, "person"), (2, near if f < 3 else far, "person")] for f in range(50)}
    assert "proximity" not in kinds(extract_events(frames_from(data, 50), W, H, FPS))


def test_feet_far_apart_is_not_proximity_even_if_boxes_overlap():
    tall_low = (50.0, 10.0, 90.0, 90.0)  # big box, feet low in the frame
    small_back = (60.0, 20.0, 70.0, 40.0)  # small box higher up, overlapping on screen
    data = {f: [(1, tall_low, "person"), (2, small_back, "person")] for f in range(50)}
    assert "proximity" not in kinds(extract_events(frames_from(data, 50), W, H, FPS))


def test_count_change_is_debounced():
    def box(x):
        return (x, 40.0, x + 20, 80.0)

    data = {}
    for f in range(50):
        tr = [(1, box(20.0), "person")]
        if f >= 20:
            tr.append((2, box(120.0), "person"))
        if f == 35:
            tr.append((3, box(160.0), "person"))  # a one-frame flicker must not count
        data[f] = tr
    ev = extract_events(frames_from(data, 50), W, H, FPS)
    counts = [e.data["count"] for e in ev if e.kind == "count_change"]
    assert counts == [2]


def test_short_flicker_tracks_are_ignored_and_summary_counts_tracks():
    data = {f: [(1, (10.0, 40.0, 30.0, 80.0), "person")] for f in range(50)}
    data.update({f: data[f] + [(9, (100.0, 40.0, 120.0, 80.0), "car")] for f in range(20, 23)})
    fr = frames_from(data, 50)
    ev = extract_events(fr, W, H, FPS)
    assert all(9 not in e.ids for e in ev)
    s = summarize(fr, FPS, EventConfig())
    assert s["tracks"] == 1 and s["by_class"] == {"person": 1}
