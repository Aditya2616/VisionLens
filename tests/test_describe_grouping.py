import pytest

from visionlens.describe import Facts, TemplateDescriber, check_faithful
from visionlens.describe.template import build_sentences, subject
from visionlens.schemas import Event


def ev(i, t, kind, ids, classes, t_end=None, **data):
    return Event(id=f"e{i}", t=t, kind=kind, ids=ids, classes=classes, t_end=t_end, data=data)


def street_events():
    """Three parked cars and two walkers that enter together, one leaves, two come close."""
    out = []
    k = 0
    for tid in (1, 2, 3):
        out.append(ev(k, 0.0, "linger", [tid], ["car"], 8.0, whole_stay=True, duration=8.0))
        k += 1
    for tid in (4, 5):
        out.append(ev(k, 2.0, "enter", [tid], ["person"], edge="left"))
        k += 1
        out.append(
            ev(
                k,
                2.0,
                "move",
                [tid],
                ["person"],
                6.0,
                direction="right",
                speed="at a steady pace",
                from_start=False,
            )
        )
        k += 1
    out.append(ev(k, 6.0, "leave", [4], ["person"], edge="right"))
    k += 1
    out.append(ev(k, 4.0, "proximity", [4, 5], ["person", "person"], 6.0, duration=2.0))
    return sorted(out, key=lambda e: e.t)


SUMMARY = {"tracks": 5, "by_class": {"car": 3, "person": 2}, "peak": 5}


def facts():
    f = Facts(8.0, 640, 360, street_events(), SUMMARY)
    TemplateDescriber().describe(f)
    return f


def test_same_kind_and_timing_are_merged_into_one_statement():
    text = [s.text for s in build_sentences(street_events(), 8.0, SUMMARY)]
    assert text[1] == "Three cars (1, 2, 3) stayed where they were throughout."
    assert "At 0:02, two people (4, 5) entered from the left and moved right." in text
    # one sentence for the pair, not one per track
    assert sum("entered" in t for t in text) == 1


def test_sentences_map_to_the_events_they_state():
    events = street_events()
    by_id = {e.id: e for e in events}
    for s in build_sentences(events, 8.0, SUMMARY)[1:]:
        assert s.event_ids, s.text
        assert all(i in by_id for i in s.event_ids)
    merged = next(s for s in build_sentences(events, 8.0, SUMMARY) if "entered" in s.text)
    kinds = sorted(by_id[i].kind for i in merged.event_ids)
    assert kinds == ["enter", "enter", "move", "move"]
    assert {t for i in merged.event_ids for t in by_id[i].ids} == {4, 5}


def test_long_scenes_are_capped_and_the_rest_is_counted():
    events = []
    for k in range(20):
        events.append(ev(k, k * 0.5, "enter", [k + 1], ["car"], edge="left" if k % 2 else "top"))
    s = build_sentences(events, 12.0, {"tracks": 20, "by_class": {"car": 20}, "peak": 4})
    assert len(s) <= 1 + 8 + 1
    assert "shown on the timeline" in s[-1].text
    cited = {i for x in s for i in x.event_ids}
    assert cited == {e.id for e in events}  # nothing is dropped silently


def test_subject_phrase():
    assert subject([("car", 4)]) == "car 4"
    assert subject([("person", 2), ("person", 1), ("car", 3)]) == "two people (1, 2) and car 3"
    assert subject([("bus", 1), ("bus", 2)]) == "two buses (1, 2)"


GOOD = (
    "Three cars (1, 2, 3) stayed put the whole time. At 0:02 two people (4, 5) came in from the "
    "left and walked right, and person 4 left toward the right at 0:06."
)


def test_guard_accepts_a_faithful_paraphrase():
    ok, why = check_faithful(GOOD, facts())
    assert ok, why


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (GOOD.replace("from the left", "from the right"), "entry edge"),
        (GOOD.replace("walked right", "walked left"), "movement"),
        (GOOD.replace("toward the right", "toward the left"), "exit"),
        (GOOD.replace("Three cars", "Four cars"), "does not match"),
        (GOOD.replace("(1, 2, 3)", "(1, 2, 9)"), "not a tracked"),
        (GOOD + " A bus stopped at 0:05.", "bus"),
        ("Person 5 left toward the right at 0:06.", "exit"),
        ("Car 1 drove in from the left and then stopped.", "entry"),
        ("Person 4 and car 1 came close together at 0:03.", "close together"),
        ("Two people (4, 5) came in from the left at 0:08 and walked right.", "time 0:08"),
        ("Five cars (1, 2, 3) stayed where they were throughout the clip.", "does not match"),
    ],
)
def test_guard_rejects_corrupted_rewrites(bad, reason):
    ok, why = check_faithful(bad, facts())
    assert not ok and reason in why, why
