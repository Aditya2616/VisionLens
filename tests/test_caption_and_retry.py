from __future__ import annotations

import numpy as np

from visionlens.caption import (
    SceneCaptioner,
    first_sentence,
    named_classes,
    pick_keyframes,
)
from visionlens.describe import Facts, TemplateDescriber, build_description
from visionlens.pipeline import PipelineConfig
from visionlens.runner import analyze_file
from visionlens.schemas import DescriptionVariant, Event


class FakeCaptioner(SceneCaptioner):
    """The real wrapper logic with the model call replaced by canned answers."""

    def __init__(self, answers):
        super().__init__(device="cpu")
        self.answers = list(answers)

    def generate(self, frame_bgr):
        return self.answers.pop(0)


def test_one_sentence_and_named_classes():
    assert first_sentence("A street. With cars.") == "A street."
    assert first_sentence("  A  street\nwith cars  ") == "A street with cars"
    assert named_classes("A street with two cars, buses and pedestrians and a car") == [
        "car",
        "bus",
        "person",
    ]
    assert named_classes("A quiet village road with trees") == []


def test_keyframes_are_spread_over_the_clip():
    thumbs = [(i, i * 0.5, np.zeros((4, 4, 3), np.uint8)) for i in range(21)]
    times = [t for t, _ in pick_keyframes(thumbs)]
    assert times == [1.0, 5.0, 9.0]
    assert len(pick_keyframes(thumbs[:1])) == 1  # a still image gives one frame


def test_caption_names_are_checked_against_what_the_detector_saw():
    frames = [(0.0, np.zeros((8, 8, 3), np.uint8))] * 3
    cap = FakeCaptioner(["A road.", "A street with a bus and cars.", "A road."])
    sc = cap.caption(frames, {"car"})
    assert sc.text == "A street with a bus and cars."
    assert sc.named == ["bus", "car"] and sc.unseen == ["bus"] and sc.flagged
    assert [f.text for f in sc.frames] == ["A road.", "A street with a bus and cars.", "A road."]
    ok = FakeCaptioner(["a", "A street with cars.", "b"]).caption(frames, {"car"})
    assert not ok.flagged and ok.unseen == []


def test_pipeline_adds_a_separate_scene_line_and_leaves_the_description_alone(scene_video, oracle):
    plain = analyze_file(scene_video, oracle, PipelineConfig(stride=2), target_fps=10, name="c")
    cap = FakeCaptioner(["The start.", "A street with people.", "The end."])
    res = analyze_file(
        scene_video, oracle, PipelineConfig(stride=2), target_fps=10, name="c", captioner=cap
    )
    assert plain.scene is None
    assert res.scene is not None and res.scene.text == "A street with people."
    assert res.timings.caption_s == res.scene.seconds
    assert res.description.template.text == plain.description.template.text  # never merged
    assert "street" not in res.description.template.text


def make_facts():
    events = [
        Event(id="e0", t=2.0, kind="enter", ids=[1], classes=["person"], data={"edge": "left"}),
        Event(id="e1", t=6.0, kind="leave", ids=[1], classes=["person"], data={"edge": "right"}),
    ]
    f = Facts(10.0, 640, 360, events, {"tracks": 1, "by_class": {"person": 1}, "peak": 1})
    return f, TemplateDescriber().describe(f)


class Scripted:
    """A describer that answers from a script and records the feedback it was given."""

    name = "llm"
    supports_feedback = True

    def __init__(self, answers):
        self.answers = list(answers)
        self.feedback = []

    def describe(self, facts, on_token=None, feedback=None):
        self.feedback.append(feedback)
        return DescriptionVariant(source="llm", text=self.answers.pop(0), model="stub")


def test_a_rejected_rewrite_is_regenerated_with_the_guards_reason():
    facts, tpl = make_facts()
    bad = "At 0:02 person 1 came in from the right and left toward the left at 0:06."
    good = "Person 1 came in from the left at 0:02 and left toward the right at 0:06."
    model = Scripted([bad, good])
    d = build_description(facts, tpl, model)
    assert d.used == "model" and d.model.accepted and d.model.attempts == 2
    assert len(d.model.attempt_log) == 1 and "entry edge" in d.model.attempt_log[0]
    assert model.feedback[0] is None
    assert "entry edge" in model.feedback[1].reason and "right" in model.feedback[1].sentence


def test_three_rejections_fall_back_to_the_template_and_say_why():
    facts, tpl = make_facts()
    bad = "Seven people came in from the right at 0:02 and then danced for a long time."
    model = Scripted([bad, bad, bad, bad])
    d = build_description(facts, tpl, model)
    assert d.used == "template" and not d.model.accepted
    assert d.model.attempts == 3 and len(d.model.attempt_log) == 3
    assert len(model.answers) == 1  # the loop stopped after three attempts


def test_describers_without_feedback_support_get_one_attempt():
    facts, tpl = make_facts()

    class Once:
        name = "x"

        def __init__(self):
            self.calls = 0

        def describe(self, facts, on_token=None):
            self.calls += 1
            return DescriptionVariant(source="llm", text="Seven people danced here for ages.")

    m = Once()
    d = build_description(facts, tpl, m)
    assert m.calls == 1 and d.model.attempts == 1 and d.used == "template"


def test_prompt_lists_the_facts_compactly_and_carries_feedback():
    from visionlens.describe.base import Feedback
    from visionlens.describe.llm import build_prompt

    facts, _ = make_facts()
    first = build_prompt(facts)
    assert "Facts (time: what happened):" in first[1]["content"]
    assert "- 0:02: person 1 entered from the left" in first[1]["content"]
    again = build_prompt(
        facts, Feedback("old text", "count 'two people' is not in the facts", "Two.")
    )
    assert [m["role"] for m in again] == ["system", "user", "assistant", "user"]
    assert "not in the facts" in again[-1]["content"] and again[2]["content"] == "old text"
