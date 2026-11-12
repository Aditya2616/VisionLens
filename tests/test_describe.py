from __future__ import annotations

import httpx
import pytest

from visionlens.describe import Facts, TemplateDescriber, build_description, check_faithful
from visionlens.describe.hosted import AnthropicDescriber, OpenAIDescriber
from visionlens.describe.template import describe_image, fmt_t
from visionlens.schemas import DescriptionVariant, Detection, Event


def make_facts():
    events = [
        Event(id="e0", t=2.0, kind="enter", ids=[1], classes=["person"], data={"edge": "left"}),
        Event(
            id="e1",
            t=2.0,
            kind="move",
            ids=[1],
            classes=["person"],
            data={"direction": "right", "speed": "at a steady pace", "from_start": False},
        ),
        Event(id="e2", t=4.0, kind="enter", ids=[2], classes=["car"], data={"edge": "right"}),
        Event(
            id="e3",
            t=6.0,
            kind="proximity",
            ids=[1, 2],
            classes=["person", "car"],
            t_end=8.0,
            data={"duration": 2.0},
        ),
        Event(id="e4", t=9.0, kind="leave", ids=[1], classes=["person"], data={"edge": "right"}),
    ]
    summary = {"tracks": 2, "by_class": {"person": 1, "car": 1}, "peak": 2}
    return Facts(duration=10.0, width=640, height=360, events=events, summary=summary)


def test_template_is_deterministic_and_cites_events():
    facts = make_facts()
    v = TemplateDescriber().describe(facts)
    assert v.text == TemplateDescriber().describe(make_facts()).text
    assert v.sentences[0].text == (
        "2 objects were tracked over 10 seconds: 1 person and 1 car. Up to 2 were in view at once."
    )
    assert "At 0:02, person 1 entered from the left and moved right at a steady pace." in v.text
    assert "At 0:06, person 1 and car 2 came close together for 2 seconds." in v.text
    assert "At 0:09, person 1 left toward the right." in v.text
    cited = {eid for s in v.sentences for eid in s.event_ids}
    assert {"e0", "e1", "e2", "e3", "e4"} <= cited


def test_empty_scene_and_time_format():
    facts = Facts(10.0, 640, 360, [], {"tracks": 0, "by_class": {}, "peak": 0})
    assert "Nothing was tracked" in TemplateDescriber().describe(facts).text
    assert fmt_t(75.4) == "1:15"


def test_guard_accepts_faithful_text():
    facts = make_facts()
    TemplateDescriber().describe(facts)
    ok, why = check_faithful(
        "A person (person 1) came in from the left at 0:02 and crossed paths with car 2 around "
        "0:06 before leaving on the right at 0:09.",
        facts,
    )
    assert ok, why


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("person 3 walked in at 0:02 and stayed for a long while", "person 3"),
        ("At 0:00 person 1 waved at car 2 for a while in the evening", "time 0:00"),
        ("Three people walked past the car 2 at 0:02 while it rained heavily", "three"),
        ("A dog ran past person 1 at 0:02 and then left towards the right", "dog"),
        ("car 1 drove through the scene at 0:04 toward the left side", "car 1"),
        ("ok", "short"),
    ],
)
def test_guard_rejects_inventions(text, key):
    facts = make_facts()
    TemplateDescriber().describe(facts)
    ok, why = check_faithful(text, facts)
    assert not ok and key in why


def test_build_description_falls_back_when_the_model_invents_things():
    facts = make_facts()
    tpl = TemplateDescriber().describe(facts)

    class Liar:
        name = "llm"

        def describe(self, facts, on_token=None):
            return DescriptionVariant(
                source="llm", text="Seven people danced for a long time.", model="stub"
            )

    d = build_description(facts, tpl, Liar())
    assert d.used == "template" and d.model is not None and not d.model.accepted

    class Honest(Liar):
        def describe(self, facts, on_token=None):
            return DescriptionVariant(
                source="llm",
                model="stub",
                text="Person 1 entered from the left at 0:02, met car 2 at 0:06 and left at 0:09.",
            )

    d = build_description(facts, tpl, Honest())
    assert d.used == "model" and d.model.accepted


def test_a_crashing_model_never_breaks_the_description():
    facts = make_facts()
    tpl = TemplateDescriber().describe(facts)

    class Boom:
        name = "llm"

        def describe(self, facts, on_token=None):
            raise RuntimeError("out of memory")

    d = build_description(facts, tpl, Boom())
    assert d.used == "template" and "out of memory" in (d.model.rejected_reason or "")


def test_trim_to_sentence_drops_a_cut_off_tail():
    from visionlens.describe.llm import trim_to_sentence

    cut = trim_to_sentence("Person 1 left the scene at 0:09 after a while. Car 2 then ent")
    assert cut == "Person 1 left the scene at 0:09 after a while."
    assert trim_to_sentence("Complete sentence here.") == "Complete sentence here."
    assert trim_to_sentence("short") == "short"


def test_image_description():
    dets = [
        Detection(box=(0, 0, 50, 90), score=0.9, cls="person"),
        Detection(box=(300, 0, 350, 90), score=0.8, cls="person"),
        Detection(box=(600, 50, 640, 90), score=0.7, cls="car"),
    ]
    v = describe_image(dets, 640)
    assert v.text.startswith("Found 3 objects: 2 people and 1 car.")
    assert "on the left" in v.text and "on the right" in v.text


def test_openai_adapter_with_stubbed_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": " Person 1 left. "}}]})

    facts = make_facts()
    TemplateDescriber().describe(facts)
    out = OpenAIDescriber(transport=httpx.MockTransport(handler)).describe(facts)
    assert out.text == "Person 1 left." and out.source == "openai"
    assert seen["auth"] == "Bearer test-key" and "person 1 entered from the left" in seen["body"]


def test_anthropic_adapter_with_stubbed_transport(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "k"
        assert request.url.path == "/v1/messages"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Car 2 came in."}]})

    facts = make_facts()
    TemplateDescriber().describe(facts)
    out = AnthropicDescriber(transport=httpx.MockTransport(handler)).describe(facts)
    assert out.text == "Car 2 came in." and out.source == "anthropic"


def test_hosted_adapters_require_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIDescriber().describe(make_facts())
