"""Describer interface, the faithfulness guard and the orchestration around them."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..schemas import Description, DescriptionVariant, Event, Sentence

TokenCallback = Callable[[str], None]


@dataclass
class Facts:
    """Everything a describer may talk about. Nothing outside this is allowed in the text."""

    duration: float
    width: int
    height: int
    events: list[Event]
    summary: dict
    sentences: list[Sentence] = field(default_factory=list)  # the template's facts, in order
    kind: str = "video"

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


class Describer(Protocol):
    name: str

    def describe(
        self, facts: Facts, on_token: TokenCallback | None = None
    ) -> DescriptionVariant: ...


# ----------------------------------------------------------------------------- guard
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_NOUNS = {
    "person": "person",
    "people": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "man": "person",
    "woman": "person",
    "men": "person",
    "women": "person",
    "child": "person",
    "car": "car",
    "cars": "car",
    "vehicle": "car",
    "vehicles": "car",
    "truck": "truck",
    "trucks": "truck",
    "bus": "bus",
    "buses": "bus",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "bike": "bicycle",
    "bikes": "bicycle",
    "cyclist": "bicycle",
    "motorcycle": "motorcycle",
    "dog": "dog",
    "dogs": "dog",
    "cat": "cat",
    "cats": "cat",
    "bird": "bird",
    "birds": "bird",
    "horse": "horse",
    "boat": "boat",
    "train": "train",
}
_LABEL_ID = re.compile(
    r"\b(person|car|truck|bus|bicycle|motorcycle|dog|cat|bird|horse|boat|train)"
    r"\s*#?(\d+)\b",
    re.I,
)


def _ints(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", text)}


def check_faithful(text: str, facts: Facts) -> tuple[bool, str | None]:
    """Reject text that mentions a track, number or object class that is not in the facts.

    This is a deliberately cheap lexical check, not a semantic proof: it catches invented ids,
    counts, times and objects, which are the failures small models actually produce here.
    """
    ftext = facts.text.lower()
    low = text.lower()
    allowed = _ints(ftext)
    for n in _ints(low):
        if n not in allowed:
            return False, f"number {n} is not in the facts"
    for w, v in _NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", low) and v not in allowed:
            return False, f"count '{w}' is not in the facts"
    fact_pairs = {(m.group(1).lower(), int(m.group(2))) for m in _LABEL_ID.finditer(ftext)}
    for m in _LABEL_ID.finditer(text):
        if (m.group(1).lower(), int(m.group(2))) not in fact_pairs:
            return False, f"'{m.group(0)}' is not a tracked object"
    fact_classes = {_NOUNS[w] for w in re.findall(r"[a-z]+", ftext) if w in _NOUNS}
    for w in re.findall(r"[a-z]+", low):
        if w in _NOUNS and _NOUNS[w] not in fact_classes:
            return False, f"object '{w}' is not in the facts"
    if len(text.strip()) < 20:
        return False, "text too short"
    return True, None


# ------------------------------------------------------------------------ orchestration
def build_description(
    facts: Facts,
    template: DescriptionVariant,
    model: Describer | None = None,
    on_token: TokenCallback | None = None,
) -> Description:
    """Template always; a model rewrite on top when given, kept only if the guard accepts it."""
    desc = Description(template=template, used="template")
    if model is None or not facts.sentences:
        return desc
    try:
        variant = model.describe(facts, on_token)
    except Exception as exc:  # noqa: BLE001 - a failing model must never break the pipeline
        variant = DescriptionVariant(
            source="llm", text="", accepted=False, rejected_reason=f"model error: {exc}"
        )
    if variant.text:
        ok, why = check_faithful(variant.text, facts)
        variant.accepted, variant.rejected_reason = ok, why
    desc.model = variant
    desc.used = "model" if variant.accepted and variant.text else "template"
    return desc
