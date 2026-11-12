"""Describer interface, the faithfulness guard and the orchestration around them."""

from __future__ import annotations

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
def check_faithful(text: str, facts: Facts) -> tuple[bool, str | None]:
    """Reject text that the event list does not support (see describe/guard.py for the rules)."""
    from .guard import check_faithful as _check
    from .template import build_sentences

    facts_text = facts.text or " ".join(
        s.text for s in build_sentences(facts.events, facts.duration, facts.summary)
    )
    return _check(text, facts.events, facts.summary, facts_text, facts.duration)


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
