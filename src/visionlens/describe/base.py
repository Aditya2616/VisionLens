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
    lines: list[str] = field(default_factory=list)  # the same facts as a compact structured list
    kind: str = "video"

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


@dataclass
class Feedback:
    """What the guard said about the previous attempt, fed back into the next one."""

    text: str
    reason: str
    sentence: str | None = None


class Describer(Protocol):
    name: str

    def describe(
        self,
        facts: Facts,
        on_token: TokenCallback | None = None,
        feedback: Feedback | None = None,
    ) -> DescriptionVariant: ...


MAX_ATTEMPTS = 3


# ----------------------------------------------------------------------------- guard
def check_detail(text: str, facts: Facts) -> tuple[bool, str | None, str | None]:
    """(accepted, reason, offending sentence); see describe/guard.py for the rules."""
    from .guard import check_detail as _check
    from .template import build_sentences

    facts_text = facts.text or " ".join(
        s.text for s in build_sentences(facts.events, facts.duration, facts.summary)
    )
    return _check(text, facts.events, facts.summary, facts_text, facts.duration)


def check_faithful(text: str, facts: Facts) -> tuple[bool, str | None]:
    """Reject text that the event list does not support (see describe/guard.py for the rules)."""
    ok, why, _ = check_detail(text, facts)
    return ok, why


# ------------------------------------------------------------------------ orchestration
def build_description(
    facts: Facts,
    template: DescriptionVariant,
    model: Describer | None = None,
    on_token: TokenCallback | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> Description:
    """Template always; a model rewrite on top when given, kept only if the guard accepts it.

    A rejected rewrite is regenerated, up to `max_attempts` times in all, with the guard's reason
    and the offending sentence fed back to the model (only for describers that accept feedback).
    The first attempt streams its tokens; the result is the first accepted attempt, or the last
    rejected one with its reason (the template is then what is shown).
    """
    import time

    desc = Description(template=template, used="template")
    if model is None or not facts.sentences:
        return desc
    can_retry = bool(getattr(model, "supports_feedback", False))
    attempts = max_attempts if can_retry else 1
    log: list[str] = []
    feedback: Feedback | None = None
    variant: DescriptionVariant | None = None
    t0 = time.perf_counter()
    for k in range(attempts):
        try:
            variant = (
                model.describe(facts, on_token if k == 0 else None, feedback)
                if k
                else model.describe(facts, on_token)
            )
        except Exception as exc:  # noqa: BLE001 - a failing model must never break the pipeline
            variant = DescriptionVariant(
                source="llm", text="", accepted=False, rejected_reason=f"model error: {exc}"
            )
            log.append(variant.rejected_reason or "")
            break
        if not variant.text:
            variant.accepted, variant.rejected_reason = False, "empty answer"
            log.append("empty answer")
            continue
        ok, why, sentence = check_detail(variant.text, facts)
        variant.accepted, variant.rejected_reason = ok, why
        if ok:
            break
        log.append(why or "")
        feedback = Feedback(variant.text, why or "", sentence)
    assert variant is not None
    variant.attempts = len(log) + (1 if variant.accepted else 0)
    variant.attempt_log = log
    variant.seconds = round(time.perf_counter() - t0, 1)
    desc.model = variant
    desc.used = "model" if variant.accepted and variant.text else "template"
    return desc
