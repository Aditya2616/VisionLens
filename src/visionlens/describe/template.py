"""Deterministic template describer: the baseline, and the fallback for every other describer."""

from __future__ import annotations

from collections import Counter

from ..schemas import DescriptionVariant, Detection, Event, Sentence
from .base import Facts, TokenCallback

PLURAL = {"person": "people", "bus": "buses"}
MAX_SENTENCES = 14


def plural(cls: str, n: int) -> str:
    if n == 1:
        return cls
    return PLURAL.get(cls, cls + "s")


def fmt_t(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def counts_phrase(by_class: dict[str, int]) -> str:
    parts = [f"{n} {plural(c, n)}" for c, n in sorted(by_class.items(), key=lambda kv: -kv[1])]
    if not parts:
        return "nothing"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def name(cls: str, tid: int) -> str:
    return f"{cls} {tid}"


def _enter_text(e: Event) -> str:
    edge = e.data.get("edge")
    who = name(e.classes[0], e.ids[0])
    if edge in ("left", "right"):
        return f"{who} entered from the {edge}"
    if edge in ("top", "bottom"):
        return f"{who} entered from the {edge}"
    return f"{who} appeared in the scene"


def _leave_text(e: Event) -> str:
    edge = e.data.get("edge")
    who = name(e.classes[0], e.ids[0])
    if edge:
        return f"{who} left toward the {edge}"
    return f"{who} disappeared from view"


def build_sentences(events: list[Event], duration: float, summary: dict) -> list[Sentence]:
    out: list[Sentence] = []
    tracks, by_class, peak = (
        summary.get("tracks", 0),
        summary.get("by_class", {}),
        summary.get("peak", 0),
    )
    if tracks == 0:
        return [Sentence(text=f"Nothing was tracked in this {duration:.0f}-second clip.")]
    head = (
        f"{tracks} {'object was' if tracks == 1 else 'objects were'} tracked over "
        f"{duration:.0f} seconds: {counts_phrase(by_class)}."
    )
    if peak > 1:
        head += f" Up to {peak} were in view at once."
    out.append(Sentence(text=head))

    by_track_move: dict[int, Event] = {e.ids[0]: e for e in events if e.kind == "move"}
    used_move: set[str] = set()
    body: list[tuple[int, float, Sentence]] = []  # (priority, time, sentence)
    for e in events:
        if e.kind == "enter":
            txt = _enter_text(e)
            ids = [e.id]
            mv = by_track_move.get(e.ids[0])
            if mv is not None and not mv.data.get("from_start"):
                txt += f" and moved {mv.data['direction']} {mv.data['speed']}"
                ids.append(mv.id)
                used_move.add(mv.id)
            body.append((1, e.t, Sentence(text=f"At {fmt_t(e.t)}, {txt}.", event_ids=ids)))
        elif e.kind == "leave":
            body.append(
                (1, e.t, Sentence(text=f"At {fmt_t(e.t)}, {_leave_text(e)}.", event_ids=[e.id]))
            )
        elif e.kind == "linger":
            who = name(e.classes[0], e.ids[0])
            if e.data.get("whole_stay"):
                txt = f"{who} stayed in place for the whole time it was in view"
            else:
                txt = f"{who} stopped for {e.data['duration']:.0f} {'second' if round(e.data['duration']) == 1 else 'seconds'}"
            body.append((0, e.t, Sentence(text=f"At {fmt_t(e.t)}, {txt}.", event_ids=[e.id])))
        elif e.kind == "proximity":
            a, b = name(e.classes[0], e.ids[0]), name(e.classes[1], e.ids[1])
            d = e.data["duration"]
            body.append(
                (
                    1,
                    e.t,
                    Sentence(
                        text=f"At {fmt_t(e.t)}, {a} and {b} came close together for "
                        f"{d:.0f} {'second' if round(d) == 1 else 'seconds'}.",
                        event_ids=[e.id],
                    ),
                )
            )
    for e in events:  # movers that were already in view when the clip began
        if e.kind == "move" and e.id not in used_move and e.data.get("from_start"):
            body.append(
                (
                    2,
                    e.t,
                    Sentence(
                        text=f"{name(e.classes[0], e.ids[0]).capitalize()} was already in view and moved "
                        f"{e.data['direction']} {e.data['speed']}.",
                        event_ids=[e.id],
                    ),
                )
            )
    counts = [e for e in events if e.kind == "count_change"]
    if counts and peak > 1:
        top = max(counts, key=lambda e: e.data["count"])
        if top.data["count"] == peak:
            body.append(
                (
                    2,
                    top.t,
                    Sentence(
                        text=f"The busiest moment was at {fmt_t(top.t)}, with {top.data['count']} in view.",
                        event_ids=[top.id],
                    ),
                )
            )
    body.sort(key=lambda x: x[1])
    if len(body) > MAX_SENTENCES:
        keep = sorted(body, key=lambda x: (x[0], x[1]))[:MAX_SENTENCES]
        extra = len(body) - MAX_SENTENCES
        body = sorted(keep, key=lambda x: x[1])
        tail = Sentence(
            text=f"{extra} smaller {'event is' if extra == 1 else 'events are'} "
            "left out here and shown on the timeline."
        )
        return out + [s for _, _, s in body] + [tail]
    return out + [s for _, _, s in body]


class TemplateDescriber:
    name = "template"

    def describe(self, facts: Facts, on_token: TokenCallback | None = None) -> DescriptionVariant:
        sentences = build_sentences(facts.events, facts.duration, facts.summary)
        facts.sentences = sentences
        text = " ".join(s.text for s in sentences)
        if on_token:
            for s in sentences:
                on_token(s.text + " ")
        return DescriptionVariant(source="template", text=text, sentences=sentences)


def describe_image(dets: list[Detection], width: int) -> DescriptionVariant:
    """Still-image variant: counts and rough horizontal position, no tracking involved."""
    if not dets:
        s = [Sentence(text="No objects were found in this image.")]
        return DescriptionVariant(source="template", text=s[0].text, sentences=s)
    c = Counter(d.cls for d in dets)
    sentences = [
        Sentence(
            text=f"Found {sum(c.values())} {'object' if len(dets) == 1 else 'objects'}"
            f": {counts_phrase(dict(c))}."
        )
    ]
    for cls in c:
        pos = []
        for d in dets:
            if d.cls == cls:
                cx = (d.box[0] + d.box[2]) / 2 / width
                pos.append(
                    "on the left" if cx < 0.34 else "on the right" if cx > 0.66 else "in the centre"
                )
        pc = Counter(pos)
        bits = ", ".join(f"{n} {w}" for w, n in pc.items())
        sentences.append(Sentence(text=f"The {plural(cls, c[cls])}: {bits}."))
    return DescriptionVariant(
        source="template", text=" ".join(s.text for s in sentences), sentences=sentences
    )
