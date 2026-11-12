"""Deterministic template describer: the baseline, and the fallback for every other describer."""

from __future__ import annotations

from collections import Counter

from ..schemas import DescriptionVariant, Detection, Event, Sentence
from .base import Facts, TokenCallback

PLURAL = {"person": "people", "bus": "buses", "bicycle": "bicycles"}


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


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
MAX_BODY = 8  # sentences after the opening one; the rest is folded into a closing count


def num(n: int) -> str:
    return WORDS[n] if n < len(WORDS) else str(n)


def subject(members: list[tuple[str, int]]) -> str:
    """'car 4', or 'three cars (1, 2, 3) and person 5', for the (class, id) pairs given."""
    by: dict[str, list[int]] = {}
    for cls, tid in members:
        by.setdefault(cls, []).append(tid)
    parts = []
    for cls, ids in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ids = sorted(ids)
        if len(ids) == 1:
            parts.append(name(cls, ids[0]))
        else:
            parts.append(f"{num(len(ids))} {plural(cls, len(ids))} ({', '.join(map(str, ids))})")
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def _secs(d: float) -> str:
    return f"{d:.0f} {'second' if round(d) == 1 else 'seconds'}"


def _bucket_groups(items: list, key, window: float = 1.0) -> list[list]:
    """Group items with an equal key whose times lie within `window` seconds of the group's first."""
    open_groups: dict = {}
    out: list[list] = []
    for it in sorted(items, key=lambda x: x[0].t):
        k = key(it)
        g = open_groups.get(k)
        if g is None or it[0].t - g[0][0].t > window:
            g = [it]
            open_groups[k] = g
            out.append(g)
        else:
            g.append(it)
    return out


def build_sentences(events: list[Event], duration: float, summary: dict) -> list[Sentence]:
    """Summarise the events the way a person would: counts, grouped statements, notable events.

    Deterministic. Every sentence lists the ids of the events it states, so the UI can link a
    sentence to its tracks and timeline marks.
    """
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
    out = [Sentence(text=head)]

    enter = {e.ids[0]: e for e in events if e.kind == "enter"}
    leave = {e.ids[0]: e for e in events if e.kind == "leave"}
    move = {e.ids[0]: e for e in events if e.kind == "move"}
    whole = {e.ids[0]: e for e in events if e.kind == "linger" and e.data.get("whole_stay")}
    stops = [e for e in events if e.kind == "linger" and not e.data.get("whole_stay")]

    # (tier, size, time, sentence); tier 0 always stays, higher tiers are dropped first when capped
    body: list[tuple[int, int, float, Sentence]] = []

    def add(tier: int, size: int, t: float, text: str, ids: list[str]) -> None:
        body.append((tier, size, t, Sentence(text=text, event_ids=ids)))

    # 1. objects that never moved and were there the whole clip
    still = [i for i in whole if i not in enter and i not in leave]
    if still:
        mem = [(whole[i].classes[0], i) for i in still]
        verb = "stayed where they were" if len(still) > 1 else "stayed where it was"
        add(
            0,
            len(still),
            0.0,
            f"{_upper(subject(mem))} {verb} throughout.",
            [whole[i].id for i in still],
        )
    # 2. objects already moving when the clip began
    starters = [(m, i) for i, m in move.items() if m.data.get("from_start") and i not in enter]
    for g in _bucket_groups(starters, lambda x: x[0].data["direction"], window=1e9):
        mem = [(m.classes[0], i) for m, i in g]
        d = g[0][0].data
        many = len(g) > 1
        speed = "" if many else f" {d['speed']}"
        add(
            0,
            len(g),
            0.0,
            f"{_upper(subject(mem))} {'were' if many else 'was'} already in view and moved "
            f"{d['direction']}{speed}.",
            [m.id for m, _ in g],
        )

    # 3. entries, merged with what the same track then did
    def tail(i: int) -> tuple[str, list[str], str]:
        m = move.get(i)
        if m is not None and not m.data.get("from_start"):
            return f"moved {m.data['direction']}", [m.id], m.data["speed"]
        if i in whole:
            return "stayed in place", [whole[i].id], ""
        return "", [], ""

    entries = []
    for i, e in enter.items():
        txt, ids, speed = tail(i)
        entries.append((e, i, txt, ids, speed))
    for g in _bucket_groups(entries, lambda x: (x[0].data.get("edge"), x[2])):
        e0, _, txt, _, speed = g[0]
        edge = e0.data.get("edge")
        mem = [(x[0].classes[0], x[1]) for x in g]
        where = f"entered from the {edge}" if edge else "appeared in the scene"
        if txt:
            where += f" and {txt}" + (f" {speed}" if len(g) == 1 and speed else "")
        ids = [x[0].id for x in g] + [y for x in g for y in x[3]]
        tier = 1 if (len(g) > 1 or txt) else 2
        add(tier, len(g), e0.t, f"At {fmt_t(e0.t)}, {subject(mem)} {where}.", ids)
    # 4. stops
    for g in _bucket_groups([(e, e.ids[0]) for e in stops], lambda x: round(x[0].data["duration"])):
        mem = [(x[0].classes[0], x[1]) for x in g]
        e0 = g[0][0]
        add(
            1,
            len(g),
            e0.t,
            f"At {fmt_t(e0.t)}, {subject(mem)} stopped for {_secs(e0.data['duration'])}.",
            [x[0].id for x in g],
        )
    # 5. objects coming close together: the two longest get a sentence, the rest one summary
    prox = sorted((e for e in events if e.kind == "proximity"), key=lambda e: -e.data["duration"])
    for e in prox[:2]:
        a, b = (name(c, i) for c, i in zip(e.classes, e.ids, strict=True))
        add(
            1,
            2,
            e.t,
            f"At {fmt_t(e.t)}, {a} and {b} came close together for {_secs(e.data['duration'])}.",
            [e.id],
        )
    if len(prox) > 2:
        rest = prox[2:]
        n = len(rest)
        add(
            2,
            n,
            min(e.t for e in rest),
            f"{_upper(num(n))} other {'pair' if n == 1 else 'pairs'} also came close together.",
            [e.id for e in rest],
        )
    # 6. exits
    exits = [(e, e.ids[0]) for e in leave.values()]
    for g in _bucket_groups(exits, lambda x: x[0].data.get("edge")):
        e0 = g[0][0]
        edge = e0.data.get("edge")
        mem = [(x[0].classes[0], x[1]) for x in g]
        what = f"left toward the {edge}" if edge else "disappeared from view"
        add(2, len(g), e0.t, f"At {fmt_t(e0.t)}, {subject(mem)} {what}.", [x[0].id for x in g])
    # 7. busiest moment
    counts = [e for e in events if e.kind == "count_change"]
    if counts and peak > 1:
        top = max(counts, key=lambda e: e.data["count"])
        if top.data["count"] == peak:
            add(
                2,
                0,
                top.t,
                f"The busiest moment was at {fmt_t(top.t)}, with {top.data['count']} in view.",
                [top.id],
            )

    kept = body
    if len(body) > MAX_BODY:
        kept = sorted(body, key=lambda x: (x[0], -x[1], x[2]))[:MAX_BODY]
    # clip-wide statements first, then everything else in time order
    kept = sorted(kept, key=lambda x: (x[0] != 0, x[2]))
    out += [x[3] for x in kept]
    shown = {eid for x in kept for eid in x[3].event_ids}
    left_out = [
        e
        for e in events
        if e.id not in shown and e.kind in ("enter", "leave", "linger", "proximity", "move")
    ]
    if left_out:
        n = len(left_out)
        out.append(
            Sentence(
                text=f"{n} further {'event is' if n == 1 else 'events are'} shown on the timeline.",
                event_ids=[e.id for e in left_out],
            )
        )
    return out


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
