"""Faithfulness guard: does a rewritten description say anything the events do not support?

The check is rule based, on purpose. It parses the rewrite sentence by sentence and verifies against
the typed event list (not against the template wording):

- every object mentioned ("car 3", "three cars (1, 2, 3)") exists, with that class, and group
  counts match the number of ids listed;
- counts and numbers appear in the facts (class totals, group sizes, peak, durations);
- times (m:ss) lie within a second of an event of the objects the sentence is about;
- directions agree with the events of those objects: "from the left" with the entry edge,
  "toward the right" with the exit edge or direction of travel, "moved right" with the direction;
- event verbs (entered, left, stayed, stopped, moved, came close together) are supported by an event
  of the same kind for one of the mentioned objects.

It is a lexical parser, not a semantic proof: a paraphrase that keeps all ids, numbers, directions
and verbs consistent but changes the story in another way would pass. It is meant to catch what
small models actually get wrong here.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..schemas import Event

CLASS_WORDS = {
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
NUMBER_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
}
_CLS = "|".join(sorted(CLASS_WORDS, key=len, reverse=True))
_ID_LIST = r"\d+(?:\s*(?:,|and|&)\s*\d+)*"
_SINGLE = re.compile(
    r"\b(person|car|truck|bus|bicycle|motorcycle|dog|cat|bird|horse|boat|train)\s*#?(\d+)\b"
)
_GROUP = re.compile(
    rf"\b(?:(\w+)\s+)?({_CLS})\s*\(\s*({_ID_LIST})\s*\)|\b({_CLS})\s+#?({_ID_LIST})\b"
)
_GENERIC_GROUP = re.compile(r"\b[a-z]+\s*\(\s*(" + _ID_LIST + r")\s*\)")
_COUNT_NOUN = re.compile(rf"\b(\d+|{'|'.join(NUMBER_WORDS)})\s+({_CLS})\b")
_TIME = re.compile(r"\b(\d+):(\d\d)\b")
_FOR_SECONDS = re.compile(r"\bfor\s+(\d+|[a-z]+)\s+seconds?\b")

ENTER = re.compile(
    r"\b(entered|enters|came in|comes in|came into|appeared|appears|arrived|arrives|"
    r"walked in|drove in|showed up)\b"
)
LEAVE = re.compile(
    r"\b(exited|exits|departed|disappeared|walked out|drove off|drove away|went out|leaves)\b"
    r"|(?<!top )(?<!bottom )(?<!upper )(?<!lower )(?<!far )(?<!the )(?<!of )\bleft\b(?! corner| edge| side| hand|-hand| of)"
)
STAY = re.compile(
    r"\b(stayed|stays|remained|remains|stood|stationary|stopped|stops|waited|parked|paused|"
    r"lingered|motionless|did not move|didn't move|where (?:it|they) (?:was|were))\b"
)
MOVE = re.compile(
    r"\b(moved|moves|moving|walked|walking|drove|driving|travelled|traveled|crossed|headed|"
    r"heading|ran|running|rode|passed|passing|went)\b"
)
CLOSE = re.compile(
    r"\b(close together|came close|come close|met|crossed paths|next to|alongside)\b"
)
_DIR_WORD = r"(left|right|top|bottom|up|down|up-left|up-right|down-left|down-right)"
FROM_DIR = re.compile(rf"\bfrom the {_DIR_WORD}\b")
TO_DIR = re.compile(rf"\b(?:toward|towards|to|on|off|through|via) the {_DIR_WORD}\b")
MOVE_DIR = re.compile(
    r"\b(?:moved|moving|moves|walked|walking|drove|driving|headed|heading|going|"
    r"travelled|traveled|rode|goes|heads)\s+(?:(?:to|toward|towards)\s+(?:the\s+)?)?"
    rf"{_DIR_WORD}\b"
)


@dataclass
class _Index:
    cls_of: dict[int, str] = field(default_factory=dict)
    enter_edges: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    leave_edges: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    move_dirs: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    kinds: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    times: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    durations: set[int] = field(default_factory=set)
    prox: set[frozenset[int]] = field(default_factory=set)
    all_times: list[float] = field(default_factory=list)
    classes: set[str] = field(default_factory=set)
    numbers: set[int] = field(default_factory=set)
    count_pairs: set[tuple[int, str]] = field(default_factory=set)
    vehicle_totals: set[int] = field(default_factory=set)


def _build_index(events: list[Event], summary: dict, facts_text: str, duration: float) -> _Index:
    ix = _Index()
    for e in events:
        for c, i in zip(e.classes, e.ids, strict=False):
            ix.cls_of[i] = c
        ix.all_times += [e.t] + ([e.t_end] if e.t_end is not None else [])
        for i in e.ids:
            ix.kinds[i].add(e.kind)
            ix.times[i] += [e.t] + ([e.t_end] if e.t_end is not None else [])
            if e.kind == "enter" and e.data.get("edge"):
                ix.enter_edges[i].add(e.data["edge"])
            if e.kind == "leave" and e.data.get("edge"):
                ix.leave_edges[i].add(e.data["edge"])
            if e.kind == "move":
                ix.move_dirs[i] |= set(str(e.data.get("direction", "")).split("-"))
        if e.kind in ("linger", "proximity") and "duration" in e.data:
            ix.durations.add(round(e.data["duration"]))
        if e.kind == "proximity" and len(e.ids) == 2:
            ix.prox.add(frozenset(e.ids))
        if e.kind == "count_change":
            ix.numbers |= {e.data.get("count", 0), e.data.get("previous", 0)}
            for c, n in (e.data.get("by_class") or {}).items():
                ix.count_pairs.add((n, c))
    ix.classes = set(ix.cls_of.values()) | set(summary.get("by_class", {}))
    for c, n in summary.get("by_class", {}).items():
        ix.count_pairs.add((n, c))
    ix.vehicle_totals = {
        sum(
            n
            for c, n in summary.get("by_class", {}).items()
            if c in ("car", "bus", "truck", "motorcycle")
        )
    }
    ix.numbers |= {summary.get("tracks", 0), summary.get("peak", 0), round(duration)}
    ix.numbers |= {int(x) for x in re.findall(r"\d+", _TIME.sub(" ", facts_text))}
    for m in _COUNT_NOUN.finditer(facts_text.lower()):
        w = m.group(1)
        ix.count_pairs.add((int(w) if w.isdigit() else NUMBER_WORDS[w], CLASS_WORDS[m.group(2)]))
    return ix


def _ids(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def check_sentence(s: str, ix: _Index) -> str | None:
    """Return a reason when the sentence is unsupported, else None."""
    low = s.lower().replace("’", "'")
    mentioned: set[int] = set()

    # --- objects and group counts
    for m in _SINGLE.finditer(low):
        cls, i = m.group(1), int(m.group(2))
        if ix.cls_of.get(i) != cls:
            return f"'{m.group(0)}' is not a tracked object"
        mentioned.add(i)
    for m in _GROUP.finditer(low):
        word, cls_w, ids1, cls_w2, ids2 = m.groups()
        cls = CLASS_WORDS[cls_w or cls_w2]
        ids = _ids(ids1 or ids2)
        for i in ids:
            if ix.cls_of.get(i) != cls:
                return f"{cls} {i} is not a tracked object"
        mentioned.update(ids)
        if word and ids1:
            n = int(word) if word.isdigit() else NUMBER_WORDS.get(word)
            if n is not None and n != len(ids):
                return f"count {word} does not match the {len(ids)} ids listed"

    for m in _GENERIC_GROUP.finditer(_GROUP.sub(" ", low)):
        for i in _ids(m.group(1)):
            if i not in ix.cls_of:
                return f"object {i} is not a tracked object"
            mentioned.add(i)

    # --- counts of objects
    for m in _COUNT_NOUN.finditer(_TIME.sub(" ", low)):
        w, noun = m.group(1), m.group(2)
        n = int(w) if w.isdigit() else NUMBER_WORDS[w]
        if noun in ("vehicle", "vehicles"):
            if n not in ix.vehicle_totals and n not in ix.numbers:
                return f"count '{w} {noun}' is not in the facts"
            continue
        if (n, CLASS_WORDS[noun]) not in ix.count_pairs:
            return f"count '{w} {noun}' is not in the facts"
    for noun in re.findall(r"[a-z]+", low):
        if noun in CLASS_WORDS and CLASS_WORDS[noun] not in ix.classes:
            return f"object '{noun}' is not in the facts"
    body = _TIME.sub(" ", _SINGLE.sub(" ", _GROUP.sub(" ", low)))
    for w in re.findall(r"[a-z]+", body):
        if w in NUMBER_WORDS and w != "one" and NUMBER_WORDS[w] not in ix.numbers:
            nxt = re.search(rf"\b{w}\s+({_CLS})\b", body)
            if not nxt:
                return f"number '{w}' is not in the facts"
    for n in (int(x) for x in re.findall(r"\b\d+\b", body)):
        if n not in ix.numbers:
            return f"number {n} is not in the facts"

    pool = mentioned or set(ix.cls_of)

    # --- times and durations
    for m in _TIME.finditer(low):
        t = int(m.group(1)) * 60 + int(m.group(2))
        near = [x for i in pool for x in ix.times.get(i, [])] or ix.all_times
        if near and min(abs(t - x) for x in near) > 1.01:
            return f"time {m.group(0)} matches no event of the objects mentioned"
    for m in _FOR_SECONDS.finditer(low):
        w = m.group(1)
        n = int(w) if w.isdigit() else NUMBER_WORDS.get(w)
        if n is not None and n not in ix.durations and n not in ix.numbers:
            return f"duration of {w} seconds is not in the events"

    # --- directions
    def allowed(d: dict[int, set[str]]) -> set[str]:
        return {x for i in pool for x in d.get(i, set())}

    for m in FROM_DIR.finditer(low):
        if m.group(1) not in allowed(ix.enter_edges):
            return f"'from the {m.group(1)}' disagrees with the entry edge"
    for m in MOVE_DIR.finditer(low):
        comps = set(m.group(1).split("-"))
        if not comps <= allowed(ix.move_dirs) | allowed(ix.leave_edges):
            return f"direction '{m.group(1)}' disagrees with the movement"
    stripped = MOVE_DIR.sub(" ", FROM_DIR.sub(" ", low))
    for m in TO_DIR.finditer(stripped):
        d = m.group(1)
        if d not in allowed(ix.leave_edges) | allowed(ix.move_dirs):
            return f"'the {d}' disagrees with the exit edge or direction of travel"

    # --- verbs
    verb_text = TO_DIR.sub(" ", stripped)
    verb_text = _TIME.sub(" ", verb_text)
    has = lambda kind: any(kind in ix.kinds.get(i, set()) for i in pool)  # noqa: E731
    if ENTER.search(verb_text) and not has("enter"):
        return "claims an entry that is not in the events"
    if LEAVE.search(verb_text) and not has("leave"):
        return "claims an exit that is not in the events"
    if STAY.search(verb_text) and not has("linger"):
        return "claims an object stayed or stopped, which the events do not say"
    clean = ENTER.sub(" ", LEAVE.sub(" ", verb_text))
    if MOVE.search(clean) and not has("move") and not has("enter"):
        return "claims movement that is not in the events"
    if CLOSE.search(verb_text):
        ms = sorted(mentioned)
        if len(ms) >= 2:
            ok = any(frozenset((a, b)) in ix.prox for k, a in enumerate(ms) for b in ms[k + 1 :])
        elif len(ms) == 1:
            ok = any(ms[0] in pair for pair in ix.prox)
        else:
            ok = bool(ix.prox)
        if not ok:
            return "claims objects were close together without such an event"
    return None


def check_faithful(
    text: str, events: list[Event], summary: dict, facts_text: str, duration: float
) -> tuple[bool, str | None]:
    if len(text.strip()) < 20:
        return False, "text too short"
    ix = _build_index(events, summary, facts_text, duration)
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not s:
            continue
        why = check_sentence(s, ix)
        if why:
            return False, why
    return True, None
