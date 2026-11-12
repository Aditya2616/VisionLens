"""Typed records shared by the core, the API, the CLI and the front end."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the analysed frame

EventKind = Literal["enter", "leave", "move", "linger", "count_change", "proximity"]


class Detection(BaseModel):
    box: Box
    score: float
    cls: str


class TrackOut(BaseModel):
    id: int
    box: Box
    cls: str
    score: float
    detected: bool = Field(description="True when this frame's box came from a detector match")


class FrameResult(BaseModel):
    i: int
    t: float
    dets: list[Detection] | None = Field(
        default=None, description="Raw detector output; None on frames the detector skipped"
    )
    tracks: list[TrackOut] = Field(default_factory=list)


class Event(BaseModel):
    id: str
    t: float
    kind: EventKind
    ids: list[int] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    pos: tuple[float, float] | None = None
    t_end: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class Sentence(BaseModel):
    text: str
    event_ids: list[str] = Field(default_factory=list)


class DescriptionVariant(BaseModel):
    source: Literal["template", "llm", "openai", "anthropic"]
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    model: str | None = None
    accepted: bool = True
    rejected_reason: str | None = None
    seconds: float | None = None


class Description(BaseModel):
    template: DescriptionVariant
    model: DescriptionVariant | None = None
    used: Literal["template", "model"] = "template"


class Timings(BaseModel):
    detect_ms_mean: float = 0.0
    detect_ms_p95: float = 0.0
    track_ms_mean: float = 0.0
    detect_calls: int = 0
    wall_s: float = 0.0
    fps_end_to_end: float = 0.0
    threads: int = 0
    cpu: str = ""
    machine_note: str = ""


class Meta(BaseModel):
    name: str
    kind: Literal["video", "image"] = "video"
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    stride: int
    detector: str
    input_size: int
    tracker: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False
    recorded: bool = False
    note: str = ""


class Result(BaseModel):
    meta: Meta
    frames: list[FrameResult]
    events: list[Event]
    description: Description | None = None
    timings: Timings = Field(default_factory=Timings)
    summary: dict[str, Any] = Field(default_factory=dict)
    ground_truth: list[list[dict[str, Any]]] | None = Field(
        default=None, description="Per-frame exact boxes; only present for synthetic scenes"
    )
