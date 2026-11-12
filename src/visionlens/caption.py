"""Optional scene captioner: a small vision-language model describes the *setting* on a few key
frames ("a street corner with tram tracks and parked cars"), which the detections and events
cannot know.

The caption is a separate, labelled line in the result (`Result.scene`). It is never merged into
the event sentences. The objects it names are cross-checked against the classes the detector
actually saw, and a mismatch is flagged in the result.
"""

from __future__ import annotations

import re
import time

import numpy as np

from .schemas import SceneCaption, SceneFrameCaption
from .sysutil import resolve_device

CAPTION_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
# Download only what inference needs: the repository also holds ONNX exports (several GB).
CAPTION_IGNORE = ["onnx/*", "*.md", ".gitattributes"]
PROMPTS = {
    "A": (
        "Describe the place in this image in one short sentence: the kind of setting and the main "
        "things you can see. Do not describe individual people."
    ),
    "B": (
        "What kind of place is this? Answer with one short sentence that names the place and the "
        "main things in it."
    ),
    "C": (
        "Describe the setting of this image in one complete sentence of at least eight words: "
        "what kind of place it is and what is in it."
    ),
}
PROMPT_KEY = "C"  # chosen by the evaluation in results/results.json (see the README)
PROMPT = PROMPTS[PROMPT_KEY]
KEYFRAME_POSITIONS = (0.1, 0.5, 0.9)  # fractions of the clip; the middle one is shown


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    m = re.search(r"(?<=[.!?])\s", text)
    out = text[: m.start()] if m else text
    return out.strip()


def named_classes(text: str) -> list[str]:
    """Detector classes (person, car, bus, ...) that a caption names, in order of first mention."""
    from .describe.guard import CLASS_WORDS

    seen: list[str] = []
    for w in re.findall(r"[a-z]+", text.lower()):
        c = CLASS_WORDS.get(w)
        if c and c not in seen:
            seen.append(c)
    return seen


def pick_keyframes(
    thumbs: list[tuple[int, float, np.ndarray]], positions=KEYFRAME_POSITIONS
) -> list[tuple[float, np.ndarray]]:
    """The stored frames closest to the given fractions of the clip (no duplicates)."""
    if not thumbs:
        return []
    t0, t1 = thumbs[0][1], thumbs[-1][1]
    out: list[tuple[float, np.ndarray]] = []
    used: set[int] = set()
    for p in positions:
        target = t0 + p * (t1 - t0)
        k = min(range(len(thumbs)), key=lambda j: abs(thumbs[j][1] - target))
        if k not in used:
            used.add(k)
            out.append((thumbs[k][1], thumbs[k][2]))
    return out


class SceneCaptioner:
    name = "smolvlm"

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 64,
        prompt: str = PROMPT,
    ) -> None:
        self.model_id = model_id or CAPTION_MODEL
        self.device = resolve_device(device)
        self.dtype = "float16" if self.device == "cuda" else "float32"
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self._proc = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._proc = AutoProcessor.from_pretrained(self.model_id, do_image_splitting=False)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self._model = (
                AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype)
                .to(self.device)
                .eval()
            )
        except RuntimeError:  # CUDA out of memory: fall back to the CPU
            self.device, self.dtype = "cpu", "float32"
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=torch.float32
            ).eval()

    def caption_frame(self, frame_bgr: np.ndarray) -> str:
        return first_sentence(self.generate(frame_bgr))

    def generate(self, frame_bgr: np.ndarray) -> str:
        """The model's raw answer for one frame (before it is cut to one sentence)."""
        import cv2
        import torch
        from PIL import Image

        self._load()
        assert self._proc is not None and self._model is not None
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        msgs = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}
        ]
        text = self._proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self._proc(text=text, images=[img], return_tensors="pt").to(self.device)
        if self.dtype == "float16":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                stop_strings=["."],  # one sentence is wanted: stop at the first full stop
                tokenizer=self._proc.tokenizer,
            )
        gen = self._proc.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return gen[0].strip()

    def caption(
        self, frames: list[tuple[float, np.ndarray]], seen_classes: set[str]
    ) -> SceneCaption | None:
        """Caption each key frame; the middle one is the clip's scene line. `seen_classes` are the
        classes the detector found, used to flag objects the caption names that it never saw."""
        if not frames:
            return None
        t0 = time.perf_counter()
        per = [SceneFrameCaption(t=round(t, 2), text=self.caption_frame(f)) for t, f in frames]
        seconds = round(time.perf_counter() - t0, 2)
        shown = per[len(per) // 2]
        named = named_classes(shown.text)
        unseen = [c for c in named if c not in seen_classes]
        return SceneCaption(
            text=shown.text,
            model=self.model_id,
            device=self.device,
            dtype=self.dtype,
            seconds=seconds,
            frames=per,
            named=named,
            unseen=unseen,
            flagged=bool(unseen),
        )
