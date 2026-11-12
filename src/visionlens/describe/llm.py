"""Local language-model rewrite of the template facts (small causal LM through transformers)."""

from __future__ import annotations

import os
import threading
import time

from ..schemas import DescriptionVariant
from ..sysutil import resolve_device
from .base import Facts, Feedback, TokenCallback

# The model with the best guard acceptance in results/results.json (checked by a test).
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
SMALL_LLM = "Qwen/Qwen2.5-0.5B-Instruct"  # the faster comparison model

SYSTEM = (
    "You describe what happens in a short video using only a list of facts. Write one short, "
    "natural paragraph of two to four sentences, the way a person would summarise the scene: say "
    "how many objects there were, group things that happened together, and skip minor details. "
    "Vary how the sentences begin. Keep every object name with its number exactly as given (for "
    "example 'car 3' or 'three cars (1, 2, 3)'), and keep directions (left, right, top, bottom), "
    "times (like 0:04) and durations exactly as given. Do not add objects, numbers, places, "
    "causes or opinions that are not in the facts."
)


def build_prompt(facts: Facts, feedback: Feedback | None = None) -> list[dict]:
    """The facts go in as a compact structured list; after a rejected attempt the conversation
    continues with the previous answer and the guard's reason."""
    listing = "\n".join(facts.lines or [f"- {s.text}" for s in facts.sentences])
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Facts (time: what happened):\n{listing}\n\nWrite the paragraph.",
        },
    ]
    if feedback is not None:
        where = f' The sentence was: "{feedback.sentence}".' if feedback.sentence else ""
        msgs += [
            {"role": "assistant", "content": feedback.text},
            {
                "role": "user",
                "content": (
                    f"That paragraph was rejected by a fact check: {feedback.reason}.{where} "
                    "Write the whole paragraph again, fixing that and using only the facts above."
                ),
            },
        ]
    return msgs


def trim_to_sentence(text: str) -> str:
    """Drop a dangling half sentence left by the token limit."""
    if text and text[-1] not in ".!?":
        cut = text.rfind(".")
        if cut > 40:
            return text[: cut + 1]
    return text


class LocalLLMDescriber:
    name = "llm"
    supports_feedback = True  # a rejected attempt can be regenerated with the guard's reason

    def __init__(
        self, model_id: str | None = None, max_new_tokens: int = 160, device: str | None = None
    ) -> None:
        self.model_id = model_id or os.environ.get("VISIONLENS_LLM", DEFAULT_LLM)
        self.max_new_tokens = max_new_tokens
        self.device = resolve_device(device)
        self.dtype = "float16" if self.device == "cuda" else "float32"
        self._tok = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
                .eval()
                .to(self.device)
            )
        except RuntimeError:  # CUDA out of memory: fall back to the CPU
            self.device, self.dtype = "cpu", "float32"
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch.float32
            ).eval()

    def describe(
        self,
        facts: Facts,
        on_token: TokenCallback | None = None,
        feedback: Feedback | None = None,
    ) -> DescriptionVariant:
        import torch
        from transformers import TextIteratorStreamer

        self._load()
        assert self._tok is not None and self._model is not None
        t0 = time.perf_counter()
        inputs = self._tok.apply_chat_template(
            build_prompt(facts, feedback),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        streamer = TextIteratorStreamer(self._tok, skip_prompt=True, skip_special_tokens=True)

        def run() -> None:
            with torch.inference_mode():
                self._model.generate(  # type: ignore[union-attr]
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    streamer=streamer,
                )

        th = threading.Thread(target=run, daemon=True)
        th.start()
        chunks: list[str] = []
        for piece in streamer:
            chunks.append(piece)
            if on_token:
                on_token(piece)
        th.join()
        return DescriptionVariant(
            source="llm",
            text=trim_to_sentence("".join(chunks).strip()),
            model=self.model_id,
            seconds=round(time.perf_counter() - t0, 1),
        )
