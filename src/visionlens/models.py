"""Which models `visionlens fetch-models` downloads, in one place (the README's download sizes
are computed from this plan)."""

from __future__ import annotations

from .caption import CAPTION_IGNORE, CAPTION_MODEL
from .describe.llm import DEFAULT_LLM, SMALL_LLM
from .detect.hf import MODELS


def fetch_plan(
    all_models: bool = False, llm: bool = False, caption: bool = False
) -> list[tuple[str, list[str]]]:
    """(repository, ignore patterns) pairs. The default plan is the default detector alone."""
    plan: list[tuple[str, list[str]]] = [(MODELS["dfine-nano"], [])]
    if all_models:
        plan += [(MODELS["dfine-small"], []), (MODELS["rtdetr-r18"], [])]
    if llm or all_models:
        plan.append((DEFAULT_LLM, []))
    if all_models:
        plan.append((SMALL_LLM, []))
    if caption or all_models:
        plan.append((CAPTION_MODEL, CAPTION_IGNORE))
    return plan
