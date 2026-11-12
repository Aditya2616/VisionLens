from __future__ import annotations

from .base import Describer, Facts, build_description, check_faithful
from .template import TemplateDescriber, describe_image

# By the evidence in results/results.json (see the README and tests/test_readme_facts.py): the
# template stays the default; the model rewrite is an option.
DEFAULT_DESCRIBER = "template"


def make_describer(kind: str) -> Describer | None:
    """'template' means no model rewrite (returns None); others return the model describer."""
    if kind in ("template", "", "none"):
        return None
    if kind == "llm":
        from .llm import LocalLLMDescriber

        return LocalLLMDescriber()
    if kind == "openai":
        from .hosted import OpenAIDescriber

        return OpenAIDescriber()
    if kind == "anthropic":
        from .hosted import AnthropicDescriber

        return AnthropicDescriber()
    raise ValueError(f"unknown describer: {kind}")


__all__ = [
    "Describer",
    "Facts",
    "TemplateDescriber",
    "build_description",
    "check_faithful",
    "describe_image",
    "make_describer",
]
