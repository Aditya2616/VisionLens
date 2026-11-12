"""Optional hosted describers (OpenAI, Anthropic) behind the same interface.

These are adapters. They are exercised in the tests against a stubbed HTTP transport only;
no hosted call was made while building this project.
"""

from __future__ import annotations

import os
import time

import httpx

from ..schemas import DescriptionVariant
from .base import Facts, TokenCallback
from .llm import SYSTEM


def _user_prompt(facts: Facts) -> str:
    bullets = "\n".join(f"- {s.text}" for s in facts.sentences)
    return f"Facts:\n{bullets}\n\nWrite the description."


class OpenAIDescriber:
    name = "openai"

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.transport = transport

    def describe(self, facts: Facts, on_token: TokenCallback | None = None) -> DescriptionVariant:
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        t0 = time.perf_counter()
        with httpx.Client(transport=self.transport, timeout=60) as client:
            r = client.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": _user_prompt(facts)},
                    ],
                },
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
        if on_token:
            on_token(text)
        return DescriptionVariant(
            source="openai", text=text, model=self.model, seconds=round(time.perf_counter() - t0, 1)
        )


class AnthropicDescriber:
    name = "anthropic"

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = os.environ.get("ANTHROPIC_MODEL", "test-model")
        self.base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        self.transport = transport

    def describe(self, facts: Facts, on_token: TokenCallback | None = None) -> DescriptionVariant:
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        t0 = time.perf_counter()
        with httpx.Client(transport=self.transport, timeout=60) as client:
            r = client.post(
                f"{self.base}/v1/messages",
                headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                json={
                    "model": self.model,
                    "max_tokens": 300,
                    "temperature": 0,
                    "system": SYSTEM,
                    "messages": [{"role": "user", "content": _user_prompt(facts)}],
                },
            )
            r.raise_for_status()
            text = "".join(b.get("text", "") for b in r.json()["content"]).strip()
        if on_token:
            on_token(text)
        return DescriptionVariant(
            source="anthropic",
            text=text,
            model=self.model,
            seconds=round(time.perf_counter() - t0, 1),
        )
