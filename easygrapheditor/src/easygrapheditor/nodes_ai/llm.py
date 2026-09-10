"""Pluggable LLM backend: Mock by default, OpenAI if configured.

No hard dependency: real backends are imported lazily and only when
env opts in (EGE_LLM_BACKEND=openai + OPENAI_API_KEY).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Protocol


class LLMBackend(Protocol):
    name: str

    async def generate(self, prompt: str, system: str = "") -> str:
        ...


class MockBackend:
    """Deterministic fake LLM: offline, fast, stable hashes for tests/demos."""

    name = "mock"

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay

    async def generate(self, prompt: str, system: str = "") -> str:
        await asyncio.sleep(self.delay)
        h = hashlib.sha1(f"{system}|{prompt}".encode()).hexdigest()[:8]
        # Structured fake response so downstream parsing/loop logic is realistic.
        head = prompt.strip().replace("\n", " ")[:160]
        return (
            f"[mock:{h}] Re: {head}\n"
            f"- claim: mock answer derived from prompt hash {h}\n"
            f"- reason: simulated thinking trace (no network)\n"
            f"- followup: ask for clarification about '{head[-40:]}'"
        )


class OpenAIBackend:
    """Thin wrapper over `openai` Async client (optional dep)."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("pip install openai to use EGE_LLM_BACKEND=openai") from e
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self._model = model

    async def generate(self, prompt: str, system: str = "") -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(model=self._model, messages=msgs)
        return (resp.choices[0].message.content or "").strip()


def get_backend(name: str = "auto", model: str = "gpt-4o-mini", delay: float = 0.05) -> LLMBackend:
    """Resolve backend by name. 'auto' prefers OpenAI only when explicitly opted in."""
    want = (name or "auto").lower()
    if want in ("mock",):
        return MockBackend(delay=delay)
    if want in ("openai",) or (want == "auto" and os.environ.get("EGE_LLM_BACKEND", "").lower() == "openai"):
        try:
            return OpenAIBackend(model=model)
        except Exception:
            if want == "openai":
                raise
            return MockBackend(delay=delay)
    return MockBackend(delay=delay)
