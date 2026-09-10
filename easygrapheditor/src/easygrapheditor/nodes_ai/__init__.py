"""AI trace data factory: prompt -> LLM generate -> back-forth refine loop -> JSONL dataset.

Chain: ai.prompt --TEXT--> ai.llm_generate --TRACE--\
                                              ai.backforth_loop --TRACE--> ai.trace_dataset --TEXT(path)
The loop node does draft -> critique -> rewrite per round (back-and-forth),
reporting executor stages so UI shows planning/fetching/thinking/writing/checking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..engine.nodes import ExecCtx, Param, PortDef, node
from ..engine.types import DataType, register_type
from .llm import get_backend

register_type(DataType("data.TEXT", "Text", "Plain text prompt/response.", "none"))
register_type(DataType("data.TRACE", "Trace", "LLM trace: prompt + rounds of draft/critique/revision.", "none"))

TEXT = "data.TEXT"
TRACE = "data.TRACE"


@node(
    type_id="ai.prompt",
    title="Prompt",
    category="AI",
    color="teal",
    description="Constant text prompt source.",
    inputs=[],
    outputs=[PortDef("text", "Text", TEXT, "out")],
)
def prompt_node(text: str = Param("Explain domain warping for terrain generation.", kind="multiline", label="Text")) -> str:  # type: ignore[no-untyped-def]
    return str(text)


@node(
    type_id="ai.llm_generate",
    title="LLM Generate",
    category="AI",
    color="teal",
    description="Single LLM call producing a trace (mock offline by default).",
    inputs=[PortDef("prompt", "Prompt", TEXT, "in")],
    outputs=[PortDef("trace", "Trace", TRACE, "out")],
)
async def llm_generate(
    ctx: ExecCtx,
    prompt: str,
    system: str = Param("You are a concise data generator.", kind="multiline", label="System"),  # type: ignore[no-untyped-def]
    model: str = Param("mock", kind="dropdown", label="Backend", options=["mock", "auto", "openai"]),  # type: ignore[no-untyped-def]
    delay: float = Param(0.05, kind="number", label="Mock delay (s)"),  # type: ignore[no-untyped-def]
) -> dict[str, Any]:
    backend = get_backend(str(model), delay=float(delay))
    ctx.report("planning", 0.2)
    t0 = time.perf_counter()
    ctx.report("fetching", 0.4)
    ctx.report("thinking", 0.6)
    draft = await backend.generate(str(prompt), system=str(system))
    ctx.report("writing", 0.8)
    ctx.report("checking", 1.0)
    return {
        "kind": "single",
        "backend": backend.name,
        "system": str(system),
        "prompt": str(prompt),
        "draft": draft,
        "final": draft,
        "rounds": [{"round": 1, "draft": draft, "critique": "", "revision": draft}],
        "ms": (time.perf_counter() - t0) * 1000.0,
    }


@node(
    type_id="ai.backforth_loop",
    title="Back-Forth Loop",
    category="AI",
    color="teal",
    description="Draft -> critique -> rewrite, N rounds. Returns full trace history.",
    inputs=[PortDef("prompt", "Prompt", TEXT, "in")],
    outputs=[PortDef("trace", "Trace", TRACE, "out")],
)
async def backforth_loop(
    ctx: ExecCtx,
    prompt: str,
    rounds: int = Param(2, kind="int", label="Rounds", min=1, max=5, step=1),  # type: ignore[no-untyped-def]
    system: str = Param("You are a generator + critic pair.", kind="multiline", label="System"),  # type: ignore[no-untyped-def]
    model: str = Param("mock", kind="dropdown", label="Backend", options=["mock", "auto", "openai"]),  # type: ignore[no-untyped-def]
    delay: float = Param(0.05, kind="number", label="Mock delay (s)"),  # type: ignore[no-untyped-def]
) -> dict[str, Any]:
    backend = get_backend(str(model), delay=float(delay))
    n = max(1, min(int(rounds), 5))
    history: list[dict[str, str]] = []
    current = str(prompt)
    t0 = time.perf_counter()
    total_steps = n * 3
    step = 0
    for r in range(1, n + 1):
        ctx.report(f"planning r{r}", step / total_steps)
        draft = await backend.generate(f"DRAFT (round {r}/{n}): {current}", system=str(system))
        step += 1
        ctx.report(f"thinking r{r}", step / total_steps)
        critique = await backend.generate(f"CRITIQUE this draft:\n{draft}\nBe specific, 3 bullets.", system=str(system))
        step += 1
        ctx.report(f"writing r{r}", step / total_steps)
        current = await backend.generate(
            f"REWRITE given critique:\nCritique: {critique}\nDraft: {draft}\nOriginal: {prompt}",
            system=str(system),
        )
        step += 1
        history.append({"round": r, "draft": draft, "critique": critique, "revision": current})  # type: ignore[dict-item]
    ctx.report("checking", 1.0)
    return {
        "kind": "backforth",
        "backend": backend.name,
        "system": str(system),
        "prompt": str(prompt),
        "rounds": history,
        "final": current,
        "ms": (time.perf_counter() - t0) * 1000.0,
    }


@node(
    type_id="ai.trace_dataset",
    title="Trace Dataset",
    category="AI",
    color="teal",
    description="Append a trace as one JSONL line. Returns the file path.",
    inputs=[PortDef("trace", "Trace", TRACE, "in")],
    outputs=[PortDef("path", "Path", TEXT, "out")],
    cacheable=False,
)
def trace_dataset(trace: dict, path: str = Param("traces.jsonl", kind="text", label="Path")) -> str:  # type: ignore[no-untyped-def]
    p = Path(str(path))
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return str(p)
