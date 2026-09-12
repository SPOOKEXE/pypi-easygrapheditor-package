"""Async AI trace data factory demo: prompt -> back-forth LLM loop -> JSONL.

Back-and-forth loop = draft -> critique -> rewrite per round (see nodes_ai).
Default backend is offline mock (no keys, deterministic). To use OpenAI:
  EGE_LLM_BACKEND=openai OPENAI_API_KEY=... uv run easygrapheditor/examples/ai_trace_factory_demo.py

Run: uv run easygrapheditor/examples/ai_trace_factory_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_runner import run_demo
from easygrapheditor.engine import Cache, Executor, Graph

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def build() -> tuple[Graph, dict[str, str]]:
    """Build the demo graph. Returns (graph, refs) with node ids (also used by view_demo.py)."""
    g = Graph()
    n_prompt = g.add_node(
        "ai.prompt",
        params={"text": "Design 3 edge-case tests for a node-graph cache (stale inputs, cycles, hash collisions)."},
        pos=(0, 0),
    )
    n_single = g.add_node(
        "ai.llm_generate",
        params={"system": "You are a concise data generator.", "model": "mock", "delay": 0.02},
        pos=(280, -80),
    )
    n_loop = g.add_node(
        "ai.backforth_loop",
        params={"rounds": 2, "system": "You are a generator + critic pair.", "model": "mock", "delay": 0.02},
        pos=(280, 120),
    )
    traces_path = str(OUT / "traces.jsonl")
    Path(traces_path).unlink(missing_ok=True)
    n_data = g.add_node("ai.trace_dataset", params={"path": traces_path}, pos=(560, 120))

    g.add_link(n_prompt.id, "text", n_single.id, "prompt")
    g.add_link(n_prompt.id, "text", n_loop.id, "prompt")
    g.add_link(n_loop.id, "trace", n_data.id, "trace")
    return g, {"prompt": n_prompt.id, "single": n_single.id, "loop": n_loop.id, "dataset": n_data.id}


def main() -> None:
    g, refs = build()

    errs = g.validate()
    print("validate:", errs)
    assert not errs, errs

    cache = Cache()
    ex = Executor(g, cache=cache)
    report = ex.run_blocking()
    print(f"run {report.ms:.1f}ms ok={report.ok()} cache={cache.summary()}")
    for nid, rep in report.per_node.items():
        print(f"  {nid}: {rep.status} {rep.ms:.1f}ms stages={len(rep.stages)} {rep.error or ''}")

    single = ex.outputs[refs["single"]]["trace"]
    looped = ex.outputs[refs["loop"]]["trace"]
    saved = ex.outputs[refs["dataset"]]["path"]
    print(f"\nsingle backend={single['backend']} ms={single['ms']:.1f}")
    print(f"  draft: {single['draft'][:160]}...")
    print(f"\nloop backend={looped['backend']} rounds={len(looped['rounds'])} ms={looped['ms']:.1f}")
    for r in looped["rounds"]:
        print(f"  round {r['round']} draft: {str(r['draft'])[:110]}...")
        print(f"           critique: {str(r['critique'])[:110]}...")
    print(f"  final: {str(looped['final'])[:200]}...")
    print(f"\ndataset: {saved} ({Path(saved).stat().st_size} bytes, {len(Path(saved).read_text().splitlines())} lines)")

    p_graph = OUT / "trace_graph.ege.json"
    p_graph.write_text(g.to_json())
    print(f"graph: {p_graph}")


if __name__ == "__main__":
    run_demo(build=build, cli_main=main, title="AI Trace Factory", description=__doc__ or "AI trace demo", script_path=__file__)
