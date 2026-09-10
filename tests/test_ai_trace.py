"""AI trace factory tests (mock backend, no network)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine import Executor, Graph


def _loop_graph(rounds=2, delay=0.001):
    g = Graph()
    p = g.add_node("ai.prompt", params={"text": "Cache edge cases?"})
    loop = g.add_node("ai.backforth_loop", params={"rounds": rounds, "system": "sys", "model": "mock", "delay": delay})
    g.add_link(p.id, "text", loop.id, "prompt")
    return g, loop


def test_backforth_loop_history():
    g, loop = _loop_graph(rounds=2)
    assert g.validate() == []
    ex = Executor(g)
    rep = ex.run_blocking()
    assert rep.ok(), {k: v.error for k, v in rep.per_node.items()}
    trace = ex.outputs[loop.id]["trace"]
    assert trace["backend"] == "mock" and len(trace["rounds"]) == 2
    assert all({"draft", "critique", "revision"} <= set(r) for r in trace["rounds"])
    assert trace["final"] == trace["rounds"][-1]["revision"]
    json.dumps(trace)  # JSONL-serialisable


def test_single_generate_trace():
    g = Graph()
    p = g.add_node("ai.prompt", params={"text": "Hello"})
    gen = g.add_node("ai.llm_generate", params={"system": "sys", "model": "mock", "delay": 0.001})
    g.add_link(p.id, "text", gen.id, "prompt")
    ex = Executor(g)
    assert ex.run_blocking().ok()
    assert ex.outputs[gen.id]["trace"]["final"]


def test_dataset_appends_jsonl(tmp_path):
    g = Graph()
    p = g.add_node("ai.prompt", params={"text": "Hello"})
    gen = g.add_node("ai.llm_generate", params={"system": "sys", "model": "mock", "delay": 0.001})
    ds = g.add_node("ai.trace_dataset", params={"path": str(tmp_path / "t.jsonl")})
    g.add_link(p.id, "text", gen.id, "prompt")
    g.add_link(gen.id, "trace", ds.id, "trace")
    ex = Executor(g)
    assert ex.run_blocking().ok()
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["prompt"] == "Hello"
