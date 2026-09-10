"""Loop execution tests: end conditions, caps, validation, serialization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine import Executor, Graph


def _count_loop(target=5.0, max_iterations=1000):
    """1 + accumulate until >= target. Returns (graph, ids)."""
    g = Graph()
    one = g.add_node("input.number", params={"value": 1.0})
    add = g.add_node("maths.arithmetic", params={"operation": "add", "absolute": False})
    acc = g.add_node("control.accumulate", params={"initial": 0.0})
    cond = g.add_node("control.end_condition", params={"mode": "threshold", "target": target, "tolerance": 0.0})
    g.add_link(one.id, "out", add.id, "a")
    g.add_link(acc.id, "current", add.id, "b")
    g.add_link(add.id, "out", acc.id, "next")
    g.add_link(acc.id, "current", cond.id, "value")
    g.add_loop("count", [one.id, add.id, acc.id, cond.id], cond.id, max_iterations=max_iterations)
    return g, {"acc": acc.id, "cond": cond.id}


def test_threshold_loop_terminates_exactly():
    g, ids = _count_loop(target=5.0)
    assert g.validate() == []
    ex = Executor(g)
    report = ex.run_blocking()
    assert report.ok(), {k: v.error for k, v in report.per_node.items()}
    assert ex.outputs[ids["acc"]]["current"] == 5.0


def test_counter_mode_runs_exact_iterations():
    g = Graph()
    counter = g.add_node("control.counter", params={})
    acc = g.add_node("control.accumulate", params={"initial": 0.0})
    cond = g.add_node("control.end_condition", params={"mode": "counter", "iterations": 3})
    g.add_link(counter.id, "count", acc.id, "next")
    g.add_link(acc.id, "current", cond.id, "value")
    g.add_loop("three", [counter.id, acc.id, cond.id], cond.id)
    ex = Executor(g)
    assert ex.run_blocking().ok()
    # passes latch 0,1,2 -> current reads 0,0,1
    assert ex.outputs[acc.id]["current"] == 1.0


def test_max_iterations_is_an_error():
    g, _ids = _count_loop(target=1e9, max_iterations=5)
    ex = Executor(g)
    report = ex.run_blocking()
    assert not report.ok()
    assert any("max_iterations=5" in (r.error or "") for r in report.per_node.values())


def test_default_max_is_1000():
    g, _ids = _count_loop(target=1e18)
    assert g.loops[0].max_iterations == 1000
    ex = Executor(g)
    report = ex.run_blocking()
    assert not report.ok()
    assert any("max_iterations=1000" in (r.error or "") for r in report.per_node.values())


def test_control_nodes_standalone_defaults():
    g = Graph()
    counter = g.add_node("control.counter", params={})
    acc = g.add_node("control.accumulate", params={"initial": 7.0})
    one = g.add_node("input.number", params={"value": 2.0})
    cond = g.add_node("control.end_condition", params={"mode": "truthy"})
    g.add_link(one.id, "out", cond.id, "value")
    ex = Executor(g)
    assert ex.run_blocking().ok()
    assert ex.outputs[counter.id]["count"] == 0.0
    assert ex.outputs[acc.id]["current"] == 7.0
    assert ex.outputs[cond.id]["done"] == 1.0


def test_loop_validation():
    g = Graph()
    g.add_node("input.number", params={"value": 1.0})
    cond = g.add_node("control.end_condition", params={"mode": "counter", "iterations": 2})
    g.add_loop("empty", [], cond.id)
    assert any("empty body" in e.message for e in g.validate())

    g2 = Graph()
    n = g2.add_node("input.number", params={"value": 1.0})
    c2 = g2.add_node("control.end_condition", params={"mode": "counter", "iterations": 2})
    g2.add_loop("outside", [n.id], c2.id)
    assert any("must be inside the body" in e.message for e in g2.validate())

    g3 = Graph()
    m = g3.add_node("input.number", params={"value": 1.0})
    g3.add_loop("wrongtype", [m.id], m.id)
    assert any("must be a control.end_condition" in e.message for e in g3.validate())

    g4 = Graph()
    p = g4.add_node("input.number", params={"value": 1.0})
    d = g4.add_node("control.end_condition", params={"mode": "counter", "iterations": 1})
    g4.add_loop("l1", [p.id, d.id], d.id)
    g4.add_loop("l2", [p.id], d.id)
    assert any("two loops" in e.message for e in g4.validate())

    g5 = Graph()
    q = g5.add_node("input.number", params={"value": 1.0})
    e5 = g5.add_node("control.end_condition", params={"mode": "counter", "iterations": 1})
    g5.add_loop("badcap", [q.id, e5.id], e5.id, max_iterations=0)
    assert any("max_iterations >= 1" in e.message for e in g5.validate())


def test_loop_serialization_roundtrip():
    g, _ids = _count_loop(target=3.0)
    g2 = Graph.from_json(g.to_json())
    assert len(g2.loops) == 1 and g2.loops[0].max_iterations == 1000
    ex = Executor(g2)
    assert ex.run_blocking().ok()
    acc_id = next(nid for nid, inst in g2.nodes.items() if inst.type_id == "control.accumulate")
    assert ex.outputs[acc_id]["current"] == 3.0
