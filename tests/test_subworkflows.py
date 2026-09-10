"""Subworkflow tests: combine/run/expand, nesting, recursion caps, validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine import Executor, Graph, combine_nodes, expand_subworkflow_node


def _adder_graph(a=2.0, b=3.0):
    g = Graph()
    n1 = g.add_node("input.number", params={"value": a})
    n2 = g.add_node("input.number", params={"value": b})
    ad = g.add_node("maths.arithmetic", params={"operation": "add", "absolute": False})
    ro = g.add_node("output.readout", params={})
    g.add_link(n1.id, "out", ad.id, "a")
    g.add_link(n2.id, "out", ad.id, "b")
    g.add_link(ad.id, "out", ro.id, "value_in")
    return g, {"n1": n1.id, "n2": n2.id, "ad": ad.id, "ro": ro.id}


def test_combine_runs_like_flat():
    g, ids = _adder_graph()
    sub = combine_nodes(g, [ids["n1"], ids["n2"], ids["ad"]], "Adder")
    assert g.validate() == []
    assert len(g.nodes) == 2  # subworkflow + readout
    assert sub.params["label"] == "Adder"
    ex = Executor(g)
    assert ex.run_blocking().ok()
    assert ex.outputs[ids["ro"]]["value"] == 5.0


def test_expand_roundtrip_restores_and_reruns():
    g, ids = _adder_graph()
    sub = combine_nodes(g, [ids["n1"], ids["n2"], ids["ad"]], "Adder")
    restored = expand_subworkflow_node(g, sub.id)
    assert len(restored) == 3 and len(g.nodes) == 4 and len(g.links) == 3
    assert g.validate() == []
    ex = Executor(g)
    assert ex.run_blocking().ok()
    assert ex.outputs[ids["ro"]]["value"] == 5.0


def test_nested_subworkflows_run():
    g, ids = _adder_graph()
    inner = combine_nodes(g, [ids["n1"], ids["n2"]], "Pair")
    outer = combine_nodes(g, [inner.id, ids["ad"]], "Adder2")
    assert g.validate() == []
    ex = Executor(g)
    assert ex.run_blocking().ok(), {k: v.error for k, v in ex.run_blocking().per_node.items()}
    assert ex.outputs[ids["ro"]]["value"] == 5.0
    assert outer.params["label"] == "Adder2"


def test_max_recursions_zero_rejects_all():
    g, ids = _adder_graph()
    combine_nodes(g, [ids["n1"], ids["n2"], ids["ad"]], "Adder")
    ex = Executor(g, max_recursions=0)
    report = ex.run_blocking()
    assert not report.ok()
    assert any("max recursions (0)" in (r.error or "") for r in report.per_node.values())


def test_max_recursions_limits_nesting():
    g, ids = _adder_graph()
    inner = combine_nodes(g, [ids["n1"], ids["n2"]], "Pair")
    combine_nodes(g, [inner.id, ids["ad"]], "Adder2")
    ex = Executor(g, max_recursions=1)
    report = ex.run_blocking()
    assert not report.ok()
    assert any("max recursions (1)" in (r.error or "") for r in report.per_node.values())


def test_default_max_recursions_is_1000():
    from easygrapheditor.engine import DEFAULT_MAX_RECURSIONS

    assert DEFAULT_MAX_RECURSIONS == 1000
    assert Executor(Graph()).max_recursions == 1000


def test_inner_unknown_type_fails_validation():
    g, _ids = _adder_graph()
    sub_id = next(iter(g.nodes))
    g.nodes[sub_id].type_id = "core.subworkflow"
    g.nodes[sub_id].params = {"label": "Bad", "nodes": [{"id": "x", "type_id": "nope.missing", "params": {}}], "links": [], "inputs": [], "outputs": []}
    assert any("unknown inner type" in e.message for e in g.validate())


def test_subworkflow_serialization_roundtrip():
    g, ids = _adder_graph()
    combine_nodes(g, [ids["n1"], ids["n2"], ids["ad"]], "Adder")
    g2 = Graph.from_json(g.to_json())
    assert g2.validate() == []
    ex = Executor(g2)
    assert ex.run_blocking().ok()
    ro_id = next(nid for nid, inst in g2.nodes.items() if inst.type_id == "output.readout")
    assert ex.outputs[ro_id]["value"] == 5.0
