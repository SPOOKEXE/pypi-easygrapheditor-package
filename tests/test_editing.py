"""Graph editing tests: add/remove/connect/disconnect + canvas snapshot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

import pytest
from easygrapheditor.engine import Executor, Graph
from easygrapheditor.ui.editing import (
    add_node_at,
    compatible_inputs,
    connect,
    disconnect,
    layout_boxes_px,
    link_labels,
    node_type_choices,
    remove_node,
    render_canvas_image,
)


def _pair():
    g = Graph()
    n = g.add_node("input.number", params={"value": 1.0})
    r = g.add_node("output.readout", params={})
    return g, n, r


def test_add_node_defaults_and_unknown():
    g = Graph()
    inst = add_node_at(g, "generate.noise", pos=(10.0, 20.0))
    assert inst.pos == (10.0, 20.0) and inst.params["frequency"] == 4.0
    with pytest.raises(ValueError, match="Unknown node type"):
        add_node_at(g, "nope.missing")


def test_remove_node_drops_links_and_loop_refs():
    g, n, r = _pair()
    g.add_link(n.id, "out", r.id, "value_in")
    cond = g.add_node("control.end_condition", params={"mode": "counter", "iterations": 1})
    g.add_loop("l", [n.id, r.id, cond.id], cond.id)
    remove_node(g, r.id)
    assert r.id not in g.nodes and g.links == []
    assert r.id not in g.loops[0].body
    with pytest.raises(ValueError, match="Unknown node"):
        remove_node(g, r.id)


def test_connect_ok_and_errors():
    g, n, r = _pair()
    link = connect(g, n.id, "out", r.id, "value_in")
    assert (link.from_node, link.from_port) == (n.id, "out")
    with pytest.raises(ValueError, match="already exists"):
        connect(g, n.id, "out", r.id, "value_in")
    with pytest.raises(ValueError, match="no input port"):
        connect(g, n.id, "out", r.id, "nope")
    with pytest.raises(ValueError, match="Unknown node"):
        connect(g, "ghost", "out", r.id, "value_in")
    # number (ANY) -> slope (FIELD): ANY connects, so use readout(NUMBER out) -> slope for mismatch
    g2 = Graph()
    ro = g2.add_node("output.readout", params={})
    sl = g2.add_node("filter.slope", params={})
    with pytest.raises(ValueError, match="Type mismatch"):
        connect(g2, ro.id, "value", sl.id, "field_in")


def test_connect_rejects_cycles_but_allows_feedback():
    g = Graph()
    a = g.add_node("maths.arithmetic", params={"operation": "add"})
    b = g.add_node("maths.arithmetic", params={"operation": "add"})
    g.add_link(a.id, "out", b.id, "a")
    with pytest.raises(ValueError, match="cycle"):
        connect(g, b.id, "out", a.id, "a")
    # loop-carried feedback through accumulate is legal
    acc = g.add_node("control.accumulate", params={"initial": 0.0})
    connect(g, b.id, "out", acc.id, "next")
    assert g.validate() == [] or all("Cycle" not in e.message for e in g.validate())


def test_disconnect_and_labels():
    g, n, r = _pair()
    assert disconnect(g, n.id, "out", r.id, "value_in") is False
    g.add_link(n.id, "out", r.id, "value_in")
    assert link_labels(g) == [f"{n.id}.out -> {r.id}.value_in"]
    assert disconnect(g, n.id, "out", r.id, "value_in") is True
    assert g.links == []


def test_compatible_inputs():
    g, n, r = _pair()
    assert (r.id, "value_in") in compatible_inputs(g, n.id, "out")
    assert all(nid != n.id for nid, _ in compatible_inputs(g, n.id, "out"))
    assert compatible_inputs(g, n.id, "nope") == []


def test_layout_and_canvas_snapshot():
    g = Graph()
    g.add_node("generate.noise", params={"resolution": 16, "frequency": 2.0, "seed": 1.0, "octaves": 1, "gain": 0.5})
    boxes, _pos = layout_boxes_px(g, 800, 600)
    assert len(boxes) == 1
    ex = Executor(g)
    assert ex.run_blocking().ok()
    img = render_canvas_image(g, ex.run_blocking(), ex.outputs, width=800, height=600)
    assert img.size == (800, 600) and img.mode == "RGB"
    colors = img.getcolors(maxcolors=1 << 20)
    assert colors is not None and len(colors) > 4  # bg + grid + box + thumbnail


def test_node_type_choices_sorted():
    choices = node_type_choices()
    assert len(choices) > 10
    assert all("[" in label for _, label in choices)
