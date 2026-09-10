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


def test_group_ungroup_and_prune():
    from easygrapheditor.ui.editing import group_nodes, ungroup

    g, n, r = _pair()
    with pytest.raises(ValueError, match="at least 2"):
        group_nodes(g, [n.id], "Solo")
    with pytest.raises(ValueError, match="Unknown nodes"):
        group_nodes(g, [n.id, r.id, "ghost"], "Bad")
    with pytest.raises(ValueError, match="color"):
        group_nodes(g, [n.id, r.id], "Bad", color="chartreuse")
    grp = group_nodes(g, [n.id, r.id], "Pair", color="blue")
    assert grp.name in [x.name for x in g.groups]
    with pytest.raises(ValueError, match="already grouped"):
        group_nodes(g, [n.id, r.id], "Again")
    freed = ungroup(g, grp.name)
    assert sorted(freed) == sorted([n.id, r.id]) and g.groups == []
    with pytest.raises(ValueError, match="Unknown group"):
        ungroup(g, grp.name)


def test_remove_prunes_groups():
    from easygrapheditor.ui.editing import group_nodes

    g, n, r = _pair()
    extra = g.add_node("input.number", params={"value": 3.0})
    group_nodes(g, [n.id, r.id, extra.id], "Trio")
    remove_node(g, extra.id)
    assert len(g.groups) == 1  # 2 members left: group survives
    remove_node(g, r.id)
    assert g.groups == []  # <2 members: group dropped


def test_groups_serialize():
    from easygrapheditor.ui.editing import group_nodes

    g, n, r = _pair()
    group_nodes(g, [n.id, r.id], "Pair")
    g2 = Graph.from_json(g.to_json())
    assert len(g2.groups) == 1 and g2.groups[0].title == "Pair"
    assert sorted(g2.groups[0].nodes) == sorted([n.id, r.id])


def test_copy_paste_roundtrip_runs_equal():
    from easygrapheditor.ui.editing import copy_selection, paste_clipboard

    g = Graph()
    n1 = g.add_node("input.number", params={"value": 2.0})
    n2 = g.add_node("input.number", params={"value": 3.0})
    ad = g.add_node("maths.arithmetic", params={"operation": "add"})
    g.add_link(n1.id, "out", ad.id, "a")
    g.add_link(n2.id, "out", ad.id, "b")
    clip = copy_selection(g, [n1.id, n2.id, ad.id])
    assert len(clip["nodes"]) == 3 and len(clip["links"]) == 2
    fresh = Graph()
    pasted = paste_clipboard(fresh, clip)
    assert len(pasted) == 3 and len(set(pasted)) == 3
    assert fresh.validate() == []
    assert fresh.nodes[pasted[0]].pos[0] == g.nodes[n1.id].pos[0] + 40.0
    ex = Executor(fresh)
    assert ex.run_blocking().ok()
    assert sorted(ex.outputs.values(), key=str)[-1] == {"out": 5.0}
    with pytest.raises(ValueError, match="Unknown nodes"):
        copy_selection(g, ["ghost"])


def test_port_value_preview():
    import numpy as np

    from easygrapheditor.engine.types import Field
    from easygrapheditor.ui.editing import port_value_preview

    assert port_value_preview(None) == "—"
    assert port_value_preview(2.5) == "2.5"
    assert port_value_preview(True) == "True"
    assert port_value_preview("x" * 200).endswith("…")
    assert "field (4, 4)" in port_value_preview(Field(data=np.zeros((4, 4), dtype=np.float32)))
    assert port_value_preview({"a": 1, "b": 2}) == "{a,b}"
