"""Widget spec/cast/adjust tests (pure, no UI toolkit imports)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine.nodes import ParamDef, canonical_kind
from easygrapheditor.ui.widgets import adjust_param_value, cast_param_value, widget_spec


def test_canonical_kind_aliases():
    assert canonical_kind("slider") == "float_slider"
    assert canonical_kind("multiline") == "textarea"
    assert canonical_kind("dropdown") == "select"
    assert canonical_kind("toggle") == "checkbox"
    assert canonical_kind("float_slider") == "float_slider"
    assert canonical_kind("number") == "number"


def test_specs_cover_every_widget_family():
    cases = {
        "float_slider": {"widget": "slider", "decimal": True},
        "step_slider": {"widget": "stepslider"},
        "int": {"widget": "numeric_int"},
        "number": {"widget": "numeric"},
        "seed": {"widget": "seed"},
        "text": {"widget": "input"},
        "textarea": {"widget": "textarea"},
        "select": {"widget": "options"},
        "checkbox": {"widget": "boolean"},
        "file": {"widget": "file"},
    }
    for kind, expected in cases.items():
        param = ParamDef(key="k", label="K", kind=kind, default=0, min=0, max=10, step=1, options=["a", "b"])
        spec = widget_spec(param)
        for field, want in expected.items():
            assert spec[field] == want, (kind, spec)
    slider = widget_spec(ParamDef(key="k", label="K", kind="float_slider", default=0.5, min=0, max=1, step=0.01))
    assert (slider["min"], slider["max"], slider["step"]) == (0.0, 1.0, 0.01)


def test_cast_param_value():
    assert cast_param_value(ParamDef(key="k", label="K", kind="number", default=1.5), "2") == 2.0
    assert cast_param_value(ParamDef(key="k", label="K", kind="int", default=1), 2.7) == 2
    assert cast_param_value(ParamDef(key="k", label="K", kind="checkbox", default=False), "true") is True
    assert cast_param_value(ParamDef(key="k", label="K", kind="checkbox", default=False), 0) is False
    assert cast_param_value(ParamDef(key="k", label="K", kind="number", default=1.5), "oops") == 1.5


def test_adjust_param_value():
    slider = ParamDef(key="k", label="K", kind="float_slider", default=0.5, min=0, max=1, step=0.1)
    assert adjust_param_value(slider, 0.5, +1) == 0.6
    assert adjust_param_value(slider, 0.95, +1) == 1.0  # clamped
    assert adjust_param_value(slider, 0.5, -1, factor=10.0) == 0.0
    boolean = ParamDef(key="k", label="K", kind="checkbox", default=False)
    assert adjust_param_value(boolean, False, +1) is True
    options = ParamDef(key="k", label="K", kind="select", default="a", options=["a", "b", "c"])
    assert adjust_param_value(options, "c", +1) == "a"  # wraps
    assert adjust_param_value(options, "a", -1) == "c"
    text = ParamDef(key="k", label="K", kind="text", default="hi")
    assert adjust_param_value(text, "hi", +1) == "hi"  # not adjustable


def test_pygame_selection_helpers():
    import pygame
    from easygrapheditor.engine import Graph
    from easygrapheditor.ui.adapters import (
        GraphAdapter,
        PygameStyle,
        compute_boxes,
        pygame_adjust_selected,
        pygame_draw_inspector,
        pygame_node_at,
    )

    pygame.init()
    try:
        g = Graph()
        n = g.add_node("input.number", params={"value": 2.0})
        screen = pygame.Surface((900, 600))
        area = screen.get_rect()
        boxes = compute_boxes(g, area, PygameStyle())
        assert pygame_node_at(g, boxes, boxes[n.id].center) == n.id
        adapter = GraphAdapter.from_any(g)
        assert pygame_adjust_selected(adapter, +1) is None  # nothing selected
        adapter.state.selection = [n.id]
        assert pygame_adjust_selected(adapter, +1) == f"{n.id}.value -> 3.0"
        insp = pygame.Surface((300, 600))
        assert pygame_draw_inspector(insp, adapter, insp.get_rect()) is not None
    finally:
        pygame.quit()


def test_describe_node_subflow_detail():
    from easygrapheditor.engine import Graph, combine_nodes
    from easygrapheditor.ui.adapters import GraphAdapter

    g = Graph()
    n1 = g.add_node("input.number", params={"value": 1.0})
    n2 = g.add_node("input.number", params={"value": 2.0})
    sub = combine_nodes(g, [n1.id, n2.id], "Pair")
    detail = GraphAdapter.from_any(g).describe_node(sub.id)
    assert detail["subflow"]["nodes"] == 2
    assert "Number" in detail["subflow"]["titles"]
