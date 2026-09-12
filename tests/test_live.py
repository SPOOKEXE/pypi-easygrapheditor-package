"""Live editor interaction tests: gradio/streamlit/pygame editing paths."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine import Graph
from easygrapheditor.ui.adapters import GraphAdapter

ROOT = Path(__file__).resolve().parents[1]


def test_gradio_live_editor_builds_on_loop_graph():
    import gradio as gr

    sys.path.insert(0, str(ROOT / "easygrapheditor" / "examples"))
    import loop_demo

    g, _refs = loop_demo.build()
    adapter = GraphAdapter.from_any(g)
    adapter.run()
    with gr.Blocks() as blocks:
        from easygrapheditor.ui.adapters import gradio_register_grapheditor

        refs = gradio_register_grapheditor(blocks, adapter, title="Loop")
    assert {"summary", "canvas", "gallery", "table", "run", "live", "clear", "notice", "adapter"} <= set(refs)
    assert adapter.ok


def _loop_page() -> None:
    """Streamlit page body, fully self-contained (AppTest re-execs it without test globals)."""
    import sys
    from pathlib import Path

    import easygrapheditor

    pkg = Path(easygrapheditor.__file__).resolve()
    sys.argv = [
        "view_demo.py",
        "--demo",
        "loop",
        "--ui",
        "--backend",
        "streamlit",
    ]
    sys.path.insert(0, str(pkg.parents[1]))  # .../easygrapheditor/src
    sys.path.insert(0, str(pkg.parents[2] / "examples"))  # .../easygrapheditor/examples
    import view_demo

    view_demo.main()


def test_streamlit_live_editor_renders_and_clears():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_loop_page, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    labels = [e.label for e in at.expander]
    assert any("Library" in label for label in labels)
    assert any("Links" in label for label in labels)
    assert any("Delete" in label for label in labels)
    assert any("Loops" in label for label in labels)
    clear = [b for b in at.button if "Clear cache" in b.label]
    assert clear, [b.label for b in at.button]
    clear[0].click().run()
    assert not at.exception, at.exception


def _area_boxes(g):
    import pygame
    from easygrapheditor.ui.adapters import PygameStyle, compute_boxes

    area = pygame.Rect(12, 46, 682 - 24, 700 - 58)
    return area, compute_boxes(g, area, PygameStyle())


def test_pygame_drag_moves_node():
    import pygame
    from easygrapheditor.ui.adapters import PygameStyle, compute_boxes
    from easygrapheditor.ui.pygame_app import run_pygame

    g = Graph()
    n = g.add_node("input.number", params={"value": 1.0})
    g.add_node("output.readout", params={})
    adapter = GraphAdapter.from_any(g)
    adapter.run()

    pygame.init()
    pygame.event.clear()
    try:
        boxes = compute_boxes(g, pygame.Rect(12, 46, 682 - 24, 700 - 58), PygameStyle())
        c = boxes[n.id].center
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": c}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (c[0] + 40, c[1]), "rel": (40, 0), "buttons": (1, 0, 0)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (c[0] + 40, c[1])}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=4)
        assert adapter.state.drag_offsets.get(n.id, (0, 0))[0] == 40
    finally:
        pygame.quit()


def test_pygame_marquee_group_ungroup():
    import pygame
    from easygrapheditor.ui.pygame_app import run_pygame

    g = Graph()
    g.add_node("input.number", params={"value": 1.0})
    g.add_node("output.readout", params={})
    adapter = GraphAdapter.from_any(g)
    adapter.run()

    pygame.init()
    pygame.event.clear()
    try:
        _area, boxes = _area_boxes(g)
        ids = sorted(boxes)
        x0 = min(boxes[i].x for i in ids) - 10
        y0 = min(boxes[i].y for i in ids) - 10
        x1 = max(boxes[i].right for i in ids) + 10
        y1 = max(boxes[i].bottom for i in ids) + 10
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (x0, y0)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (x1, y1), "rel": (0, 0), "buttons": (1, 0, 0)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (x1, y1)}))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_g}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=6)
        assert len(adapter.state.selection) == 2
        assert len(g.groups) == 1 and len(g.groups[0].nodes) == 2
        pygame.init()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_u}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=3)
        assert g.groups == []
    finally:
        pygame.quit()


def test_pygame_right_click_unplug_and_reroute():
    import pygame
    from easygrapheditor.ui.pygame_app import _port_anchors, run_pygame

    g = Graph()
    n = g.add_node("input.number", params={"value": 1.0})
    a = g.add_node("maths.arithmetic", params={"operation": "add"})
    r = g.add_node("output.readout", params={})
    g.add_link(n.id, "out", a.id, "a")
    adapter = GraphAdapter.from_any(g)
    adapter.run()

    pygame.init()
    pygame.event.clear()
    try:
        _area, boxes = _area_boxes(g)
        ins, outs = _port_anchors(g, boxes)
        ix, iy = next((x, y) for p, x, y in ins[a.id] if p == "a")
        # right-click the linked input: unplug
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 3, "pos": (ix, iy)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 3, "pos": (ix, iy)}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=3)
        assert g.links == []
        # reroute: drag from output, drop on the readout input
        pygame.init()
        pygame.event.clear()
        _area, boxes = _area_boxes(g)
        ins, outs = _port_anchors(g, boxes)
        ox, oy = next((x, y) for p, x, y in outs[n.id] if p == "out")
        ix, iy = next((x, y) for p, x, y in ins[r.id] if p == "value_in")
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (ox, oy)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (ix, iy), "rel": (0, 0), "buttons": (1, 0, 0)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (ix, iy)}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=4)
        assert any(l.from_node == n.id and l.to_node == r.id for l in g.links)
    finally:
        pygame.quit()


def test_pygame_pan_zoom_copy_paste():
    import pygame
    from easygrapheditor.ui.pygame_app import run_pygame

    g = Graph()
    n = g.add_node("input.number", params={"value": 1.0})
    adapter = GraphAdapter.from_any(g)
    adapter.run()

    pygame.init()
    pygame.event.clear()
    try:
        # middle-drag pans (viewport must move)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 2, "pos": (400, 300)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (460, 300), "rel": (60, 0), "buttons": (0, 1, 0)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 2, "pos": (460, 300)}))
        # wheel zooms about cursor
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 4, "pos": (400, 300)}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=4)
        assert adapter.state.viewport.zoom == pytest.approx(1.1)
        assert adapter.state.viewport.x == pytest.approx(27.2, abs=0.5)
        # select + Ctrl+C / Ctrl+V duplicates the node
        adapter.state.selection = [n.id]
        pygame.init()
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c, "mod": pygame.KMOD_CTRL}))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_v, "mod": pygame.KMOD_CTRL}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=3)
        assert len(g.nodes) == 2
        assert len(adapter.state.selection) == 1 and adapter.state.selection[0] != n.id
    finally:
        pygame.quit()


def test_pygame_wire_and_delete():
    import pygame
    from easygrapheditor.ui.adapters import PygameStyle, compute_boxes
    from easygrapheditor.ui.pygame_app import _port_anchors, run_pygame

    g = Graph()
    n = g.add_node("input.number", params={"value": 1.0})
    a = g.add_node("maths.arithmetic", params={"operation": "add"})
    adapter = GraphAdapter.from_any(g)
    adapter.run()

    pygame.init()
    pygame.event.clear()
    try:
        boxes = compute_boxes(g, pygame.Rect(12, 46, 682 - 24, 700 - 58), PygameStyle())
        _ins, outs = _port_anchors(g, boxes)
        ox, oy = next((x, y) for p, x, y in outs[n.id] if p == "out")
        ins, _o = _port_anchors(g, boxes)
        ix, iy = next((x, y) for p, x, y in ins[a.id] if p == "a")
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (ox, oy)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (ix, iy), "rel": (0, 0), "buttons": (1, 0, 0)}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (ix, iy)}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=4)
        assert any(l.from_node == n.id and l.to_node == a.id for l in g.links), "wire drag should create a link"
        # run_pygame quits pygame on exit: re-init before posting more events
        pygame.init()
        pygame.event.clear()
        # layout changed after wiring: recompute boxes before clicking
        boxes = compute_boxes(g, pygame.Rect(12, 46, 682 - 24, 700 - 58), PygameStyle())
        # click the arithmetic node, then Delete it
        c = boxes[a.id].center
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": c}))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": c}))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DELETE}))
        run_pygame(adapter, size=(1100, 700), autorun=False, max_frames=4)
        assert a.id not in g.nodes and g.links == []
    finally:
        pygame.quit()
