"""Pygame backend: live node editor loop on top of DI adapters.

Click/drag nodes to move them, drag output ports onto input ports to wire
(type-checked, cycles rejected), N opens the node library, Del removes,
L toggles live auto-run, C clears the cache, E expands a subworkflow.

For embedding into your own game/app, call
:func:`easygrapheditor.ui.adapters.pygame_register_grapheditor` with your
screen Surface each frame — no loop is owned there.
"""

from __future__ import annotations

from typing import Any

from ..engine.nodes import canonical_kind, get_node
from .adapters import (
    GraphAdapter,
    PygameStyle,
    compute_boxes,
    pygame_adjust_selected,
    pygame_draw_inspector,
    pygame_node_at,
    pygame_register_grapheditor,
)
from .canvas import EditorState
from .editing import connect, node_type_choices, remove_node
from .widgets import adjust_param_value


def _port_anchors(graph: Any, boxes: dict) -> tuple[dict, dict]:
    """(node -> [(port, x, y)]) for input (left) / output (right) edges."""
    ins: dict[str, list] = {}
    outs: dict[str, list] = {}
    for nid, inst in graph.nodes.items():
        box = boxes.get(nid)
        if box is None:
            continue
        ndef = get_node(inst.type_id)
        in_ports = [p.key for p in ndef.inputs] if ndef else []
        if inst.type_id == "core.subworkflow":
            in_ports = [m["key"] for m in inst.params.get("inputs", [])]
        out_ports = [p.key for p in ndef.outputs] if ndef else []
        if inst.type_id == "core.subworkflow":
            out_ports = [m["key"] for m in inst.params.get("outputs", [])]
        ins[nid] = [(p, box.x, int(box.y + (i + 1) * box.height / (len(in_ports) + 1))) for i, p in enumerate(in_ports)]
        outs[nid] = [(p, box.right, int(box.y + (i + 1) * box.height / (len(out_ports) + 1))) for i, p in enumerate(out_ports)]
    return ins, outs


def _near(points: list, pos: tuple[int, int], radius: int = 9) -> tuple | None:
    for key, x, y in points:
        if abs(x - pos[0]) <= radius and abs(y - pos[1]) <= radius:
            return key, x, y
    return None


def run_pygame(
    state: EditorState | GraphAdapter,
    size: tuple[int, int] = (1100, 700),
    title: str = "Graph Editor",
    autorun: bool = True,
    max_frames: int = 0,
) -> GraphAdapter:
    """Open the live editor window (see module docstring for controls).

    ``max_frames`` caps the loop (0 = unlimited); used by headless tests with
    a posted QUIT event.
    """
    try:
        import pygame
    except ImportError as e:
        raise ImportError("Install the pygame extra: pip install 'easygrapheditor[pygame]'") from e

    adapter = GraphAdapter.from_any(state)
    if autorun and adapter.report is None:
        adapter.run()
    offsets = adapter.state.drag_offsets
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    style = PygameStyle()
    graph_w = int(size[0] * 0.62)

    live = False
    msg = ""
    dragging: tuple[str, int, int] | None = None
    wiring: tuple[str, str] | None = None
    wire_pos: tuple[int, int] = (0, 0)
    library_open = False
    placing: str | None = None
    lib_scroll = 0

    def buttons() -> dict[str, Any]:
        labels = ["Run", f"Live:{'on' if live else 'off'}", "Clear", "+Add"]
        widths = [70, 90, 80, 70]
        rects, x = {}, 12
        for label, w in zip(labels, widths):
            rects[label.split(":")[0]] = pygame.Rect(x, 8, w, 30)
            x += w + 8
        rects["_msg_x"] = x
        return rects

    frames = 0
    running = True
    while running:
        graph_area = pygame.Rect(12, 46, graph_w - 24, size[1] - 58)
        insp_area = pygame.Rect(graph_w, 46, size[0] - graph_w - 12, size[1] - 58)
        boxes = compute_boxes(adapter.state.graph, graph_area, style, offsets=offsets)
        ins, outs = _port_anchors(adapter.state.graph, boxes)
        btns = buttons()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                hit_btn = next((k for k, r in btns.items() if not k.startswith("_") and r.collidepoint(pos)), None)
                if hit_btn == "Run":
                    adapter.run()
                    msg = "ran graph"
                elif hit_btn == "Live":
                    live = not live
                    msg = f"live {'on' if live else 'off'}"
                elif hit_btn == "Clear":
                    adapter.clear_cache()
                    adapter.run()
                    msg = "cache cleared + reran"
                elif hit_btn == "+Add":
                    library_open, placing = True, None
                elif library_open:
                    picked = _library_click(node_type_choices(), lib_rect(graph_area), pos, lib_scroll)
                    if picked == "__close__":
                        library_open = False
                    elif picked:
                        placing, library_open = picked, False
                        msg = f"placing {picked}: click canvas"
                elif placing:
                    from .editing import add_node_at

                    inst = add_node_at(adapter.state.graph, placing, pos=(float(pos[0]), float(pos[1])))
                    adapter.mark_dirty()
                    msg = f"added {inst.id}"
                    placing = None
                    if live:
                        adapter.run()
                else:
                    # port wiring wins over dragging when starting on an output anchor
                    started = None
                    for nid, anchors in outs.items():
                        hit = _near([(p, x, y) for p, x, y in anchors], pos)
                        if hit:
                            started = (nid, hit[0])
                            break
                    if started:
                        wiring, wire_pos = started, pos
                    else:
                        hit = pygame_node_at(adapter.state.graph, boxes, pos)
                        adapter.state.selection = [hit] if hit else []
                        if hit:
                            r = boxes[hit]
                            dragging = (hit, pos[0] - r.x, pos[1] - r.y)
            elif event.type == pygame.MOUSEMOTION:
                if wiring:
                    wire_pos = event.pos
                elif dragging:
                    nid, dx, dy = dragging
                    # store drag as pixel offset from layout position
                    base = compute_boxes(adapter.state.graph, graph_area, style)[nid]
                    offsets[nid] = (event.pos[0] - dx - base.x, event.pos[1] - dy - base.y)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if wiring:
                    target = None
                    for nid, anchors in ins.items():
                        hit = _near([(p, x, y) for p, x, y in anchors], event.pos)
                        if hit:
                            target = (nid, hit[0])
                            break
                    if target:
                        try:
                            link = connect(adapter.state.graph, wiring[0], wiring[1], target[0], target[1])
                            adapter.mark_dirty()
                            msg = f"connected {link.from_node}.{link.from_port} -> {link.to_node}.{link.to_port}"
                            if live:
                                adapter.run()
                        except ValueError as e:
                            msg = f"rejected: {e}"
                    wiring = None
                elif dragging:
                    dragging = None
                    if live:
                        adapter.run()
            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                factor = 10.0 if mods & (pygame.KMOD_SHIFT | pygame.KMOD_LSHIFT | pygame.KMOD_RSHIFT) else 1.0
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    if library_open or placing:
                        library_open, placing = False, None
                    else:
                        running = False
                elif event.key == pygame.K_r:
                    adapter.run()
                    msg = "ran graph"
                elif event.key == pygame.K_l:
                    live = not live
                    msg = f"live {'on' if live else 'off'}"
                elif event.key == pygame.K_c:
                    adapter.clear_cache()
                    adapter.run()
                    msg = "cache cleared + reran"
                elif event.key == pygame.K_n:
                    library_open = not library_open
                elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                    if adapter.state.selection:
                        remove_node(adapter.state.graph, adapter.state.selection[0])
                        adapter.state.selection = []
                        adapter.mark_dirty()
                        msg = "deleted node"
                        if live:
                            adapter.run()
                elif event.key == pygame.K_LEFTBRACKET:
                    _tweak_and_maybe_run(adapter, -1, factor, live)
                elif event.key == pygame.K_RIGHTBRACKET:
                    _tweak_and_maybe_run(adapter, +1, factor, live)
                elif event.key == pygame.K_t:
                    _flip_first(adapter, ("checkbox", "toggle"), live)
                elif event.key == pygame.K_d:
                    _flip_first(adapter, ("select", "dropdown"), live)
                elif event.key == pygame.K_e and _expand_selected(adapter, live):
                    msg = "expanded subworkflow"

        screen.fill((12, 14, 18))
        _draw_bar(screen, btns, msg, style, live)
        pygame_register_grapheditor(screen, adapter, rect=graph_area, style=style, title=title, offsets=offsets)
        if wiring:
            import pygame as _pg

            _pg.draw.line(screen, (240, 220, 120), _wire_start(boxes, outs, wiring), wire_pos, 2)
        if library_open:
            _draw_library(screen, lib_rect(graph_area), node_type_choices(), lib_scroll, style)
        pygame_draw_inspector(screen, adapter, insp_area, style)
        pygame.display.flip()
        clock.tick(30)
        frames += 1
        if max_frames and frames >= max_frames:
            running = False
    pygame.quit()
    return adapter


def lib_rect(graph_area: Any) -> Any:
    import pygame

    return pygame.Rect(graph_area.x + 20, graph_area.y + 20, 320, min(480, graph_area.height - 40))


def _library_click(types: list, rect: Any, pos: tuple[int, int], scroll: int) -> str | None:
    if not rect.collidepoint(pos):
        return "__close__"
    idx = (pos[1] - rect.y - 30 + scroll) // 22
    if 0 <= idx < len(types):
        return types[int(idx)][0]
    return None


def _wire_start(boxes: dict, outs: dict, wiring: tuple[str, str]) -> tuple[int, int]:
    nid, port = wiring
    for p, x, y in outs.get(nid, []):
        if p == port:
            return (x, y)
    return boxes[nid].midright


def _draw_bar(screen: Any, btns: dict, msg: str, style: PygameStyle, live: bool) -> None:
    import pygame

    from .adapters import _make_font, _render_text

    font = _make_font(14)
    small = _make_font(12)
    static = {"Run": "Run [R]", "Clear": "Clear [C]", "+Add": "+Add [N]"}
    for key, rect in btns.items():
        if key.startswith("_"):
            continue
        label = f"Live:{'on' if live else 'off'} [L]" if key == "Live" else static.get(key, key)
        pygame.draw.rect(screen, (45, 50, 62), rect, border_radius=5)
        screen.blit(_render_text(font, label, style.text), (rect.x + 8, rect.y + 6))
    if msg:
        screen.blit(_render_text(small, msg[:80], style.dim), (btns["_msg_x"] + 4, btns["Run"].y + 8))


def _draw_library(screen: Any, rect: Any, types: list, scroll: int, style: PygameStyle) -> None:
    import pygame

    from .adapters import _make_font, _render_text

    font = _make_font(14)
    small = _make_font(12)
    pygame.draw.rect(screen, (28, 31, 40), rect, border_radius=8)
    pygame.draw.rect(screen, style.edge, rect, 2, border_radius=8)
    screen.blit(_render_text(font, "Library: click to place (Esc cancels)", style.text), (rect.x + 10, rect.y + 8))
    for i, (_tid, label) in enumerate(types):
        y = rect.y + 30 + i * 22 - scroll
        if rect.y + 28 <= y <= rect.bottom - 20:
            screen.blit(_render_text(small, label[:40], style.dim), (rect.x + 12, y))


def _selected_param(adapter: GraphAdapter, kinds: tuple[str, ...]) -> Any:
    if not adapter.state.selection:
        return None
    inst = adapter.state.graph.nodes.get(adapter.state.selection[0])
    ndef = get_node(inst.type_id) if inst else None
    if inst is None or ndef is None:
        return None
    for pdef in ndef.params:
        if canonical_kind(pdef.kind) in kinds:
            return inst, pdef
    return None


def _tweak_and_maybe_run(adapter: GraphAdapter, direction: int, factor: float, live: bool) -> None:
    if pygame_adjust_selected(adapter, direction, factor) is not None and live:
        adapter.run()


def _flip_first(adapter: GraphAdapter, kinds: tuple[str, ...], live: bool) -> None:
    found = _selected_param(adapter, kinds)
    if found is None:
        return
    inst, pdef = found
    inst.params[pdef.key] = adjust_param_value(pdef, inst.params.get(pdef.key, pdef.default), +1)
    if live:
        adapter.run()


def _expand_selected(adapter: GraphAdapter, live: bool) -> bool:
    if not adapter.state.selection:
        return False
    nid = adapter.state.selection[0]
    inst = adapter.state.graph.nodes.get(nid)
    if inst is not None and inst.type_id == "core.subworkflow":
        adapter.expand_subworkflow(nid)
        if live:
            adapter.run()
        return True
    return False


def run_pygame_viewer(state: EditorState | GraphAdapter, **kw: Any) -> GraphAdapter:
    """Alias of :func:`run_pygame` used by the demo launcher."""
    return run_pygame(state, **kw)
