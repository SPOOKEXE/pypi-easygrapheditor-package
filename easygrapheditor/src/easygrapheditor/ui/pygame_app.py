"""Pygame backend: live node editor loop on top of DI adapters.

Canvas (ComfyUI-style):
  click/drag node ......... select / move (multi-select moves together,
                            grouped nodes move as one unless Shift held)
  shift+click / marquee ... add to selection (left-drag empty space)
  drag output port ........ wire (compatible inputs ringed); drop elsewhere cancels
  drag from linked input . reroute (unplugs, starts wire from the source)
  right-click input port .. unplug its links
  right/middle-drag ....... pan · wheel zooms (F resets view)
  N ....................... node library overlay (click to place, wheel scrolls)
  G / U ................... group selection / ungroup intersecting groups
  Del ..................... delete selection · Ctrl+C / Ctrl+V copy-paste
  [ ] ..................... tweak param (Shift = x10) · T toggle · D cycle
  E ....................... expand subworkflow · R run · L live · C clear · Q quit

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
    port_anchors,
    pygame_adjust_selected,
    pygame_draw_inspector,
    pygame_node_at,
    pygame_register_grapheditor,
)
from .canvas import EditorState
from .editing import (
    compatible_inputs,
    connect,
    disconnect,
    node_type_choices,
    port_value_preview,
    remove_nodes,
)
from .widgets import adjust_param_value


def _port_anchors(graph: Any, boxes: dict) -> tuple[dict, dict]:
    """Back-compat wrapper: (node -> [(port, x, y)]) without dtypes."""
    ins, outs = port_anchors(graph, boxes)
    strip = lambda anchors: {nid: [(p, x, y) for p, x, y, _ in items] for nid, items in anchors.items()}
    return strip(ins), strip(outs)


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
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    style = PygameStyle()
    graph_w = int(size[0] * 0.62)

    live = False
    msg = ""
    view = [0.0, 0.0, 1.0]  # ox, oy, zoom
    dragging: dict[str, tuple[int, int]] = {}
    wiring: tuple[str, str] | None = None
    wire_pos: tuple[int, int] = (0, 0)
    marquee: tuple[int, int] | None = None
    mouse_pos: tuple[int, int] = (0, 0)
    panning: tuple[int, int, int, int, bool] | None = None
    library_open = False
    placing: str | None = None
    lib_scroll = 0
    clipboard: dict | None = None

    def buttons() -> dict[str, Any]:
        labels = ["Run", f"Live:{'on' if live else 'off'}", "Clear", "+Add"]
        widths = [70, 90, 80, 70]
        rects, x = {}, 12
        for label, w in zip(labels, widths):
            rects[label.split(":")[0]] = pygame.Rect(x, 8, w, 30)
            x += w + 8
        rects["_msg_x"] = x
        return rects

    def base_box(nid: str, area: Any) -> Any:
        return compute_boxes(adapter.state.graph, area, style, offsets={}, view=tuple(view))[nid]

    frames = 0
    running = True

    def _mods(event: Any = None) -> int:
        real = pygame.key.get_mods()
        posted = getattr(event, "mod", 0) if event is not None else 0
        return real | posted

    def _shift(event: Any = None) -> bool:
        return bool(_mods(event) & (pygame.KMOD_SHIFT | pygame.KMOD_LSHIFT | pygame.KMOD_RSHIFT))

    def _ctrl(event: Any = None) -> bool:
        return bool(_mods(event) & (pygame.KMOD_CTRL | pygame.KMOD_LCTRL | pygame.KMOD_RCTRL))

    while running:
        graph_area = pygame.Rect(12, 46, graph_w - 24, size[1] - 58)
        insp_area = pygame.Rect(graph_w, 46, size[0] - graph_w - 12, size[1] - 58)
        boxes = compute_boxes(adapter.state.graph, graph_area, style, offsets=adapter.state.drag_offsets, view=tuple(view))
        ins, outs = port_anchors(adapter.state.graph, boxes)
        btns = buttons()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (2, 3):
                # right/middle: pan; right-click (no drag) on a linked input unplugs it
                pos = event.pos
                target = None
                if event.button == 3:
                    for nid, anchors in ins.items():
                        hit = _near([(p, x, y) for p, x, y, _ in anchors], pos)
                        if hit and any(l.to_node == nid and l.to_port == hit[0] for l in adapter.state.graph.links):
                            target = (nid, hit[0])
                            break
                panning = (pos[0] - view[0], pos[1] - view[1], pos[0], pos[1], False, target)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
                if library_open and lib_rect(graph_area).collidepoint(event.pos):
                    lib_scroll = max(0, lib_scroll + (22 if event.button == 5 else -22))
                else:
                    factor = 1.1 if event.button == 4 else 1 / 1.1
                    k_old = view[2]
                    k_new = min(2.5, max(0.35, k_old * factor))
                    mx, my = event.pos[0] - graph_area.x, event.pos[1] - graph_area.y
                    view[0] = event.pos[0] - graph_area.x - (mx - view[0]) * (k_new / k_old)
                    view[1] = event.pos[1] - graph_area.y - (my - view[1]) * (k_new / k_old)
                    view[2] = k_new
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
                elif minimap_rect(graph_area).collidepoint(pos):
                    centered = _minimap_center_on(adapter.state.graph, minimap_rect(graph_area), pos, graph_area, style)
                    if centered:
                        view[0], view[1] = centered
                else:
                    shift = _shift(event)
                    # reroute: drag from a linked input unplugs it and starts from the source
                    rerouted = None
                    for nid, anchors in ins.items():
                        hit = _near([(p, x, y) for p, x, y, _ in anchors], pos)
                        if hit and any(l.to_node == nid and l.to_port == hit[0] for l in adapter.state.graph.links):
                            src = next(l for l in adapter.state.graph.links if l.to_node == nid and l.to_port == hit[0])
                            for l in [x for x in adapter.state.graph.links if x.to_node == nid and x.to_port == hit[0]]:
                                disconnect(adapter.state.graph, l.from_node, l.from_port, l.to_node, l.to_port)
                            adapter.mark_dirty()
                            rerouted = (src.from_node, src.from_port)
                            msg = "rerouting link"
                            break
                    if rerouted:
                        wiring, wire_pos = rerouted, pos
                        continue
                    started = None
                    for nid, anchors in outs.items():
                        hit = _near([(p, x, y) for p, x, y, _ in anchors], pos)
                        if hit:
                            started = (nid, hit[0])
                            break
                    if started:
                        wiring, wire_pos = started, pos
                    else:
                        hit = pygame_node_at(adapter.state.graph, boxes, pos)
                        if hit is None:
                            marquee = pos if not shift else marquee or pos
                            if not shift:
                                adapter.state.selection = []
                        else:
                            if hit in adapter.state.selection and (shift or len(adapter.state.selection) > 1):
                                if shift and hit in adapter.state.selection:
                                    adapter.state.selection = [s for s in adapter.state.selection if s != hit]
                                # else: keep multi-selection for group-drag
                            else:
                                adapter.state.selection = [hit]
                            members = _move_set(adapter, hit, bool(shift))
                            dragging = {nid: (pos[0] - boxes[nid].x, pos[1] - boxes[nid].y) for nid in members}
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                if panning:
                    px, py, sx, sy, moved, target = panning
                    if abs(event.pos[0] - sx) + abs(event.pos[1] - sy) > 5:
                        moved = True
                    if moved:
                        view[0], view[1] = event.pos[0] - px, event.pos[1] - py
                    panning = (px, py, sx, sy, moved, target)
                elif wiring:
                    wire_pos = event.pos
                elif marquee:
                    pass  # end set on button-up
                elif dragging:
                    # each grabbed node follows its own grab point (formation preserved)
                    k = view[2]
                    for nid, (dx, dy) in dragging.items():
                        base = base_box(nid, graph_area)
                        lx = (base.x - graph_area.x - view[0]) / k
                        ly = (base.y - graph_area.y - view[1]) / k
                        adapter.state.drag_offsets[nid] = (
                            (event.pos[0] - dx - graph_area.x - view[0]) / k - lx,
                            (event.pos[1] - dy - graph_area.y - view[1]) / k - ly,
                        )
            elif event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
                if panning and event.button == 3:
                    _px, _py, _sx, _sy, moved, target = panning
                    if not moved and target:
                        nid, port = target
                        n = sum(1 for l in adapter.state.graph.links if l.to_node == nid and l.to_port == port)
                        adapter.state.graph.links = [l for l in adapter.state.graph.links if not (l.to_node == nid and l.to_port == port)]
                        adapter.mark_dirty()
                        msg = f"unplugged {n} link(s) into {nid}.{port}"
                        if live:
                            adapter.run()
                panning = None
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if wiring:
                    target = None
                    for nid, anchors in ins.items():
                        hit = _near([(p, x, y) for p, x, y, _ in anchors], event.pos)
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
                elif marquee:
                    x0, y0 = marquee
                    x1, y1 = event.pos
                    rect = pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
                    picked = [nid for nid, r in boxes.items() if rect.colliderect(r)]
                    adapter.state.selection = sorted(set(adapter.state.selection if _shift(event) else []) | set(picked))
                    msg = f"selected {len(adapter.state.selection)}"
                    marquee = None
                elif dragging:
                    dragging = {}
                    if live:
                        adapter.run()
            elif event.type == pygame.KEYDOWN:
                ctrl = _ctrl(event)
                factor = 10.0 if _shift(event) else 1.0
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    if wiring or marquee or library_open or placing:
                        wiring, marquee, library_open, placing = None, None, False, None
                    else:
                        running = False
                elif event.key == pygame.K_r:
                    adapter.run()
                    msg = "ran graph"
                elif event.key == pygame.K_l:
                    live = not live
                    msg = f"live {'on' if live else 'off'}"
                elif event.key == pygame.K_c and ctrl:
                    if adapter.state.selection:
                        from .editing import copy_selection

                        clipboard = copy_selection(adapter.state.graph, adapter.state.selection)
                        msg = f"copied {len(adapter.state.selection)} node(s)"
                elif event.key == pygame.K_v and ctrl:
                    if clipboard:
                        from .editing import paste_clipboard

                        adapter.state.selection = paste_clipboard(adapter.state.graph, clipboard)
                        adapter.mark_dirty()
                        msg = f"pasted {len(adapter.state.selection)} node(s)"
                        if live:
                            adapter.run()
                elif event.key == pygame.K_c:
                    adapter.clear_cache()
                    adapter.run()
                    msg = "cache cleared + reran"
                elif event.key == pygame.K_n:
                    library_open = not library_open
                elif event.key == pygame.K_g:
                    if len(adapter.state.selection) >= 2:
                        from .editing import group_nodes

                        try:
                            grp = group_nodes(adapter.state.graph, adapter.state.selection, f"Group {len(adapter.state.graph.groups) + 1}")
                            msg = f"grouped {len(grp.nodes)} nodes"
                        except ValueError as e:
                            msg = str(e)
                    else:
                        msg = "select 2+ nodes to group"
                elif event.key == pygame.K_u:
                    names = {g.name for nid in adapter.state.selection for g in [adapter.state.graph.group_of(nid)] if g}
                    if names:
                        from .editing import ungroup

                        for name in names:
                            ungroup(adapter.state.graph, name)
                        msg = f"ungrouped {len(names)}"
                    else:
                        msg = "selection is in no group"
                elif event.key == pygame.K_f:
                    view[0], view[1], view[2] = 0.0, 0.0, 1.0
                    msg = "view reset"
                elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                    if adapter.state.selection:
                        remove_nodes(adapter.state.graph, adapter.state.selection)
                        adapter.state.selection = []
                        adapter.mark_dirty()
                        msg = "deleted selection"
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

        compat: set[tuple[str, str]] = set()
        if wiring:
            compat = {(nid, port) for nid, port in compatible_inputs(adapter.state.graph, wiring[0], wiring[1])}
        screen.fill((12, 14, 18))
        _draw_bar(screen, btns, msg, style, live)
        pygame_register_grapheditor(screen, adapter, rect=graph_area, style=style, title=title,
                                    offsets=adapter.state.drag_offsets, view=tuple(view), highlight=compat)
        if wiring:
            import pygame as _pg

            _pg.draw.line(screen, (240, 220, 120), _wire_start(boxes, outs, wiring), wire_pos, 2)
        if marquee:
            import pygame as _pg

            x0, y0 = marquee
            _pg.draw.rect(screen, (140, 180, 240), pygame.Rect(min(x0, mouse_pos[0]), min(y0, mouse_pos[1]),
                                                              abs(mouse_pos[0] - x0), abs(mouse_pos[1] - y0)), 1)
        if library_open:
            _draw_library(screen, lib_rect(graph_area), node_type_choices(), lib_scroll, style)
        _draw_tooltip(screen, adapter, boxes, ins, outs, mouse_pos, wiring, style)
        _draw_minimap(screen, adapter, graph_area, style, tuple(view))
        pygame_draw_inspector(screen, adapter, insp_area, style)
        pygame.display.flip()
        adapter.state.viewport.x, adapter.state.viewport.y, adapter.state.viewport.zoom = view[0], view[1], view[2]
        clock.tick(30)
        frames += 1
        if max_frames and frames >= max_frames:
            running = False
    pygame.quit()
    return adapter


def _move_set(adapter: GraphAdapter, hit: str, single: bool) -> list[str]:
    """Nodes a drag moves: whole group (unless Shift) or current multi-selection."""
    if single:
        return [hit]
    group = adapter.state.graph.group_of(hit)
    if group and len(group.nodes) > 1:
        return list(group.nodes)
    if hit in adapter.state.selection and len(adapter.state.selection) > 1:
        return list(adapter.state.selection)
    return [hit]


def lib_rect(graph_area: Any) -> Any:
    import pygame

    return pygame.Rect(graph_area.x + 20, graph_area.y + 20, 320, min(480, graph_area.height - 40))


def minimap_rect(graph_area: Any) -> Any:
    import pygame

    return pygame.Rect(graph_area.x + 10, graph_area.bottom - 140, 190, 130)


def _minimap_center_on(graph: Any, mm: Any, pos: tuple[int, int], area: Any, style: PygameStyle) -> tuple[float, float] | None:
    """Shift the view so the clicked minimap node lands at area center. Returns (ox, oy)."""

    from .adapters import compute_boxes as _boxes

    mini = _boxes(graph, mm, style)
    if not mini:
        return None
    target = min(mini, key=lambda nid: abs(mini[nid].centerx - pos[0]) + abs(mini[nid].centery - pos[1]))
    main = _boxes(graph, area, style)
    box = main[target]
    return (area.centerx - box.centerx, area.centery - box.centery)


def _draw_minimap(screen: Any, adapter: GraphAdapter, area: Any, style: PygameStyle, view: tuple[float, float, float]) -> None:
    import pygame

    from .adapters import _make_font, _render_text
    from .adapters import compute_boxes as _boxes

    mm = minimap_rect(area)
    pygame.draw.rect(screen, (22, 25, 33), mm, border_radius=6)
    pygame.draw.rect(screen, (70, 76, 90), mm, 1, border_radius=6)
    mini = _boxes(adapter.state.graph, mm, style)
    for nid, rect in mini.items():
        rep = adapter.report.per_node.get(nid) if adapter.report else None
        color = style.idle if rep is None else (style.ok if rep.status in ("ok", "cached") else style.err)
        pygame.draw.rect(screen, color, rect, border_radius=1)
    _ = view
    screen.blit(_render_text(_make_font(11), "overview", style.dim), (mm.x + 6, mm.y + 4))


def _draw_tooltip(screen: Any, adapter: GraphAdapter, boxes: dict, ins: dict, outs: dict,
                  mouse: tuple[int, int], wiring: tuple[str, str] | None, style: PygameStyle) -> None:
    import pygame

    from .adapters import _make_font, _render_text

    small = _make_font(12)
    text = None
    for nid, anchors in {**outs, **ins}.items():
        for port, x, y, dtype in anchors:
            if abs(x - mouse[0]) <= 10 and abs(y - mouse[1]) <= 10:
                value = adapter.executor.outputs.get(nid, {}).get(port)
                direction = "out" if nid in outs and any(p == port for p, _, _, _ in outs[nid]) else "in"
                text = f"{port} [{dtype}] ({direction}) = {port_value_preview(value)}"
                break
        if text:
            break
    if text is None:
        hit = pygame_node_at(adapter.state.graph, boxes, mouse)
        if hit:
            inst = adapter.state.graph.nodes[hit]
            rep = adapter.report.per_node.get(hit) if adapter.report else None
            status = rep.status if rep else "not run"
            text = f"{inst.type_id} · {status}"
    if wiring and text is None:
        text = "drop on a ringed input · Esc cancels"
    if text is None:
        return
    surf = _render_text(small, text[:90], style.text)
    x = min(mouse[0] + 14, screen.get_width() - surf.get_width() - 6)
    y = min(mouse[1] + 16, screen.get_height() - surf.get_height() - 6)
    bg = pygame.Surface((surf.get_width() + 12, surf.get_height() + 8))
    bg.fill((28, 31, 40))
    screen.blit(bg, (x - 6, y - 4))
    screen.blit(surf, (x, y))


def _library_click(types: list, rect: Any, pos: tuple[int, int], scroll: int) -> str | None:
    if not rect.collidepoint(pos):
        return "__close__"
    idx = (pos[1] - rect.y - 30 + scroll) // 22
    if 0 <= idx < len(types):
        return types[int(idx)][0]
    return None


def _wire_start(boxes: dict, outs: dict, wiring: tuple[str, str]) -> tuple[int, int]:
    nid, port = wiring
    for p, x, y, _dt in outs.get(nid, []):
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
