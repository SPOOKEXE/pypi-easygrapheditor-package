"""Pygame backend: standalone viewer loop on top of DI adapters.

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
from .widgets import adjust_param_value


def run_pygame(
    state: EditorState | GraphAdapter,
    size: tuple[int, int] = (1100, 700),
    title: str = "Graph Editor",
    autorun: bool = True,
    max_frames: int = 0,
) -> GraphAdapter:
    """Open a window viewer with click-select + keyboard manipulation.

    Click a node to select it (stored in ``state.selection``). Keys:
    ``[`` / ``]`` tweak the first numeric/slider param (Shift = x10),
    ``T`` flips the first checkbox, ``D`` cycles the first dropdown,
    ``E`` expands a selected subworkflow, ``R`` re-runs, ``Q``/Esc quits.

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
    graph_w = int(size[0] * 0.68)
    frames = 0
    running = True
    while running:
        graph_area = pygame.Rect(12, 12, graph_w - 24, size[1] - 24)
        insp_area = pygame.Rect(graph_w, 12, size[0] - graph_w - 12, size[1] - 24)
        boxes = compute_boxes(adapter.state.graph, graph_area, style)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hit = pygame_node_at(adapter.state.graph, boxes, event.pos)
                adapter.state.selection = [hit] if hit else []
            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                factor = 10.0 if mods & (pygame.KMOD_SHIFT | pygame.KMOD_LSHIFT | pygame.KMOD_RSHIFT) else 1.0
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    adapter.run()
                elif event.key == pygame.K_LEFTBRACKET:
                    _tweak_and_rerun(adapter, -1, factor)
                elif event.key == pygame.K_RIGHTBRACKET:
                    _tweak_and_rerun(adapter, +1, factor)
                elif event.key == pygame.K_t:
                    _flip_first(adapter, ("checkbox", "toggle"))
                elif event.key == pygame.K_d:
                    _cycle_first(adapter, ("select", "dropdown"))
                elif event.key == pygame.K_e:
                    _expand_selected(adapter)
        pygame_register_grapheditor(screen, adapter, rect=graph_area, style=style, title=title)
        pygame_draw_inspector(screen, adapter, insp_area, style)
        pygame.display.flip()
        clock.tick(30)
        frames += 1
        if max_frames and frames >= max_frames:
            running = False
    pygame.quit()
    return adapter


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


def _tweak_and_rerun(adapter: GraphAdapter, direction: int, factor: float) -> None:
    if pygame_adjust_selected(adapter, direction, factor) is not None:
        adapter.run()


def _flip_first(adapter: GraphAdapter, kinds: tuple[str, ...]) -> None:
    found = _selected_param(adapter, kinds)
    if found is None:
        return
    inst, pdef = found
    inst.params[pdef.key] = adjust_param_value(pdef, inst.params.get(pdef.key, pdef.default), +1)
    adapter.run()


def _cycle_first(adapter: GraphAdapter, kinds: tuple[str, ...]) -> None:
    _flip_first(adapter, kinds)  # adjust_param_value cycles options with direction=+1


def _expand_selected(adapter: GraphAdapter) -> None:
    if not adapter.state.selection:
        return
    nid = adapter.state.selection[0]
    inst = adapter.state.graph.nodes.get(nid)
    if inst is not None and inst.type_id == "core.subworkflow":
        adapter.expand_subworkflow(nid)
        adapter.run()


def run_pygame_viewer(state: EditorState | GraphAdapter, **kw: Any) -> GraphAdapter:
    """Alias of :func:`run_pygame` used by the demo launcher."""
    return run_pygame(state, **kw)
