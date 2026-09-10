"""Pygame backend: standalone viewer loop on top of DI adapters.

For embedding into your own game/app, call
:func:`easygrapheditor.ui.adapters.pygame_register_grapheditor` with your
screen Surface each frame — no loop is owned there.
"""

from __future__ import annotations

from typing import Any

from .adapters import GraphAdapter, PygameStyle, pygame_register_grapheditor
from .canvas import EditorState


def run_pygame(
    state: EditorState | GraphAdapter,
    size: tuple[int, int] = (1100, 700),
    title: str = "Graph Editor",
    autorun: bool = True,
    max_frames: int = 0,
) -> GraphAdapter:
    """Open a window viewer: ``R`` re-runs the graph, ``Q``/Esc quits.

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
    frames = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    adapter.run()
        pygame_register_grapheditor(screen, adapter, style=style, title=title)
        pygame.display.flip()
        clock.tick(30)
        frames += 1
        if max_frames and frames >= max_frames:
            running = False
    pygame.quit()
    return adapter


def run_pygame_viewer(state: EditorState | GraphAdapter, **kw: Any) -> GraphAdapter:
    """Alias of :func:`run_pygame` used by the demo launcher."""
    return run_pygame(state, **kw)
