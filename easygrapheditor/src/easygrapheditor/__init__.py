"""EasyGraphEditor — Gradio-style node graph editor. See editor.md."""

from __future__ import annotations

from pathlib import Path

from . import nodes_ai as _ai  # noqa: F401 - registers AI trace nodes
from . import nodes_builtin as _builtin  # noqa: F401 - registers builtins
from . import nodes_comfy_demo as _comfy  # noqa: F401 - registers mocks
from . import nodes_control as _control  # noqa: F401 - registers loop primitives
from .engine import (
    Cache,
    Executor,
    Field,
    Graph,
    Image,
    Number,
    Param,
    list_nodes,
    load_graph,
    node,
    save_graph,
)
from .engine.execute import RunReport
from .ui.canvas import EditorState

__version__ = "0.1.0"

__all__ = [
    "Cache",
    "Editor",
    "EditorState",
    "Executor",
    "Field",
    "Graph",
    "Image",
    "Number",
    "Param",
    "RunReport",
    "__version__",
    "list_nodes",
    "load_graph",
    "node",
    "save_graph",
]


class Editor:
    """Gradio-style facade: declare nodes, get an editor."""

    def __init__(self, nodes: list | None = None, title: str = "EasyGraphEditor") -> None:
        self.title = title
        self.nodes = nodes if nodes is not None else list_nodes()
        self.graph = Graph()
        self.state = EditorState(graph=self.graph)

    def run_headless(self) -> RunReport:
        return Executor(self.graph).run_blocking()

    def save(self, path: str | Path) -> Path:
        return save_graph(self.graph, path)

    def load(self, path: str | Path) -> None:
        graph, _view = load_graph(path)
        self.graph = graph
        self.state = EditorState(graph=graph)

    def to_gradio(self, host=None, launch: bool = False, **opts):  # type: ignore[no-untyped-def]
        """Visualise in Gradio. Pass your ``gr.Blocks``/``gr.Tab`` as ``host``
        to embed (DI style); otherwise a standalone app is built (and launched
        with ``launch=True``)."""
        if host is not None:
            from .ui.adapters import gradio_register_grapheditor

            return gradio_register_grapheditor(host, self.state, title=self.title, **opts)
        from .ui.gradio_app import build_gradio_app

        app = build_gradio_app(self.state, title=self.title, **opts)
        return app.launch() if launch else app

    def to_streamlit(self, container=None, **opts):  # type: ignore[no-untyped-def]
        """Render into your Streamlit container (tab/sidebar/expander/Page).

        Defaults to the current page. DI style — we never own the script run.
        """
        from .ui.streamlit_app import build_streamlit_app

        return build_streamlit_app(self.state, container=container, **opts)

    def to_pygame(self, screen=None, **opts):  # type: ignore[no-untyped-def]
        """Draw onto your pygame Surface (DI style), or open a viewer window
        when ``screen`` is None."""
        if screen is not None:
            from .ui.adapters import pygame_register_grapheditor

            return pygame_register_grapheditor(screen, self.state, **opts)
        from .ui.pygame_app import run_pygame

        return run_pygame(self.state, title=self.title, **opts)
