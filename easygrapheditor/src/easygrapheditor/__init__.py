"""EasyGraphEditor, a Gradio-style node graph editor. See editor.md."""

from __future__ import annotations

from pathlib import Path

from . import nodes_ai as _ai  # noqa: F401 - registers AI trace nodes
from . import nodes_builtin as _builtin  # noqa: F401 - registers builtins
from . import nodes_comfy_demo as _comfy  # noqa: F401 - registers mocks
from . import nodes_control as _control  # noqa: F401 - registers loop primitives
from .engine import ANY as Any
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

__version__ = "0.2.0"

__all__ = [
    "Any",
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
        self._scoped = nodes is not None
        self.nodes = nodes if nodes is not None else list_nodes()
        self.state = EditorState(graph=Graph())
        self.state.tabs[0].title = title
        self.state.allowed_node_types = self.node_types if self._scoped else None

    @property
    def graph(self) -> Graph:
        """The graph owned by the active editor tab."""
        return self.state.graph

    @graph.setter
    def graph(self, value: Graph) -> None:
        self.state.graph = value
        if self.state._tab_graphs:
            self.state._tab_graphs[self.state.active_tab] = value

    @property
    def node_types(self) -> set[str]:
        """Registered types allowed by this editor's library scope.

        Passing decorated functions or ``NodeDef`` objects to ``Editor(nodes=``
        narrows the add-node library. Execution still accepts a loaded graph,
        so old files are not made impossible to inspect.
        """
        types: set[str] = set()
        for item in self.nodes:
            definition = getattr(item, "_node_def", item)
            type_id = getattr(definition, "type_id", None)
            if isinstance(type_id, str):
                types.add(type_id)
        return types

    def add_node(self, type_id: str, pos: tuple[float, float] = (0.0, 0.0)):
        """Add a node through the undoable editor command boundary."""
        if self._scoped and type_id not in self.node_types:
            raise ValueError(f"Node type {type_id!r} is outside this Editor's scoped nodes")
        from .ui.editing import add_node_at

        return self.state.apply("add node", lambda: add_node_at(self.graph, type_id, pos=pos))

    def run_headless(self, graph_path: str | Path | None = None) -> RunReport:
        """Execute the current graph, or load and execute an ``.ege.json`` path."""
        if graph_path is not None:
            self.load(graph_path)
        report = Executor(self.graph).run_blocking()
        self.state.run_report = report
        self.state.log(f"run: {'ok' if report.ok() else 'errors'}")
        return report

    def save(self, path: str | Path) -> Path:
        if self.state.tabs:
            self.state.tabs[self.state.active_tab].dirty = False
        saved = save_graph(self.graph, path, view=self.state.view_dict())
        self.state.log(f"saved {saved.name}")
        return saved

    def load(self, path: str | Path) -> None:
        graph, _view = load_graph(path)
        self.state = EditorState(graph=graph)
        self.state.allowed_node_types = self.node_types if self._scoped else None
        self.state.load_view(_view)
        self.state.log(f"loaded {Path(path).name}")

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

        Defaults to the current page. DI style means the caller owns the script run.
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
