"""Shared, command-based editor state used by every frontend.

The engine intentionally knows nothing about cursor state, layouts beyond node
positions, or browser preferences. This module is the small boundary where
those concerns live. Commands snapshot serialisable editor data, making
undo/redo predictable and letting ``Editor.save`` persist the view separately.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..engine.execute import RunReport
from ..engine.graph import Graph


@dataclass
class Viewport:
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0

    def clamp(self) -> None:
        self.zoom = min(2.5, max(0.35, float(self.zoom)))


@dataclass
class EditorTab:
    """A lightweight workflow tab record owned by the UI layer."""

    title: str = "Untitled"
    dirty: bool = False


@dataclass
class EditorCommand:
    label: str
    before: dict[str, Any]
    after: dict[str, Any]
    at: float = field(default_factory=time.monotonic)


@dataclass
class EditorState:
    graph: Graph
    selection: list[str] = field(default_factory=list)
    viewport: Viewport = field(default_factory=Viewport)
    run_report: RunReport | None = None
    errors: list[str] = field(default_factory=list)
    live: bool = True
    snap_to_grid: bool = True
    grid_size: float = 20.0
    allowed_node_types: set[str] | None = None
    drag_offsets: dict[str, tuple[float, float]] = field(default_factory=dict)
    tabs: list[EditorTab] = field(default_factory=lambda: [EditorTab()])
    active_tab: int = 0
    settings: dict[str, Any] = field(default_factory=lambda: {
        "modern_nodes": True,
        "show_pricing": False,
        "dev_mode": False,
        "token_weight_precision": 3,
        "show_errors": True,
    })
    console: list[str] = field(default_factory=list)
    is_running: bool = False
    working_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_error: str = ""
    run_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)
    _undo: list[EditorCommand] = field(default_factory=list, init=False, repr=False)
    _redo: list[EditorCommand] = field(default_factory=list, init=False, repr=False)
    _pending_live_at: float | None = field(default=None, init=False, repr=False)
    _tab_graphs: list[Graph] = field(default_factory=list, init=False, repr=False)
    _tab_views: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.selection = [nid for nid in self.selection if nid in self.graph.nodes]
        self.viewport.clamp()
        self._tab_graphs = [self.graph]
        self._tab_views = [self._current_view()]

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def view_dict(self) -> dict[str, Any]:
        """Return the persisted UI-only portion of an ``.ege.json`` file."""
        return {
            "viewport": asdict(self.viewport),
            "selection": list(self.selection),
            "live": self.live,
            "snap_to_grid": self.snap_to_grid,
            "grid_size": self.grid_size,
            "tabs": [asdict(tab) for tab in self.tabs],
            "active_tab": self.active_tab,
            "settings": copy.deepcopy(self.settings),
            "workflows": [graph.to_dict() for graph in self._tab_graphs],
            "tab_views": copy.deepcopy(self._tab_views[:self.active_tab] + [self._current_view()] + self._tab_views[self.active_tab + 1:]),
        }

    def _current_view(self) -> dict[str, Any]:
        return {"viewport": asdict(self.viewport), "selection": list(self.selection), "drag_offsets": copy.deepcopy(self.drag_offsets)}

    def new_tab(self, title: str = "Untitled") -> int:
        """Create an independent empty workflow and make it active."""
        self._tab_views[self.active_tab] = self._current_view()
        self._tab_graphs.append(Graph())
        self._tab_views.append({"viewport": asdict(Viewport()), "selection": [], "drag_offsets": {}})
        self.tabs.append(EditorTab(title=title))
        return self.switch_tab(len(self.tabs) - 1)

    def switch_tab(self, index: int) -> int:
        if not 0 <= index < len(self.tabs):
            raise IndexError("tab index out of range")
        self._tab_views[self.active_tab] = self._current_view()
        self.active_tab = index
        self.graph = self._tab_graphs[index]
        view = self._tab_views[index]
        self.viewport = Viewport(**view.get("viewport", {}))
        self.viewport.clamp()
        self.selection = [nid for nid in view.get("selection", []) if nid in self.graph.nodes]
        self.drag_offsets = dict(view.get("drag_offsets", {}))
        self.run_report = None
        self.log(f"tab: {self.tabs[index].title}")
        return index

    def close_tab(self, index: int | None = None) -> bool:
        if len(self.tabs) == 1:
            return False
        index = self.active_tab if index is None else index
        if not 0 <= index < len(self.tabs):
            return False
        del self.tabs[index], self._tab_graphs[index], self._tab_views[index]
        self.active_tab = min(self.active_tab, len(self.tabs) - 1)
        self.switch_tab(min(index, len(self.tabs) - 1))
        return True

    def desktop_settings_path(self) -> Path:
        return Path.home() / ".easygrapheditor.json"

    def load_desktop_settings(self, path: Path | None = None) -> None:
        try:
            saved = json.loads((path or self.desktop_settings_path()).read_text())
            if isinstance(saved, dict):
                self.settings.update(saved.get("settings", saved))
        except (OSError, ValueError, TypeError):
            return

    def save_desktop_settings(self, path: Path | None = None) -> Path:
        target = path or self.desktop_settings_path()
        target.write_text(json.dumps({"settings": self.settings}, indent=2))
        return target

    def load_view(self, view: dict[str, Any] | None) -> None:
        """Apply tolerant persisted view data from old and new graph files."""
        view = view or {}
        vp = view.get("viewport", view.get("view", {}))
        if isinstance(vp, dict):
            self.viewport = Viewport(**{k: vp[k] for k in ("x", "y", "zoom") if k in vp})
            self.viewport.clamp()
        self.selection = [nid for nid in view.get("selection", []) if nid in self.graph.nodes]
        self.live = bool(view.get("live", self.live))
        self.snap_to_grid = bool(view.get("snap_to_grid", self.snap_to_grid))
        self.grid_size = max(1.0, float(view.get("grid_size", self.grid_size)))
        raw_tabs = view.get("tabs")
        if isinstance(raw_tabs, list) and raw_tabs:
            self.tabs = [EditorTab(title=str(t.get("title", "Untitled")), dirty=bool(t.get("dirty", False)))
                         for t in raw_tabs if isinstance(t, dict)] or [EditorTab()]
        self.active_tab = min(max(0, int(view.get("active_tab", 0))), len(self.tabs) - 1)
        if isinstance(view.get("settings"), dict):
            self.settings.update(view["settings"])
        workflows = view.get("workflows")
        raw_views = view.get("tab_views")
        if isinstance(workflows, list) and workflows:
            self._tab_graphs = [Graph.from_dict(item) for item in workflows if isinstance(item, dict)]
            self.tabs = self.tabs[:len(self._tab_graphs)] or [EditorTab()]
            while len(self.tabs) < len(self._tab_graphs):
                self.tabs.append(EditorTab(f"Workflow {len(self.tabs) + 1}"))
            self._tab_views = list(raw_views) if isinstance(raw_views, list) else [self._current_view() for _ in self._tab_graphs]
            while len(self._tab_views) < len(self._tab_graphs):
                self._tab_views.append(self._current_view())
            self.active_tab = min(self.active_tab, len(self._tab_graphs) - 1)
            self.switch_tab(self.active_tab)

    def _snapshot(self) -> dict[str, Any]:
        return {"graph": self.graph.to_dict(), "view": self.view_dict(), "drag_offsets": copy.deepcopy(self.drag_offsets)}

    def _restore(self, snapshot: dict[str, Any]) -> None:
        restored = Graph.from_dict(snapshot["graph"])
        # Keep Graph identity because callers may already retain it.
        self.graph.nodes = restored.nodes
        self.graph.links = restored.links
        self.graph.loops = restored.loops
        self.graph.groups = restored.groups
        self.drag_offsets = dict(snapshot.get("drag_offsets", {}))
        view = copy.deepcopy(snapshot.get("view", {}))
        view.pop("workflows", None)
        view.pop("tab_views", None)
        self.load_view(view)
        self.run_report = None

    def apply(
        self,
        label: str,
        action: Callable[[], Any],
        *,
        debounce_live: bool = True,
        affects_execution: bool = True,
    ) -> Any:
        """Run one graph/view mutation and make it undoable."""
        before = self._snapshot()
        result = action()
        after = self._snapshot()
        if before != after:
            self._undo.append(EditorCommand(label, before, after))
            self._redo.clear()
            if self.tabs:
                self.tabs[self.active_tab].dirty = True
            if affects_execution:
                self.run_report = None
            self.log(label)
            if affects_execution and self.live and debounce_live:
                self.schedule_live_run()
        return result

    def undo(self) -> bool:
        if not self._undo:
            return False
        command = self._undo.pop()
        self._restore(command.before)
        self._redo.append(command)
        self.log(f"undo: {command.label}")
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        command = self._redo.pop()
        self._restore(command.after)
        self._undo.append(command)
        self.log(f"redo: {command.label}")
        return True

    def log(self, message: str) -> None:
        self.console.append(message)
        del self.console[:-200]

    def schedule_live_run(self, delay_ms: int = 500) -> None:
        self._pending_live_at = time.monotonic() + delay_ms / 1000.0

    def live_run_due(self, now: float | None = None) -> bool:
        if self._pending_live_at is None:
            return False
        if (now if now is not None else time.monotonic()) < self._pending_live_at:
            return False
        self._pending_live_at = None
        return self.live

    def move_nodes(self, node_ids: list[str], dx: float, dy: float) -> None:
        """Move graph nodes, snapping persisted positions when requested."""
        def move() -> None:
            for nid in node_ids:
                inst = self.graph.nodes.get(nid)
                if inst is None:
                    continue
                x, y = inst.pos
                nx, ny = x + dx, y + dy
                if self.snap_to_grid:
                    nx = round(nx / self.grid_size) * self.grid_size
                    ny = round(ny / self.grid_size) * self.grid_size
                inst.pos = (nx, ny)
        self.apply("move nodes", move, debounce_live=False, affects_execution=False)

    def snap_selection(self) -> None:
        self.move_nodes(list(self.selection), 0.0, 0.0)

    def toggle_collapsed(self, collapsed: bool | None = None) -> None:
        def change() -> None:
            for nid in self.selection:
                inst = self.graph.nodes.get(nid)
                if inst is not None:
                    inst.collapsed = (not inst.collapsed) if collapsed is None else collapsed
        self.apply("collapse nodes", change, debounce_live=False, affects_execution=False)

    def group_selection(self, title: str = "Group", color: str = "slate") -> Any:
        """Create a visual group through the same history boundary as canvas edits."""
        from .editing import group_nodes

        return self.apply("group nodes", lambda: group_nodes(self.graph, list(self.selection), title, color))

    def ungroup_selection(self) -> list[str]:
        """Dissolve every visual group touched by the current selection."""
        from .editing import ungroup

        names = {group.name for nid in self.selection for group in [self.graph.group_of(nid)] if group}
        if not names:
            return []
        freed: list[str] = []

        def dissolve() -> None:
            for name in names:
                freed.extend(ungroup(self.graph, name))

        self.apply("ungroup nodes", dissolve)
        return freed

    def set_param(self, node_id: str, key: str, value: Any) -> None:
        """Set a node parameter as an undoable edit."""
        def change() -> None:
            if node_id not in self.graph.nodes:
                raise ValueError(f"Unknown node: {node_id}")
            self.graph.nodes[node_id].params[key] = value

        self.apply(f"set {key}", change)

    def fit_view(self, width: float = 1000.0, height: float = 650.0) -> None:
        """Fit persisted node positions into a nominal canvas."""
        if not self.graph.nodes:
            self.viewport = Viewport()
            return
        xs = [node.pos[0] for node in self.graph.nodes.values()]
        ys = [node.pos[1] for node in self.graph.nodes.values()]
        span_x, span_y = max(xs) - min(xs) + 260.0, max(ys) - min(ys) + 160.0
        self.viewport.zoom = min(2.0, max(0.35, min(width / span_x, height / span_y)))
        self.viewport.x = -(min(xs) - 40.0) * self.viewport.zoom
        self.viewport.y = -(min(ys) - 40.0) * self.viewport.zoom
        self.viewport.clamp()
        self.log("fit canvas")
