"""UI adapters: backend-neutral visualisation over an executed graph.

Design (dependency injection): the caller owns the host object and hands it
to us. We never create windows, servers, or blocks ourselves here:

* gradio:      ``gradio_register_grapheditor(blocks_or_tab, source, ...)``
* streamlit:   ``streamlit_register_grapheditor(source, container=st_tab, ...)``
* pygame:      ``pygame_register_grapheditor(screen, source, ...)``

``source`` is a ``GraphAdapter`` (or anything it accepts: ``EditorState`` /
``Graph``). Standalone entry points that *do* own the loop/server live in
``gradio_app.py`` / ``streamlit_app.py`` / ``pygame_app.py`` and delegate here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..engine.cache import Cache
from ..engine.execute import CancellationToken, Executor, RunReport
from ..engine.graph import SUBWORKFLOW_TYPE_ID, Graph
from ..engine.nodes import canonical_kind, get_node
from ..engine.types import Field, Image
from .canvas import EditorState
from .inspector import inspect_node, inspect_types
from .library import categories, node_detail, search_nodes
from .web_canvas import editor_canvas_html, editor_canvas_js, streamlit_canvas_js
from .widgets import (
    adjust_param_value,
    cast_param_value,
    gradio_param_component,
    streamlit_param_widget,
)

try:
    from PIL import Image as _PILImage
except ImportError:  # pragma: no cover - pillow is a hard dep, this is just typing safety
    _PILImage = Any  # type: ignore[no-redef, misc]


# ------------------------------------------------------------------ payloads
def field_to_pil(f: Field) -> Any:
    """Height field (0..1) -> greyscale PIL image."""
    from PIL import Image as PILImage

    arr = (np.clip(f.data, 0, 1) * 255).astype(np.uint8)
    return PILImage.fromarray(arr, mode="L")


def image_to_pil(img: Image) -> Any:
    """Colour payload -> RGB PIL image."""
    from PIL import Image as PILImage

    arr = np.asarray(img.data)
    if arr.ndim == 2:
        return PILImage.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return PILImage.fromarray(arr.astype(np.uint8), mode="RGB")


def payload_to_pil(value: Any) -> Any | None:
    if isinstance(value, Image):
        return image_to_pil(value)
    if isinstance(value, Field):
        return field_to_pil(value)
    return None


def _value_preview(value: Any) -> str:
    """Keep inspector result values compact and safe to serialize."""
    if isinstance(value, (Image, Field)):
        shape = getattr(getattr(value, "data", None), "shape", None)
        return f"{type(value).__name__}{tuple(shape) if shape else ''}"
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


# ------------------------------------------------------------------- adapter
@dataclass
class AdapterOptions:
    title: str = "Graph Editor"
    show_images: bool = True
    show_table: bool = True
    inspector: bool = True  # param editors / loop controls / expand buttons
    editable: bool = True
    run_label: str = "▶ Run"


class GraphAdapter:
    """Backend-neutral view-model: owns execution + summarisation, not display."""

    def __init__(self, state: EditorState, cache: Cache | None = None) -> None:
        self.state = state
        self.cache = cache or Cache()
        self.executor = Executor(state.graph, cache=self.cache)
        self.report: RunReport | None = state.run_report
        self._force_next_run = False
        self._run_thread: threading.Thread | None = None
        self._cancel_token: CancellationToken | None = None
        self._run_lock = threading.RLock()

    @classmethod
    def from_any(cls, source: GraphAdapter | EditorState | Graph, **kw: Any) -> GraphAdapter:
        if isinstance(source, cls):
            return source
        if isinstance(source, EditorState):
            return cls(source, **kw)
        if isinstance(source, Graph):
            return cls(EditorState(graph=source), **kw)
        raise TypeError(f"Cannot adapt {type(source)!r}: pass GraphAdapter, EditorState or Graph.")

    # -- execution ---------------------------------------------------
    def run(self, *, force: bool = False) -> RunReport:
        force = force or self._force_next_run
        report = self.executor.run_blocking(dirty_only=not force)
        self._force_next_run = False
        self.report = report
        self.state.run_report = report
        self.state.log(f"run: {'ok' if report.ok() else 'errors'} ({report.ms:.1f} ms)")
        return report

    def _record_run_event(self, event: dict[str, Any]) -> None:
        """Receive executor stage events from its worker thread safely."""
        nid = event.get("node_id")
        if not isinstance(nid, str):
            return
        with self.state.run_lock:
            item = self.state.working_nodes.setdefault(nid, {"status": "working", "stages": []})
            item["status"] = "working"
            if event.get("event") == "stage":
                stage, frac = str(event.get("stage", "working")), float(event.get("frac", 0.0))
                item["stage"], item["frac"] = stage, frac
                item["stages"] = [*item.get("stages", []), (stage, frac)]

    def start_run(self, *, force: bool = False) -> bool:
        """Start one nonblocking execution and immediately expose working state."""
        with self._run_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                return False
            self._cancel_token = CancellationToken()
            executor = Executor(self.state.graph, cache=self.cache, on_event=self._record_run_event)
            self.executor = executor
            with self.state.run_lock:
                self.state.is_running = True
                self.state.run_error = ""
                self.state.working_nodes = {
                    nid: {"status": "queued", "stages": []} for nid in self.state.graph.nodes
                }
            def worker() -> None:
                try:
                    report = executor.run_blocking(dirty_only=not (force or self._force_next_run), cancel_token=self._cancel_token)
                    self._force_next_run = False
                    with self.state.run_lock:
                        self.report = report
                        self.state.run_report = report
                        self.state.working_nodes = {}
                except Exception as exc:  # noqa: BLE001 - report error into UI state
                    with self.state.run_lock:
                        self.state.run_error = f"{type(exc).__name__}: {exc}"
                        self.state.working_nodes = {}
                finally:
                    with self.state.run_lock:
                        self.state.is_running = False
                    self.state.log("run: done" if not self.state.run_error else f"run: {self.state.run_error}")
            self._run_thread = threading.Thread(target=worker, name="easygrapheditor-run", daemon=True)
            self._run_thread.start()
            return True

    def cancel_run(self) -> bool:
        with self._run_lock:
            if self._cancel_token is None or self._run_thread is None or not self._run_thread.is_alive():
                return False
            self._cancel_token.cancel()
            return True

    def clear_cache(self) -> None:
        """Forget cached node outputs (next run recomputes everything)."""
        self.cache.clear()
        self.executor.outputs.clear()
        # Existing releases do not expose invalidation publicly. Reset only
        # known optional executor state, and fall back to a full next run.
        fingerprints = getattr(self.executor, "_param_fingerprints", None)
        if isinstance(fingerprints, dict):
            fingerprints.clear()
        self._force_next_run = True
        self.state.log("cache cleared")

    def mark_dirty(self) -> None:
        """Invalidate the last report after a structural edit."""
        self.report = None
        self.state.run_report = None

    def mutate(self, label: str, action: Any, *, live: bool | None = None) -> Any:
        """Apply a backend action through the shared undo/history boundary."""
        self.state.live = self.state.live if live is None else bool(live)
        result = self.state.apply(label, action)
        self.mark_dirty()
        return result

    def consume_live_run(self) -> RunReport | None:
        """Run a pending debounced edit when the hosting event loop polls."""
        if self.state.live_run_due():
            self.start_run()
        return self.report

    def switch_tab(self, index: int) -> int:
        selected = self.state.switch_tab(index)
        self.executor = Executor(self.state.graph, cache=self.cache)
        self.mark_dirty()
        return selected

    def new_tab(self, title: str = "Untitled") -> int:
        selected = self.state.new_tab(title)
        self.executor = Executor(self.state.graph, cache=self.cache)
        self.mark_dirty()
        return selected

    def close_tab(self, index: int | None = None) -> bool:
        closed = self.state.close_tab(index)
        if closed:
            self.executor = Executor(self.state.graph, cache=self.cache)
            self.mark_dirty()
        return closed

    def dispatch_web_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply the shared browser action protocol and return fresh status."""
        return dispatch_web_action(self, action)

    def status(self) -> dict[str, Any]:
        """Dense status-bar facts for all frontends."""
        report = self.report
        hits = sum(1 for item in report.per_node.values() if item.cache_hit) if report else 0
        return {
            "zoom": round(self.state.viewport.zoom * 100),
            "nodes": len(self.state.graph.nodes),
            "links": len(self.state.graph.links),
            "selected": len(self.state.selection),
            "cache_hits": hits,
            "cache_total": len(report.per_node) if report else 0,
            "run_ms": round(report.ms, 1) if report else 0.0,
            "live": self.state.live,
            "is_running": self.state.is_running,
            "working": dict(self.state.working_nodes),
            "run_error": self.state.run_error,
        }

    @property
    def ok(self) -> bool:
        return self.report.ok() if self.report is not None else False

    # -- inspection --------------------------------------------------
    def loop_of(self, nid: str) -> str:
        """Loop name containing ``nid`` ("" when in no loop)."""
        for loop in self.state.graph.loops:
            if nid in loop.body:
                return loop.name
        return ""

    def node_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        groups_of = {nid: group.title for group in self.state.graph.groups for nid in group.nodes}
        for nid, inst in self.state.graph.nodes.items():
            ndef = get_node(inst.type_id)
            rep = self.report.per_node.get(nid) if self.report else None
            subflow = len(inst.params.get("nodes", [])) if inst.type_id == SUBWORKFLOW_TYPE_ID else 0
            working = self.state.working_nodes.get(nid, {})
            rows.append(
                {
                    "node": nid,
                    "title": ndef.title if ndef else inst.type_id,
                    "type": inst.type_id,
                    "status": working.get("status", rep.status if rep else "not run"),
                    "ms": round(rep.ms, 1) if rep else 0.0,
                    "cached": bool(rep and rep.cache_hit),
                    "error": rep.error if rep and rep.error else "",
                    "loop": self.loop_of(nid),
                    "subflow": subflow,
                    "group": groups_of.get(nid, ""),
                    "stage": working.get("stage", ""),
                    "frac": working.get("frac", 0.0),
                }
            )
        return rows

    def describe_node(self, nid: str) -> dict[str, Any]:
        """Hover/inspector detail: definition + live params + loop/subflow context."""
        inst = self.state.graph.nodes.get(nid)
        if inst is None:
            return {"error": f"Unknown node: {nid}"}
        detail = inspect_node(inst.type_id)
        ndef = get_node(inst.type_id)
        params = []
        for pdef in ndef.params if ndef else []:
            params.append(
                {"key": pdef.key, "label": pdef.label or pdef.key, "kind": pdef.kind,
                 "value": inst.params.get(pdef.key, pdef.default)}
            )
        detail["params"] = params
        detail["loop"] = self.loop_of(nid)
        group = self.state.graph.group_of(nid)
        detail["group"] = {"name": group.name, "title": group.title} if group else None
        detail["instance_params"] = dict(inst.params)
        if inst.type_id == SUBWORKFLOW_TYPE_ID:
            from ..engine.subworkflows import describe_subworkflow

            detail["subflow"] = describe_subworkflow(inst)
        return detail

    def inspector_data(self, nid: str | None = None) -> dict[str, Any]:
        """Return the selection-driven inspector model used by every host."""
        nid = nid or (self.state.selection[0] if self.state.selection else None)
        if not nid:
            return {"empty": "Select a node to inspect parameters, ports, results, and cache state."}
        detail = self.describe_node(nid)
        if "error" in detail:
            return detail
        report = self.report.per_node.get(nid) if self.report else None
        working = self.state.working_nodes.get(nid, {})
        incoming = [
            {"node": link.from_node, "port": link.from_port}
            for link in self.state.graph.links if link.to_node == nid
        ]
        outgoing = [
            {"node": link.to_node, "port": link.to_port}
            for link in self.state.graph.links if link.from_node == nid
        ]
        values = self.executor.outputs.get(nid, {})
        detail.update({
            "selected": nid,
            "status": working.get("status", report.status if report else "not run"),
            "timing_ms": round(report.ms, 3) if report else 0.0,
            "cache_hit": bool(report and report.cache_hit),
            "stages": working.get("stages", getattr(report, "stages", [] if report is None else None)),
            "stage": working.get("stage", ""),
            "frac": working.get("frac", 0.0),
            "error": report.error if report and report.error else "",
            "connections": {"incoming": incoming, "outgoing": outgoing},
            "results": {key: _value_preview(value) for key, value in values.items()},
            "previews": [key for key, value in values.items() if payload_to_pil(value) is not None],
        })
        return detail

    def expand_subworkflow(self, nid: str) -> list[str]:
        """Inline a subworkflow node; returns restored ids (re-runs nothing)."""
        from ..engine.subworkflows import expand_subworkflow_node

        restored = self.state.apply("expand subworkflow", lambda: expand_subworkflow_node(self.state.graph, nid))
        self.mark_dirty()
        self.state.selection = [r for r in restored[:1]]
        return restored

    def output_images(self) -> list[tuple[str, Any]]:
        """All Field/Image node outputs as (label, PIL image)."""
        out: list[tuple[str, Any]] = []
        for nid, port_map in self.executor.outputs.items():
            inst = self.state.graph.nodes.get(nid)
            ndef = get_node(inst.type_id) if inst else None
            label = ndef.title if ndef else nid
            for port, value in port_map.items():
                img = payload_to_pil(value)
                if img is not None:
                    out.append((f"{label} · {port}", img))
        return out

    def summarize(self, title: str = "Graph Editor") -> str:
        g = self.state.graph
        lines = [f"## {title}", f"{len(g.nodes)} nodes · {len(g.links)} links · {len(g.loops)} loops"]
        for loop in g.loops:
            lines.append(f"↻ loop `{loop.name}`: {len(loop.body)} nodes · max_iterations={loop.max_iterations}")
        if self.report is None:
            lines.append("_Not run yet. Press Run._")
            return "\n\n".join(lines)
        ok_n = sum(1 for r in self.report.per_node.values() if r.status in ("ok", "cached"))
        lines.append(f"Run: **{'ok' if self.report.ok() else 'errors'}** · {self.report.ms:.1f} ms · {ok_n}/{len(self.report.per_node)} ok")
        for row in self.node_rows():
            flag = "✓" if row["status"] in ("ok", "cached") else ("✗" if row["status"] == "error" else "·")
            extra = " (cached)" if row["cached"] else ""
            badges = ""
            if row["loop"]:
                badges += f" [loop:{row['loop']}]"
            if row["subflow"]:
                badges += f" [sub:{row['subflow']}]"
            err = f": {row['error']}" if row["error"] else ""
            lines.append(f"- {flag} `{row['title']}` ({row['node']}): {row['status']}{extra}{badges} · {row['ms']} ms{err}")
        return "\n".join(lines)


def dispatch_web_action(adapter: GraphAdapter, action: dict[str, Any]) -> dict[str, Any]:
    """Shared JSON protocol used by Gradio and Streamlit web canvases.

    Invalid browser payloads raise ``ValueError`` before touching graph state.
    Every mutating action goes through ``EditorState.apply`` or ``mutate``.
    """
    from .editing import add_node_at, connect, disconnect, group_nodes, remove_nodes, ungroup

    if not isinstance(action, dict) or not isinstance(action.get("type"), str):
        raise TypeError("web action must contain a string type")
    kind, data = action["type"], action.get("data", {})
    if not isinstance(data, dict):
        raise TypeError("web action data must be an object")
    state, graph = adapter.state, adapter.state.graph
    if kind == "select":
        selected = data.get("nodes", [data.get("node")])
        state.selection = [nid for nid in selected if isinstance(nid, str) and nid in graph.nodes]
    elif kind == "marquee":
        state.selection = [nid for nid in data.get("nodes", []) if isinstance(nid, str) and nid in graph.nodes]
    elif kind == "viewport":
        def set_view() -> None:
            for key in ("x", "y", "zoom"):
                if key in data:
                    setattr(state.viewport, key, float(data[key]))
            state.viewport.clamp()
        state.apply("set viewport", set_view, debounce_live=False, affects_execution=False)
    elif kind == "positions":
        positions = data.get("positions", {})
        if not isinstance(positions, dict):
            raise ValueError("positions must be an object")
        def move() -> None:
            for nid, pos in positions.items():
                if nid not in graph.nodes or not isinstance(pos, (list, tuple)) or len(pos) != 2:
                    continue
                x, y = float(pos[0]), float(pos[1])
                if state.snap_to_grid:
                    x, y = round(x / state.grid_size) * state.grid_size, round(y / state.grid_size) * state.grid_size
                graph.nodes[nid].pos = (x, y)
        state.apply("move nodes", move, debounce_live=False, affects_execution=False)
    elif kind == "connect":
        adapter.mutate("connect nodes", lambda: connect(graph, str(data["from_node"]), str(data["from_port"]), str(data["to_node"]), str(data["to_port"])))
    elif kind == "disconnect":
        adapter.mutate("disconnect nodes", lambda: disconnect(graph, str(data["from_node"]), str(data["from_port"]), str(data["to_node"]), str(data["to_port"])))
    elif kind == "add":
        type_id = str(data["type_id"])
        if state.allowed_node_types is not None and type_id not in state.allowed_node_types:
            raise ValueError(f"Node type {type_id!r} is outside this Editor's scoped nodes")
        adapter.mutate("add node", lambda: add_node_at(graph, type_id, tuple(data.get("pos", (0.0, 0.0)))))
    elif kind == "delete":
        ids = [nid for nid in data.get("nodes", state.selection) if nid in graph.nodes]
        adapter.mutate("delete nodes", lambda: remove_nodes(graph, ids))
        state.selection = []
    elif kind == "group":
        ids = [nid for nid in data.get("nodes", state.selection) if nid in graph.nodes]
        adapter.mutate("group nodes", lambda: group_nodes(graph, ids, str(data.get("title", "Group")), str(data.get("color", "slate"))))
    elif kind == "ungroup":
        names = data.get("names") or [group.name for nid in state.selection for group in [graph.group_of(nid)] if group]
        adapter.mutate("ungroup nodes", lambda: [ungroup(graph, str(name)) for name in names])
    elif kind == "collapse":
        state.toggle_collapsed(data.get("collapsed"))
    elif kind == "snap":
        state.snap_selection()
    elif kind == "fit":
        state.fit_view(float(data.get("width", 1000)), float(data.get("height", 650)))
    elif kind == "undo":
        state.undo()
        adapter.executor = Executor(graph, cache=adapter.cache)
    elif kind == "redo":
        state.redo()
        adapter.executor = Executor(graph, cache=adapter.cache)
    elif kind == "param":
        nid, key = str(data["node"]), str(data["key"])
        inst = graph.nodes.get(nid)
        definition = get_node(inst.type_id) if inst else None
        param = next((item for item in definition.params if item.key == key), None) if definition else None
        if param is None:
            raise ValueError(f"Unknown parameter {nid}.{key}")
        state.set_param(nid, key, cast_param_value(param, data.get("value")))
        adapter.mark_dirty()
    elif kind == "run":
        adapter.start_run(force=bool(data.get("force", False)))
    elif kind == "live":
        state.live = bool(data.get("value"))
    elif kind == "settings":
        values = data.get("settings", data)
        if not isinstance(values, dict):
            raise TypeError("settings must be an object")
        state.settings.update(values)
    elif kind == "cache":
        adapter.clear_cache()
        adapter.start_run(force=True)
    elif kind == "cancel":
        adapter.cancel_run()
    else:
        raise ValueError(f"unknown web action: {kind}")
    return adapter.status()


def decode_gradio_action(evt: Any) -> dict[str, Any] | None:
    """Extract a custom Gradio HTML event payload without trusting its shape."""
    raw = getattr(evt, "_data", evt)
    if isinstance(raw, dict) and isinstance(raw.get("value"), dict):
        raw = raw["value"]
    return raw if isinstance(raw, dict) else None


def handle_gradio_event(adapter: GraphAdapter, evt: Any) -> dict[str, Any]:
    """Dispatch Gradio's custom EventData payload and return current status."""
    action = decode_gradio_action(evt)
    if action is None:
        return adapter.status()
    try:
        return adapter.dispatch_web_action(action)
    except (KeyError, TypeError, ValueError) as exc:
        adapter.state.log(f"web action rejected: {exc}")
        adapter.state.errors.append(str(exc))
        return {**adapter.status(), "error": str(exc)}


# -------------------------------------------------------------------- layout
def layout_graph(graph: Graph) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Depth (column) per node from topo order + stable (x, y) unit positions.

    Returns (positions, depths) with x = depth, y = index within depth.
    Falls back to insertion order when the graph has a cycle.
    """
    # A graph created by the engine has default (0, 0) positions. Keep the
    # useful automatic layout until the editor has actually positioned a node.
    # Once it has, positions are the source of truth and survive save/load.
    positioned = any(inst.pos != (0.0, 0.0) for inst in graph.nodes.values())
    try:
        order = graph.topo_order()
    except ValueError:
        order = list(graph.nodes)
    depth: dict[str, int] = {nid: 0 for nid in graph.nodes}
    for nid in order:
        for link in graph.links:
            if link.from_node == nid and link.to_node in depth:
                depth[link.to_node] = max(depth[link.to_node], depth[nid] + 1)
    columns: dict[int, list[str]] = {}
    for nid in order:
        columns.setdefault(depth[nid], []).append(nid)
    if positioned:
        pos = {nid: (float(graph.nodes[nid].pos[0]), float(graph.nodes[nid].pos[1])) for nid in graph.nodes}
    else:
        pos = {nid: (float(depth[nid]), float(i)) for col, members in columns.items() for i, nid in enumerate(members)}
    return pos, depth


# ------------------------------------------------- gradio (host-provided)
def gradio_register_grapheditor(host: Any, source: GraphAdapter | EditorState | Graph, **opts: Any) -> dict[str, Any]:
    """Mount a LIVE node editor into a caller-provided Gradio container.

    ``host`` is your ``gr.Blocks`` / ``gr.Tab`` / ``gr.Row`` / ``gr.Column``.
    we add our components inside it (e.g. your submenu screen) and return
    component refs. The editor supports Run, Live auto-run, Clear cache,
    per-node param editing, add/delete nodes, and connect/disconnect links,
    with a canvas snapshot + output previews refreshed on every action.
    """
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError("Install the gradio extra: pip install 'easygrapheditor[gradio]'") from e

    from .editing import (
        GROUP_COLORS,
        add_node_at,
        compatible_inputs,
        connect,
        disconnect,
        group_nodes,
        link_labels,
        node_type_choices,
        port_value_preview,
        remove_node,
        render_canvas_image,
        ungroup,
    )

    o = AdapterOptions(**{k: v for k, v in opts.items() if k in AdapterOptions.__dataclass_fields__})
    adapter = GraphAdapter.from_any(source)
    types = node_type_choices(adapter.state.allowed_node_types)
    web_height = int(opts.get("height", 720))

    def _refresh() -> tuple[str, Any, str, dict[str, Any], list, list]:
        return (
            adapter.summarize(o.title),
            render_canvas_image(adapter.state.graph, adapter.report, adapter.executor.outputs),
            editor_canvas_html(adapter.state, height=web_height, include_script=False, editable=o.editable,
                               outputs=adapter.executor.outputs),
            adapter.status(),
            [img for _, img in adapter.output_images()],
            adapter.node_rows(),
        )

    def _refresh_full() -> tuple:
        """_refresh + fresh node/link picker choices (structure may have changed)."""
        s, c, w, st, g, t = _refresh()
        ids = sorted(adapter.state.graph.nodes)
        links = link_labels(adapter.state.graph)
        return (s, c, w, st, g, t, gr.Dropdown(choices=ids), gr.Dropdown(choices=ids),
                gr.Dropdown(choices=links), gr.Dropdown(choices=ids), adapter.inspector_data())

    with host:
        gr.Markdown(f"## {o.title}", visible=False)
        web_canvas = gr.HTML(
            value=editor_canvas_html(adapter.state, height=web_height, include_script=False, editable=o.editable,
                                     outputs=adapter.executor.outputs),
            js_on_load=editor_canvas_js(adapter.state, editable=o.editable, outputs=adapter.executor.outputs),
            server_functions=[lambda action: action],
            label="Interactive canvas",
        )
        summary = gr.Markdown(adapter.summarize(o.title), visible=False)
        with gr.Row(visible=False):
            run_btn = gr.Button(o.run_label, variant="primary")
            live_box = gr.Checkbox(value=adapter.state.live, label="Live (auto-run on change)", visible=o.editable)
            clear_btn = gr.Button("Clear cache + rerun")
            undo_btn = gr.Button("Undo", size="sm", visible=o.editable)
            redo_btn = gr.Button("Redo", size="sm", visible=o.editable)
            collapse_btn = gr.Button("Collapse selected", size="sm", visible=o.editable)
            snap_btn = gr.Button("Snap selected", size="sm", visible=o.editable)
            fit_btn = gr.Button("Fit", size="sm")
        with gr.Row(visible=False):
            tab_select = gr.Dropdown(
                choices=[(tab.title, str(index)) for index, tab in enumerate(adapter.state.tabs)],
                value=str(adapter.state.active_tab), label="Graph tab", scale=3,
            )
            tab_new = gr.Button("New tab", size="sm", visible=o.editable)
            tab_close = gr.Button("Close tab", size="sm", visible=o.editable)
        status = gr.JSON(value=adapter.status(), label="Status", visible=False)
        canvas = gr.Image(value=render_canvas_image(adapter.state.graph, adapter.report, adapter.executor.outputs), label="Canvas", visible=False)
        gallery = gr.Gallery(value=[img for _, img in adapter.output_images()], label="Previews", visible=False)
        table = gr.JSON(value=adapter.node_rows(), label="Nodes", visible=False)
        notice = gr.Markdown("", visible=False)
        with gr.Row(visible=False):
            library_panel = gr.JSON(value={"categories": categories(), "results": [node_detail(n.type_id) for n in search_nodes(allowed=adapter.state.allowed_node_types)]}, label="Library")
            inspector_panel = gr.JSON(value=adapter.inspector_data(), label="Inspector")
            types_panel = gr.JSON(value=inspect_types(), label="Types")
        with gr.Accordion("Settings", open=False, visible=False):
            modern = gr.Checkbox(value=bool(adapter.state.settings.get("modern_nodes", True)), label="Modern node design")
            pricing = gr.Checkbox(value=bool(adapter.state.settings.get("show_pricing", False)), label="Show pricing")
            dev_mode = gr.Checkbox(value=bool(adapter.state.settings.get("dev_mode", False)), label="Developer mode")
            precision = gr.Slider(0, 8, value=int(adapter.state.settings.get("token_weight_precision", 3)), step=1, label="Token weight precision")
            errors_toggle = gr.Checkbox(value=bool(adapter.state.settings.get("show_errors", True)), label="Show errors")
            settings_btn = gr.Button("Apply settings", size="sm")

            def _apply_settings(a: bool, p: bool, d: bool, precision_value: float, b: bool) -> tuple:
                adapter.dispatch_web_action({"type": "settings", "data": {"modern_nodes": a, "show_pricing": p, "dev_mode": d, "token_weight_precision": int(precision_value), "show_errors": b}})
                return _refresh_full()


        def _refresh_ids() -> list[str]:
            return sorted(adapter.state.graph.nodes)

        # -- handlers (wired to buttons after all components exist) --------
        def _on_run(_live: bool = False) -> tuple:
            adapter.state.live = bool(_live)
            adapter.start_run()
            return _refresh_full()

        def _maybe_live(live: bool) -> tuple:
            adapter.state.live = bool(live)
            return _refresh_full()

        def _on_clear() -> tuple:
            adapter.clear_cache()
            adapter.start_run(force=True)
            return _refresh_full()

        def _on_undo(live: bool) -> tuple:
            adapter.state.undo()
            adapter.executor = Executor(adapter.state.graph, cache=adapter.cache)
            return _maybe_live(live)

        def _on_redo(live: bool) -> tuple:
            adapter.state.redo()
            adapter.executor = Executor(adapter.state.graph, cache=adapter.cache)
            return _maybe_live(live)

        def _on_collapse(live: bool) -> tuple:
            adapter.state.toggle_collapsed()
            return _maybe_live(live)

        def _on_snap(live: bool) -> tuple:
            adapter.state.snap_selection()
            return _maybe_live(live)

        def _on_fit() -> tuple:
            adapter.state.fit_view()
            return _refresh_full()

        def _tab_update() -> Any:
            return gr.Dropdown(
                choices=[(tab.title, str(index)) for index, tab in enumerate(adapter.state.tabs)],
                value=str(adapter.state.active_tab),
            )

        def _on_tab_switch(index: str | None) -> tuple:
            if index is not None:
                adapter.switch_tab(int(index))
            return (*_refresh_full(), _tab_update())

        def _on_tab_new() -> tuple:
            adapter.new_tab(f"Graph {len(adapter.state.tabs) + 1}")
            return (*_refresh_full(), _tab_update())

        def _on_tab_close() -> tuple:
            adapter.close_tab()
            return (*_refresh_full(), _tab_update())

        with gr.Accordion("Library: add node", open=False, visible=False):
            type_drop = gr.Dropdown(choices=[label for _, label in types], label="Node type")
            add_btn = gr.Button("Add node", size="sm")

            def _on_add(label: str | None, live: bool) -> tuple:
                if not label:
                    s = _refresh_full()
                    return (*s, gr.Markdown("Pick a node type first.", visible=True))
                type_id = next(t for t, lab in types if lab == label)
                inst = adapter.mutate("add node", lambda: add_node_at(adapter.state.graph, type_id, pos=(0.0, 0.0)), live=live)
                s = _maybe_live(live)
                note = f"Added `{inst.id}`. Reload the UI to refresh the inspector."
                return (*s, gr.Markdown(note, visible=True))

        with gr.Accordion("Links: connect / disconnect", open=False, visible=False):
            from_node = gr.Dropdown(choices=_refresh_ids(), label="From node")
            from_port = gr.Dropdown(choices=[], label="From port")
            to_node = gr.Dropdown(choices=_refresh_ids(), label="To node")
            to_port = gr.Dropdown(choices=[], label="To port (compatible first)")
            link_btn = gr.Button("Connect", size="sm")
            unlink_drop = gr.Dropdown(choices=link_labels(adapter.state.graph), label="Existing link")
            unlink_btn = gr.Button("Disconnect", size="sm")

            def _ports(nid: str | None, direction: str) -> list[str]:
                inst = adapter.state.graph.nodes.get(nid or "")
                if inst is None:
                    return []
                if inst.type_id == SUBWORKFLOW_TYPE_ID:
                    maps = inst.params.get("inputs" if direction == "in" else "outputs", [])
                    return [m["key"] for m in maps]
                ndef = get_node(inst.type_id)
                if ndef is None:
                    return []
                return [p.key for p in (ndef.inputs if direction == "in" else ndef.outputs)]

            def _on_from_node(nid: str | None) -> Any:
                return gr.Dropdown(choices=_ports(nid, "out"))

            def _on_to_node(nid: str | None, from_nid: str | None, from_pt: str | None) -> Any:
                ports = _ports(nid, "in")
                if from_nid and from_pt:
                    compat = [p for _, p in compatible_inputs(adapter.state.graph, from_nid, from_pt)]
                    ports = sorted(set(ports), key=lambda p: (0 if p in compat else 1, p))
                return gr.Dropdown(choices=ports)

            from_node.change(_on_from_node, inputs=[from_node], outputs=[from_port])
            to_node.change(_on_to_node, inputs=[to_node, from_node, from_port], outputs=[to_port])

            def _on_link(a: str | None, ap: str | None, b: str | None, bp: str | None, live: bool) -> tuple:
                try:
                    link = adapter.mutate("connect nodes", lambda: connect(adapter.state.graph, a or "", ap or "", b or "", bp or ""), live=live)
                    s = _maybe_live(live)
                    note = f"Connected `{link.from_node}.{link.from_port} → {link.to_node}.{link.to_port}`."
                except ValueError as e:
                    s = _refresh_full()
                    note = f"Cannot connect: {e}"
                return (*s, gr.Markdown(note, visible=True))

            def _on_unlink(label: str | None, live: bool) -> tuple:
                if label:
                    for link in adapter.state.graph.links:
                        if f"{link.from_node}.{link.from_port} -> {link.to_node}.{link.to_port}" == label:
                            adapter.mutate("disconnect nodes", lambda link=link: disconnect(adapter.state.graph, link.from_node, link.from_port, link.to_node, link.to_port), live=live)
                            break
                return (*_maybe_live(live), gr.Markdown("", visible=False))

        with gr.Accordion("Delete node", open=False, visible=False):
            del_drop = gr.Dropdown(choices=_refresh_ids(), label="Node")
            del_btn = gr.Button("Delete node + incident links", size="sm")

            def _on_delete(nid: str | None, live: bool) -> tuple:
                if nid and nid in adapter.state.graph.nodes:
                    adapter.mutate("delete node", lambda: remove_node(adapter.state.graph, nid), live=live)
                return (*_maybe_live(live), gr.Markdown("", visible=False))

        # -- wire buttons (every action refreshes views + picker choices) --
        outs_full = [summary, canvas, web_canvas, status, gallery, table, from_node, to_node, unlink_drop, del_drop, inspector_panel]
        outs_note = [*outs_full, notice]
        settings_btn.click(_apply_settings, inputs=[modern, pricing, dev_mode, precision, errors_toggle], outputs=outs_full)
        run_btn.click(_on_run, inputs=[live_box], outputs=outs_full)
        clear_btn.click(_on_clear, outputs=outs_full)
        undo_btn.click(_on_undo, inputs=[live_box], outputs=outs_full)
        redo_btn.click(_on_redo, inputs=[live_box], outputs=outs_full)
        collapse_btn.click(_on_collapse, inputs=[live_box], outputs=outs_full)
        snap_btn.click(_on_snap, inputs=[live_box], outputs=outs_full)
        fit_btn.click(_on_fit, outputs=outs_full)

        def _on_web_action(_html_value: Any, evt: gr.EventData) -> tuple:
            """Use Gradio's EventData custom payload, never the HTML value."""
            result = handle_gradio_event(adapter, evt)
            message = result.get("error", "")
            refreshed = list(_refresh_full())
            refreshed[2] = gr.skip()
            return (*refreshed, gr.Markdown(message, visible=bool(message)))

        _on_web_action.__annotations__["evt"] = gr.EventData
        web_canvas.click(_on_web_action, inputs=[web_canvas], outputs=[*outs_full, notice])
        tab_select.change(_on_tab_switch, inputs=[tab_select], outputs=[*outs_full, tab_select])
        tab_new.click(_on_tab_new, outputs=[*outs_full, tab_select])
        tab_close.click(_on_tab_close, outputs=[*outs_full, tab_select])
        live_timer = gr.Timer(0.5)

        def _on_live_tick() -> tuple:
            adapter.consume_live_run()
            summary_value, canvas_value, _web_value, status_value, gallery_value, table_value = _refresh()
            return (summary_value, canvas_value, status_value, gallery_value, table_value,
                    adapter.inspector_data())

        live_timer.tick(_on_live_tick, outputs=[summary, canvas, status, gallery, table, inspector_panel])
        add_btn.click(_on_add, inputs=[type_drop, live_box], outputs=outs_note)
        link_btn.click(_on_link, inputs=[from_node, from_port, to_node, to_port, live_box], outputs=outs_note)
        unlink_btn.click(_on_unlink, inputs=[unlink_drop, live_box], outputs=outs_note)
        del_btn.click(_on_delete, inputs=[del_drop, live_box], outputs=outs_note)

        if o.inspector and o.editable:
            with gr.Accordion("Inspector: node params (Apply writes back)", open=False, visible=False):
                for nid, inst in list(adapter.state.graph.nodes.items()):
                    ndef = get_node(inst.type_id)
                    if ndef is None:
                        continue
                    with gr.Accordion(f"{ndef.title} (`{nid}`)", open=False):
                        comps = [gradio_param_component(gr, p, inst.params.get(p.key, p.default)) for p in ndef.params]

                        def _on_apply(*values: Any, _nid: str = nid, _ndef: Any = ndef) -> tuple:
                            live = bool(values[-1])
                            for pdef, raw in zip(_ndef.params, values[:-1]):
                                adapter.state.set_param(_nid, pdef.key, cast_param_value(pdef, raw))
                            adapter.mark_dirty()
                            return _maybe_live(live)

                        gr.Button("Apply params", size="sm").click(_on_apply, inputs=[*comps, live_box], outputs=outs_full)
                        if inst.type_id == SUBWORKFLOW_TYPE_ID:
                            expand_btn = gr.Button("Expand subworkflow inline", size="sm")

                            def _on_expand(_x: Any = None, _nid: str = nid) -> tuple:
                                restored = adapter.expand_subworkflow(_nid)
                                s = _refresh_full()
                                note = f"Expanded into {len(restored)} nodes. Reload the UI to refresh the inspector."
                                return (*s, gr.Markdown(note, visible=True))

                            expand_btn.click(_on_expand, outputs=[*outs_full, notice])

            if adapter.state.graph.loops:
                with gr.Accordion("Loops: max_iterations (Apply writes back)", open=False, visible=False):
                    for i, loop in enumerate(adapter.state.graph.loops):
                        max_box = gr.Number(value=loop.max_iterations, label=f"{loop.name} · max_iterations", precision=0)

                        def _on_loop_apply(v: float, live: bool, _i: int = i) -> tuple:
                            adapter.mutate("set loop cap", lambda: setattr(adapter.state.graph.loops[_i], "max_iterations", max(1, int(v))), live=live)
                            return _maybe_live(live)

                        gr.Button(f"Apply {loop.name}", size="sm").click(_on_loop_apply, inputs=[max_box, live_box], outputs=outs_full)

        with gr.Accordion("Ports: live values", open=False, visible=False):
            port_node = gr.Dropdown(choices=sorted(adapter.state.graph.nodes), label="Node")
            ports_md = gr.Markdown("Pick a node to inspect its ports.")
            ports_gallery = gr.Gallery(value=[], label="Port previews")

            def _on_ports(nid: str | None) -> tuple[str, list, dict[str, Any]]:
                if not nid or nid not in adapter.state.graph.nodes:
                    return "Pick a node to inspect its ports.", [], adapter.inspector_data(None)
                detail = adapter.describe_node(nid)
                lines = [f"**{detail.get('title', nid)}** (`{nid}`)"]
                for side in ("inputs", "outputs"):
                    lines.append(f"_{side.capitalize()}_")
                    for key, dtype in detail.get(side, []):
                        preview = ""
                        if side == "outputs":
                            preview = f" = {port_value_preview(adapter.executor.outputs.get(nid, {}).get(key))}"
                        lines.append(f"- `{key}` [{dtype}]{preview}")
                images = []
                for _key, img in _port_images(adapter, nid):
                    images.append(img)
                return "\n".join(lines), images, adapter.inspector_data(nid)

            port_node.change(_on_ports, inputs=[port_node], outputs=[ports_md, ports_gallery, inspector_panel])

        with gr.Accordion("Groups: visual multi-node boxes", open=False, visible=False):
            group_multi = gr.Dropdown(choices=sorted(adapter.state.graph.nodes), multiselect=True, label="Nodes")
            group_title = gr.Textbox(value="", label="Title")
            group_color = gr.Dropdown(choices=sorted(GROUP_COLORS), value="slate", label="Color")
            group_btn = gr.Button("Group selection", size="sm")
            ungroup_drop = gr.Dropdown(choices=[g.name for g in adapter.state.graph.groups], label="Group")
            ungroup_btn = gr.Button("Ungroup", size="sm")

            def _on_group(nids: list[str] | None, title: str, color: str, live: bool) -> tuple:
                try:
                    grp = adapter.mutate("group nodes", lambda: group_nodes(adapter.state.graph, list(nids or []), title or "Group", color), live=live)
                    s = _maybe_live(live)
                    return (*s, gr.Markdown(f"Grouped `{grp.name}` ({len(grp.nodes)} nodes).", visible=True))
                except ValueError as e:
                    return (*_refresh_full(), gr.Markdown(f"Cannot group: {e}", visible=True))

            def _on_ungroup(name: str | None, live: bool) -> tuple:
                if name:
                    adapter.mutate("ungroup nodes", lambda: ungroup(adapter.state.graph, name), live=live)
                return (*_maybe_live(live), gr.Markdown("", visible=False))

            group_btn.click(_on_group, inputs=[group_multi, group_title, group_color, live_box], outputs=[*outs_full, notice])
            ungroup_btn.click(_on_ungroup, inputs=[ungroup_drop, live_box], outputs=[*outs_full, notice])
    return {"summary": summary, "canvas": canvas, "web_canvas": web_canvas, "status": status,
            "gallery": gallery, "table": table, "run": run_btn, "live": live_box, "clear": clear_btn,
            "undo": undo_btn, "redo": redo_btn, "collapse": collapse_btn, "snap": snap_btn,
            "fit": fit_btn, "library": library_panel, "inspector": inspector_panel,
            "types": types_panel, "tabs": tab_select, "new_tab": tab_new, "close_tab": tab_close,
            "settings": settings_btn, "notice": notice, "adapter": adapter}


def _port_images(adapter: GraphAdapter, nid: str) -> list[tuple[str, Any]]:
    """(port, PIL) previews for a node's Field/Image outputs."""
    out = []
    for port, value in adapter.executor.outputs.get(nid, {}).items():
        img = payload_to_pil(value)
        if img is not None:
            out.append((port, img))
    return out


# ----------------------------------------------- streamlit (host-provided)
class StreamlitContainer(Protocol):
    def markdown(self, body: str, *a: Any, **k: Any) -> Any: ...
    def image(self, image: Any, *a: Any, **k: Any) -> Any: ...
    def json(self, data: Any, *a: Any, **k: Any) -> Any: ...
    def button(self, label: str, *a: Any, **k: Any) -> bool: ...
    def expander(self, label: str, *a: Any, **k: Any) -> Any: ...
    def selectbox(self, label: str, options: Any, *a: Any, **k: Any) -> Any: ...
    def checkbox(self, label: str, *a: Any, **k: Any) -> bool: ...
    def number_input(self, label: str, *a: Any, **k: Any) -> Any: ...


def streamlit_register_grapheditor(
    source: GraphAdapter | EditorState | Graph, container: StreamlitContainer | None = None, **opts: Any
) -> GraphAdapter:
    """Render a LIVE node editor into a caller-provided Streamlit container.

    ``container`` is your tab/sidebar/expander (``st.tabs(...)[0]``,
    ``st.sidebar``, ``st.expander(...)`` …); defaults to the ``streamlit``
    module itself, i.e. the current page. Supports Run, Live auto-run,
    Clear cache, per-node param editing, add/delete nodes, and
    connect/disconnect links, with a canvas snapshot + previews.
    Pass ``session_key="..."`` to persist the adapter (graph, cache,
    selection) across reruns. Returns the adapter.
    """
    if container is None:
        try:
            import streamlit as st
        except ImportError as e:
            raise ImportError("Install the streamlit extra: pip install 'easygrapheditor[streamlit]'") from e
        container = st
    from .editing import (
        GROUP_COLORS,
        add_node_at,
        compatible_inputs,
        connect,
        disconnect,
        group_nodes,
        link_labels,
        node_type_choices,
        port_value_preview,
        remove_node,
        render_canvas_image,
        ungroup,
    )

    session_key = opts.get("session_key")
    o = AdapterOptions(**{k: v for k, v in opts.items() if k in AdapterOptions.__dataclass_fields__})
    adapter = GraphAdapter.from_any(source)
    if session_key:
        try:
            import streamlit as st

            adapter = st.session_state.get(session_key) or adapter
            st.session_state[session_key] = adapter
        except Exception:  # noqa: BLE001 - headless/fake runs have no session state
            adapter.state.errors.append("session persistence unavailable headless")

    def _rerun() -> bool:
        try:
            import streamlit as st

            st.rerun()
            return True
        except Exception:  # noqa: BLE001 - headless/fake containers
            return False

    def _mount_interactive_canvas() -> bool:
        """Mount the primary editor before secondary host controls."""
        try:
            import streamlit.components.v2 as components

            bridge = components.component(
                f"easygrapheditor_canvas_{id(adapter.state):x}",
                html=editor_canvas_html(adapter.state, include_script=False, editable=o.editable,
                                        outputs=adapter.executor.outputs),
                js=streamlit_canvas_js(adapter.state, editable=o.editable, outputs=adapter.executor.outputs),
            )

            def on_action(action: dict[str, Any]) -> None:
                try:
                    adapter.dispatch_web_action(action)
                except (KeyError, TypeError, ValueError) as exc:
                    adapter.state.log(f"web action rejected: {exc}")
                    adapter.state.errors.append(str(exc))

            bridge(key=f"ege-canvas-{id(adapter.state)}", data=adapter.status(), on_action=on_action)
            return True
        except Exception as exc:  # noqa: BLE001 - embedding and test environments can omit components
            adapter.state.log(f"web canvas unavailable: {type(exc).__name__}")
            return False

    interactive_canvas = _mount_interactive_canvas()
    if not interactive_canvas:
        container.image(render_canvas_image(adapter.state.graph, adapter.report, adapter.executor.outputs), caption="Canvas")

    live = adapter.state.live
    if container.button(o.run_label):
        adapter.start_run()
    tab_labels = [tab.title for tab in adapter.state.tabs]
    active_label = tab_labels[adapter.state.active_tab]
    chosen_tab = container.selectbox("Graph tab", tab_labels, index=adapter.state.active_tab, key="ege:tab")
    if chosen_tab != active_label:
        adapter.switch_tab(tab_labels.index(chosen_tab))
        _rerun()
    if o.editable and container.button("New graph tab", key="ege:tab:new"):
        adapter.new_tab(f"Graph {len(adapter.state.tabs) + 1}")
        _rerun()
    if o.editable and container.button("Close graph tab", key="ege:tab:close"):
        adapter.close_tab()
        _rerun()
    if o.editable:
        with container.expander("Toolbar: undo, view, live mode + cache", expanded=False):
            live = bool(container.checkbox("Live (auto-run on change)", value=adapter.state.live, key="ege:live"))
            adapter.state.live = live
            if container.button("Undo", key="ege:undo"):
                adapter.state.undo()
                adapter.executor = Executor(adapter.state.graph, cache=adapter.cache)
            if container.button("Redo", key="ege:redo"):
                adapter.state.redo()
                adapter.executor = Executor(adapter.state.graph, cache=adapter.cache)
            if container.button("Collapse selected", key="ege:collapse"):
                adapter.state.toggle_collapsed()
            if container.button("Snap selected", key="ege:snap"):
                adapter.state.snap_selection()
            if container.button("Fit canvas", key="ege:fit"):
                adapter.state.fit_view()
            if container.button("Clear cache + rerun", key="ege:clear"):
                adapter.clear_cache()
                adapter.start_run(force=True)
    else:
        live = False
    with container.expander("Settings", expanded=False):
        modern = bool(container.checkbox("Modern node design", value=bool(adapter.state.settings.get("modern_nodes", True)), key="ege:settings:modern"))
        pricing = bool(container.checkbox("Show pricing", value=bool(adapter.state.settings.get("show_pricing", False)), key="ege:settings:pricing"))
        dev_mode = bool(container.checkbox("Developer mode", value=bool(adapter.state.settings.get("dev_mode", False)), key="ege:settings:dev"))
        precision = int(container.number_input("Token weight precision", value=int(adapter.state.settings.get("token_weight_precision", 3)), min_value=0, max_value=8, step=1, key="ege:settings:precision"))
        show_errors = bool(container.checkbox("Show errors", value=bool(adapter.state.settings.get("show_errors", True)), key="ege:settings:errors"))
        if container.button("Apply settings", key="ege:settings:apply"):
            adapter.dispatch_web_action({"type": "settings", "data": {"modern_nodes": modern, "show_pricing": pricing, "dev_mode": dev_mode, "token_weight_precision": precision, "show_errors": show_errors}})
            _rerun()
    if o.inspector and o.editable:
        with container.expander("Library: add node", expanded=False):
            labels = [label for _, label in node_type_choices(adapter.state.allowed_node_types)]
            picked = container.selectbox("Node type", labels, key="ege:add:type")
            if container.button("Add node", key="ege:add:go") and picked:
                lookup = {label: t for t, label in node_type_choices(adapter.state.allowed_node_types)}
                inst = adapter.mutate("add node", lambda: add_node_at(adapter.state.graph, lookup[picked], pos=(0.0, 0.0)), live=live)
                container.markdown(f"Added `{inst.id}`.")
                _rerun()
        with container.expander("Links: connect / disconnect", expanded=False):
            ids = sorted(adapter.state.graph.nodes)
            from_nid = container.selectbox("From node", ids, key="ege:link:from") if ids else None
            from_ports = _st_ports(adapter, from_nid, "out")
            from_pt = container.selectbox("From port", from_ports, key="ege:link:from-port") if from_ports else None
            to_nid = container.selectbox("To node", ids, key="ege:link:to") if ids else None
            to_ports = _st_ports(adapter, to_nid, "in")
            if from_nid and from_pt:
                compat = {p for _, p in compatible_inputs(adapter.state.graph, from_nid, from_pt)}
                to_ports = sorted(set(to_ports), key=lambda p: (0 if p in compat else 1, p))
            to_pt = container.selectbox("To port (compatible first)", to_ports, key="ege:link:to-port") if to_ports else None
            if container.button("Connect", key="ege:link:go"):
                try:
                    link = adapter.mutate("connect nodes", lambda: connect(adapter.state.graph, from_nid or "", from_pt or "", to_nid or "", to_pt or ""), live=live)
                    container.markdown(f"Connected `{link.from_node}.{link.from_port} → {link.to_node}.{link.to_port}`.")
                except ValueError as e:
                    container.markdown(f"Cannot connect: {e}")
                _rerun()
            existing = link_labels(adapter.state.graph)
            doomed = container.selectbox("Existing link", existing, key="ege:unlink") if existing else None
            if existing and container.button("Disconnect", key="ege:unlink:go") and doomed:
                for link in adapter.state.graph.links:
                    if f"{link.from_node}.{link.from_port} -> {link.to_node}.{link.to_port}" == doomed:
                        adapter.mutate("disconnect nodes", lambda link=link: disconnect(adapter.state.graph, link.from_node, link.from_port, link.to_node, link.to_port), live=live)
                        break
                _rerun()
        with container.expander("Delete node", expanded=False):
            doomed = container.selectbox("Node", ids, key="ege:delete") if ids else None
            if ids and container.button("Delete node + incident links", key="ege:delete:go") and doomed:
                adapter.mutate("delete node", lambda: remove_node(adapter.state.graph, doomed), live=live)
                _rerun()
        with container.expander("Ports: live values", expanded=False):
            inspected = container.selectbox("Node", ids, key="ege:ports:node") if ids else None
            if inspected:
                detail = adapter.describe_node(inspected)
                container.markdown(f"**{detail.get('title', inspected)}** (`{inspected}`)")
                for side in ("inputs", "outputs"):
                    container.markdown(f"_{side.capitalize()}_")
                    for key, dtype in detail.get(side, []):
                        preview = ""
                        if side == "outputs":
                            preview = f" = {port_value_preview(adapter.executor.outputs.get(inspected, {}).get(key))}"
                        container.markdown(f"- `{key}` [{dtype}]{preview}")
                for _port, img in _port_images(adapter, inspected):
                    container.image(img, caption=f"{inspected} · {_port}")
        with container.expander("Inspector: selected node", expanded=False):
            container.json(adapter.inspector_data())
        with container.expander("Groups: visual multi-node boxes", expanded=False):
            picked = container.multiselect("Nodes", ids, key="ege:group:nodes") if ids else []
            gtitle = container.text_input("Title", value="", key="ege:group:title")
            gcolor = container.selectbox("Color", sorted(GROUP_COLORS), key="ege:group:color")
            if container.button("Group selection", key="ege:group:go"):
                try:
                    grp = adapter.mutate("group nodes", lambda: group_nodes(adapter.state.graph, list(picked or []), gtitle or "Group", gcolor), live=live)
                    container.markdown(f"Grouped `{grp.name}` ({len(grp.nodes)} nodes).")
                except ValueError as e:
                    container.markdown(f"Cannot group: {e}")
                _rerun()
            existing_groups = [(g.name, g.title, len(g.nodes)) for g in adapter.state.graph.groups]
            if existing_groups:
                container.json([{"name": n, "title": t, "nodes": c} for n, t, c in existing_groups])
                doomed_g = container.selectbox("Group", [n for n, _, _ in existing_groups], key="ege:ungroup")
                if container.button("Ungroup", key="ege:ungroup:go") and doomed_g:
                    adapter.mutate("ungroup nodes", lambda: ungroup(adapter.state.graph, doomed_g), live=live)
                    _rerun()
        with container.expander("Inspector: node params", expanded=False):
            for nid, inst in list(adapter.state.graph.nodes.items()):
                ndef = get_node(inst.type_id)
                if ndef is None:
                    continue
                detail = adapter.describe_node(nid)
                with container.expander(f"{ndef.title} (`{nid}`)", expanded=False):
                    if detail.get("loop"):
                        container.markdown(f"_Loop: `{detail['loop']}`_")
                    if detail.get("subflow"):
                        container.markdown(f"_Subworkflow: {detail['subflow']['nodes']} nodes: {', '.join(detail['subflow']['titles'][:8])}_")
                        if container.button("Expand subworkflow inline", key=f"ege:expand:{nid}"):
                            adapter.expand_subworkflow(nid)
                            _rerun()
                    for pdef in ndef.params:
                        key = f"ege:{nid}:{pdef.key}"
                        current = inst.params.get(pdef.key, pdef.default)
                        new_val = streamlit_param_widget(container, pdef, current, key)
                        casted = cast_param_value(pdef, new_val)
                        if casted != current:
                            adapter.state.set_param(nid, pdef.key, casted)
                            adapter.mark_dirty()
        if adapter.state.graph.loops:
            with container.expander("Loops: max_iterations", expanded=False):
                for loop in adapter.state.graph.loops:
                    loop_value = max(1, int(container.number_input(f"{loop.name} · max_iterations", value=int(loop.max_iterations), step=1, key=f"ege:loop:{loop.name}")))
                    if loop_value != loop.max_iterations:
                        adapter.mutate("set loop cap", lambda loop=loop, loop_value=loop_value: setattr(loop, "max_iterations", loop_value), live=live)
    # A fragment is a host-supported 500ms wake-up. Without it a deferred
    # canvas mutation would wait for unrelated user input before live run.
    try:
        import streamlit as st

        if container is st and hasattr(st, "fragment"):
            @st.fragment(run_every=0.5)
            def _live_wakeup() -> None:
                adapter.consume_live_run()

            _live_wakeup()
        else:
            adapter.consume_live_run()
    except Exception:  # noqa: BLE001 - fake/headless Streamlit containers
        adapter.consume_live_run()
    container.markdown(adapter.summarize(o.title))
    container.json(adapter.status())
    with container.expander("Types", expanded=False):
        container.json(inspect_types())
    if o.show_images:
        for label, img in adapter.output_images():
            container.image(img, caption=label, width="stretch")
    if o.show_table:
        container.json(adapter.node_rows())
    return adapter


def _st_ports(adapter: GraphAdapter, nid: str | None, direction: str) -> list[str]:
    inst = adapter.state.graph.nodes.get(nid or "")
    if inst is None:
        return []
    if inst.type_id == SUBWORKFLOW_TYPE_ID:
        return [m["key"] for m in inst.params.get("inputs" if direction == "in" else "outputs", [])]
    ndef = get_node(inst.type_id)
    if ndef is None:
        return []
    return [p.key for p in (ndef.inputs if direction == "in" else ndef.outputs)]


# --------------------------------------------------- pygame (host-provided)
@dataclass
class PygameStyle:
    bg: tuple[int, int, int] = (0, 0, 0)
    panel: tuple[int, int, int] = (12, 12, 12)
    edge: tuple[int, int, int] = (90, 140, 200)
    text: tuple[int, int, int] = (230, 230, 235)
    dim: tuple[int, int, int] = (150, 155, 165)
    ok: tuple[int, int, int] = (90, 200, 130)
    err: tuple[int, int, int] = (230, 110, 110)
    idle: tuple[int, int, int] = (140, 140, 150)
    node_w: int = 210
    node_h: int = 70
    col_gap: int = 60
    row_gap: int = 18
    grid_minor: tuple[int, int, int] = (28, 28, 28)
    grid_major: tuple[int, int, int] = (48, 48, 48)


def _make_font(px: int) -> tuple[str, Any]:
    """Best-effort font with a Pillow fallback for minimal pygame builds.

    (Some builds, e.g. pygame 2.6.1 on cp314 without SDL_ttf, ship no font
    extension. Pillow keeps labels readable without SDL_ttf.)
    """
    import pygame

    try:
        f = pygame.font.SysFont("sans", px)
        f.render("x", True, (0, 0, 0))
        return ("font", f)
    except Exception:  # noqa: BLE001, S110 - fall through to freetype
        pass
    try:
        import pygame.freetype

        pygame.freetype.init()
        return ("freetype", pygame.freetype.SysFont(None, px))
    except Exception:  # noqa: BLE001 - fall through to Pillow
        from PIL import ImageFont

        try:
            return ("pillow", ImageFont.truetype("DejaVuSans.ttf", px))
        except OSError:
            return ("pillow", ImageFont.load_default(size=px))


def _render_text(kind_font: tuple[str, Any], text: str, color: tuple[int, int, int]) -> Any:
    import pygame

    kind, f = kind_font
    if kind == "font":
        return f.render(text, True, color)
    if kind == "freetype":
        surf, _rect = f.render(text, color)
        return surf
    from PIL import ImageDraw, ImageFont

    font = f if f is not None else ImageFont.load_default()
    left, top, right, bottom = font.getbbox(text or " ")
    width, height = max(1, right - left), max(1, bottom - top)
    image = _PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((-left, -top), text, font=font, fill=(*color, 255))
    return pygame.image.fromstring(image.tobytes(), image.size, "RGBA")


def _draw_pygame_grid(screen: Any, area: Any, view: tuple[float, float, float], style: PygameStyle) -> None:
    """Draw a pan and zoom aware grid inside the graph viewport."""
    import pygame

    ox, oy, zoom = view
    minor = max(8, round(20 * zoom))
    major = minor * 5
    for spacing, color in ((minor, style.grid_minor), (major, style.grid_major)):
        start_x = area.x + round(ox) % spacing
        start_y = area.y + round(oy) % spacing
        for x in range(start_x, area.right, spacing):
            pygame.draw.line(screen, color, (x, area.y), (x, area.bottom))
        for y in range(start_y, area.bottom, spacing):
            pygame.draw.line(screen, color, (area.x, y), (area.right, y))


def _pygame_node_metrics(graph: Graph, nid: str, style: PygameStyle) -> tuple[int, int, bool]:
    """Return node width, content height, and whether it owns a preview."""
    inst = graph.nodes[nid]
    definition = get_node(inst.type_id)
    inputs = definition.inputs if definition else []
    outputs = definition.outputs if definition else []
    has_preview = any(port.dtype in ("data.FIELD", "data.IMAGE") for port in outputs)
    port_rows = max(1, len(inputs), len(outputs))
    content_rows = 64 if has_preview else min(3, len(definition.params) if definition else 0) * 15 + 10
    return style.node_w, max(style.node_h, 28 + port_rows * 18 + content_rows), has_preview


def compute_boxes(
    graph: Graph,
    area: Any,
    style: PygameStyle,
    offsets: dict[str, tuple[float, float]] | None = None,
    view: tuple[float, float, float] = (0.0, 0.0, 1.0),
    apply_offsets: bool = False,
) -> dict[str, Any]:
    """Node id -> pygame.Rect. ``offsets`` adds session drag displacement (layout px);
    ``view`` is (ox, oy, k) pan/zoom applied in screen space."""
    import pygame

    from .editing import layout_boxes_px

    ox, oy, k = view
    plain, _pos = layout_boxes_px(graph, area.w, area.h, style.node_w, style.node_h, style.col_gap, style.row_gap,
                                  top=36, left=16)
    out = {}
    for nid, (x, y, _w, _h) in plain.items():
        # Once a drag has committed to graph positions, retained offsets are
        # history/debug information only and must not shift a second time.
        dx, dy = (offsets or {}).get(nid, (0.0, 0.0)) if apply_offsets else (0.0, 0.0)
        w, h, _preview = _pygame_node_metrics(graph, nid, style)
        out[nid] = pygame.Rect(int(area.x + ox + (x + dx) * k), int(area.y + oy + (y + dy) * k),
                               max(8, int(w * k)), max(8, int(h * k)))
    return out


#: Port dot colors by data type (ComfyUI-style type coloring).
DTYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "data.NUMBER": (220, 200, 110),
    "data.FIELD": (110, 160, 230),
    "data.IMAGE": (230, 130, 180),
    "data.TEXT": (110, 200, 190),
    "data.TRACE": (110, 200, 190),
    "data.ANY": (150, 150, 160),
}


def port_color(dtype: str) -> tuple[int, int, int]:
    if dtype in DTYPE_COLORS:
        return DTYPE_COLORS[dtype]
    if dtype.startswith("data."):
        return (110, 160, 230)
    return (140, 150, 180)  # extension types (IMAGE/MASK/VIDEO/...)


def port_anchors(graph: Graph, boxes: dict[str, Any]) -> tuple[dict[str, list], dict[str, list]]:
    """(node -> [(port, x, y, dtype)]) for input (left) / output (right) edges."""
    ins: dict[str, list] = {}
    outs: dict[str, list] = {}
    for nid, inst in graph.nodes.items():
        box = boxes.get(nid)
        if box is None:
            continue
        ndef = get_node(inst.type_id)
        if inst.type_id == SUBWORKFLOW_TYPE_ID:
            in_ports = [(m["key"], str(m.get("dtype", "data.ANY"))) for m in inst.params.get("inputs", [])]
            out_ports = [(m["key"], str(m.get("dtype", "data.ANY"))) for m in inst.params.get("outputs", [])]
        else:
            in_ports = [(p.key, p.dtype) for p in ndef.inputs] if ndef else []
            out_ports = [(p.key, p.dtype) for p in ndef.outputs] if ndef else []
        base_w, _base_h, _preview = _pygame_node_metrics(graph, nid, PygameStyle())
        scale = box.width / base_w
        ins[nid] = [(p, box.x, int(box.y + (38 + i * 18) * scale), dt)
                    for i, (p, dt) in enumerate(in_ports)]
        outs[nid] = [(p, box.right, int(box.y + (38 + i * 18) * scale), dt)
                     for i, (p, dt) in enumerate(out_ports)]
    return ins, outs


def output_thumbnail_surface(value: Any, size: int = 56) -> Any | None:
    """Field/Image payload -> small pygame Surface (live preview)."""
    try:
        import pygame
    except ImportError:
        return None
    img = payload_to_pil(value)
    if img is None:
        return None
    img = img.convert("RGB").resize((size, size))
    return pygame.image.frombuffer(img.tobytes(), (size, size), "RGB")


def pygame_node_at(graph: Graph, boxes: dict[str, Any], point: tuple[int, int]) -> str | None:
    """Node id under ``point`` (insertion order wins on overlap)."""
    for nid in graph.nodes:
        if nid in boxes and boxes[nid].collidepoint(point):
            return nid
    return None


def pygame_adjust_selected(
    adapter: GraphAdapter,
    direction: int,
    factor: float = 1.0,
    kinds: tuple[str, ...] = ("float_slider", "slider", "step_slider", "int", "number", "seed", "select", "dropdown"),
) -> str | None:
    """Nudge the first adjustable param of the selected node. Returns a log line."""
    if not adapter.state.selection:
        return None
    nid = adapter.state.selection[0]
    inst = adapter.state.graph.nodes.get(nid)
    ndef = get_node(inst.type_id) if inst else None
    if inst is None or ndef is None:
        return None
    for pdef in ndef.params:
        if canonical_kind(pdef.kind) not in kinds:
            continue
        new_val = adjust_param_value(pdef, inst.params.get(pdef.key, pdef.default), direction, factor)
        if new_val != inst.params.get(pdef.key, pdef.default):
            inst.params[pdef.key] = new_val
            return f"{nid}.{pdef.key} -> {new_val}"
    return None


def pygame_draw_inspector(screen: Any, adapter: GraphAdapter, rect: Any, style: PygameStyle | None = None) -> Any:
    """Side panel with aligned ports, values, previews, and parameters."""
    import pygame

    st = style or PygameStyle()
    font = _make_font(14)
    small = _make_font(12)
    screen.fill(st.bg, rect)
    pygame.draw.rect(screen, st.panel, rect, border_radius=8)
    x, y = rect.x + 10, rect.y + 10

    def line(text: str, f: Any = small, color: Any = None) -> None:
        nonlocal y
        if y > rect.bottom - 20:
            return
        screen.blit(_render_text(f, text[:44], color or st.text), (x, y))
        y += 18

    def table_header() -> None:
        nonlocal y
        for label, offset in (("Name", 0), ("Type", 92), ("Value", 160)):
            screen.blit(_render_text(small, label, st.dim), (x + offset, y))
        y += 17

    def port_row(name: str, dtype: str, value: Any) -> None:
        nonlocal y
        if y > rect.bottom - 24:
            return
        short_type = str(dtype).removeprefix("data.")
        preview = output_thumbnail_surface(value, 56)
        value_text = _value_preview(value) if value is not None else "not ready"
        screen.blit(_render_text(small, name[:13], st.text), (x, y))
        screen.blit(_render_text(small, short_type[:10], st.dim), (x + 92, y))
        screen.blit(_render_text(small, value_text[:20], st.dim), (x + 160, y))
        if preview is not None and y + 62 <= rect.bottom:
            screen.blit(preview, (x + 160, y + 17))
            y += 62
        else:
            y += 18

    line("Inspector", font)
    sel = adapter.state.selection[:1]
    if not sel:
        line("click a node to select", small, st.dim)
    else:
        detail = adapter.describe_node(sel[0])
        if "error" in detail:
            line(detail["error"], small, st.err)
            return rect
        line(f"{detail.get('title', sel[0])}", font)
        if detail.get("loop"):
            line(f"[loop:{detail['loop']}]", small, st.edge)
        line("Inputs", font)
        table_header()
        for key, dtype in detail.get("inputs", []):
            link = next((item for item in adapter.state.graph.links
                         if item.to_node == sel[0] and item.to_port == key), None)
            value = adapter.executor.outputs.get(link.from_node, {}).get(link.from_port) if link else None
            port_row(key, dtype, value)
        line("Outputs", font)
        table_header()
        for key, dtype in detail.get("outputs", []):
            port_row(key, dtype, adapter.executor.outputs.get(sel[0], {}).get(key))
        line("Parameters", font)
        table_header()
        for p in detail.get("params", [])[:10]:
            screen.blit(_render_text(small, str(p["label"])[:13], st.text), (x, y))
            screen.blit(_render_text(small, str(p["kind"]).removeprefix("data.")[:10], st.dim), (x + 92, y))
            screen.blit(_render_text(small, str(p["value"])[:20], st.dim), (x + 160, y))
            y += 18
        sub = detail.get("subflow")
        if sub:
            line(f"[sub:{sub['nodes']}] {sub['label']}", small, st.edge)
            for title in sub["titles"][:6]:
                line(f"  - {title}", small, st.dim)
    if adapter.state.graph.loops:
        line("Loops:", font)
        for loop in adapter.state.graph.loops:
            line(f"~ {loop.name}: max={loop.max_iterations}", small, st.dim)
    line("[ ] tweak · T toggle · D cycle · E expand · R run", small, st.dim)
    return rect


def pygame_register_grapheditor(
    screen: Any,
    source: GraphAdapter | EditorState | Graph,
    rect: Any | None = None,
    style: PygameStyle | None = None,
    title: str = "Graph Editor",
    offsets: dict[str, tuple[float, float]] | None = None,
    view: tuple[float, float, float] = (0.0, 0.0, 1.0),
    thumbnails: bool = True,
    highlight: set[tuple[str, str]] | None = None,
    apply_offsets: bool = False,
) -> Any:
    """Draw graph visualisation onto a caller-provided pygame Surface.

    ``screen`` is your display/subsurface. We paint inside ``rect`` (default:
    the whole surface inset by 12px) and return the dirty rect. ``offsets``
    applies session drag displacement (defaults to the state's drag_offsets);
    ``view`` is (ox, oy, k) pan/zoom; ``thumbnails`` toggles live Field/Image
    previews; ``highlight`` rings (node, port) anchors (e.g. while wiring).
    No event loop is owned here; see ``pygame_app.run_pygame_viewer``
    for the standalone live editor loop.
    """
    try:
        import pygame
    except ImportError as e:
        raise ImportError("Install the pygame extra: pip install 'easygrapheditor[pygame]'") from e

    st = style or PygameStyle()
    adapter = GraphAdapter.from_any(source)
    if offsets is None:
        offsets = adapter.state.drag_offsets
    graph = adapter.state.graph
    area = rect or screen.get_rect().inflate(-24, -24)
    title_font = _make_font(15)
    font = _make_font(max(8, round(15 * view[2])))
    small = _make_font(max(7, round(12 * view[2])))

    old_clip = screen.get_clip()
    screen.set_clip(area)
    screen.fill(st.bg, area)
    pygame.draw.rect(screen, st.panel, area, border_radius=8)
    _draw_pygame_grid(screen, area, view, st)
    pygame.draw.rect(screen, st.grid_major, area, 1, border_radius=8)
    screen.blit(_render_text(title_font, title, st.text), (area.x + 12, area.y + 8))

    boxes = compute_boxes(graph, area, st, offsets=offsets, view=view, apply_offsets=apply_offsets)
    ins, outs = port_anchors(graph, boxes)
    for link in graph.links:
        if link.from_node in boxes and link.to_node in boxes:
            source = next(((x, y) for port, x, y, _dtype in outs.get(link.from_node, []) if port == link.from_port), boxes[link.from_node].midright)
            target = next(((x, y) for port, x, y, _dtype in ins.get(link.to_node, []) if port == link.to_port), boxes[link.to_node].midleft)
            handle = min(120.0, max(16.0, abs(target[0] - source[0]) * 0.45))
            points = []
            for index in range(17):
                t = index / 16
                u = 1.0 - t
                x = u**3 * source[0] + 3 * u**2 * t * (source[0] + handle) + 3 * u * t**2 * (target[0] - handle) + t**3 * target[0]
                y = u**3 * source[1] + 3 * u**2 * t * source[1] + 3 * u * t**2 * target[1] + t**3 * target[1]
                points.append((round(x), round(y)))
            pygame.draw.lines(screen, st.edge, False, points, 2)

    # visual groups behind member nodes
    from .editing import GROUP_COLORS

    for group in graph.groups:
        members = [boxes[nid] for nid in group.nodes if nid in boxes]
        if not members:
            continue
        gx0 = min(b.x for b in members) - 14
        gy0 = min(b.y for b in members) - 30
        gx1 = max(b.right for b in members) + 14
        gy1 = max(b.bottom for b in members) + 14
        color = GROUP_COLORS.get(group.color, GROUP_COLORS["slate"])
        pygame.draw.rect(screen, color, pygame.Rect(gx0, gy0, gx1 - gx0, gy1 - gy0), 2, border_radius=10)
        screen.blit(_render_text(font, group.title[:36], color), (gx0 + 10, gy0 + 6))

    thumbs: dict[str, Any] = {}
    if thumbnails:
        for nid, port_map in adapter.executor.outputs.items():
            for value in port_map.values():
                surf = output_thumbnail_surface(value)
                if surf is not None:
                    thumbs[nid] = surf
                    break

    selected = set(adapter.state.selection)
    highlight = highlight or set()
    for nid, inst in graph.nodes.items():
        ndef = get_node(inst.type_id)
        rep = adapter.report.per_node.get(nid) if adapter.report else None
        working = adapter.state.working_nodes.get(nid, {})
        color = st.edge if working else (st.idle if rep is None else (st.ok if rep.status in ("ok", "cached") else st.err))
        r = boxes[nid]
        scale = r.width / st.node_w
        pygame.draw.rect(screen, (22, 24, 30), r, border_radius=4)
        pygame.draw.rect(screen, color, r, 2, border_radius=6)
        header = pygame.Rect(r.x, r.y, r.width, max(12, round(26 * scale)))
        pygame.draw.rect(screen, (45, 50, 62), header, border_top_left_radius=4, border_top_right_radius=4)
        if nid in selected:
            pygame.draw.rect(screen, (240, 240, 245), r, 4, border_radius=6)
        badges = ""
        loop = adapter.loop_of(nid)
        if loop:
            badges += f" [loop:{loop}]"
        if inst.type_id == SUBWORKFLOW_TYPE_ID:
            badges += f" [sub:{len(inst.params.get('nodes', []))}]"
        group = graph.group_of(nid)
        if group:
            badges += f" [{group.title[:12]}]"
        screen.blit(_render_text(font, (ndef.title if ndef else inst.type_id)[:22], st.text), (r.x + 8, r.y + 5))
        sub = f"{working.get('stage', working.get('status'))} {working.get('frac', 0.0) * 100:.0f}%" if working else (f"{rep.status} {rep.ms:.0f}ms" if rep else "not run")
        sub_surf = _render_text(small, f"{sub}{badges}"[:22], st.dim)
        screen.blit(sub_surf, (r.right - sub_surf.get_width() - 7, r.y + 7))
        input_ports = ins.get(nid, [])
        output_ports = outs.get(nid, [])
        for index in range(max(len(input_ports), len(output_ports))):
            if index < len(input_ports):
                port, _x, y, dtype = input_ports[index]
                label = f"{port} : {dtype.removeprefix('data.')}"
                screen.blit(_render_text(small, label[:22], st.dim), (r.x + 8, y - 6))
            if index < len(output_ports):
                port, _x, y, dtype = output_ports[index]
                label = f"{port} : {dtype.removeprefix('data.')}"
                label_surf = _render_text(small, label[:22], st.dim)
                screen.blit(label_surf, (r.right - label_surf.get_width() - 8, y - 6))
        port_rows = max(1, len(input_ports), len(output_ports))
        content_y = r.y + round((28 + port_rows * 18) * scale)
        if nid in thumbs:
            preview_size = max(18, round(56 * scale))
            preview = pygame.transform.smoothscale(thumbs[nid], (preview_size, preview_size))
            screen.blit(preview, (r.x + 8, content_y))
        if ndef:
            param_x = r.x + (round(72 * scale) if nid in thumbs else 8)
            for index, pdef in enumerate(ndef.params[:3]):
                value = inst.params.get(pdef.key, pdef.default)
                screen.blit(_render_text(small, f"{pdef.label or pdef.key}={value}"[:25], st.dim),
                            (param_x, content_y + index * max(12, round(15 * scale))))
    # typed port dots (ringed when in the wiring highlight set)
    for nid, anchors in ins.items():
        for port, x, y, dtype in anchors:
            pygame.draw.circle(screen, port_color(dtype), (x, y), 4)
            if (nid, port) in highlight:
                pygame.draw.circle(screen, (240, 240, 245), (x, y), 7, 2)
    for nid, anchors in outs.items():
        for port, x, y, dtype in anchors:
            pygame.draw.circle(screen, port_color(dtype), (x, y), 4)
            if (nid, port) in highlight:
                pygame.draw.circle(screen, (240, 240, 245), (x, y), 7, 2)

    hint = _render_text(small, f"{len(graph.nodes)} nodes · {len(graph.links)} links · click select · [R] run · [Q] quit", st.dim)
    screen.blit(hint, (area.x + 12, area.bottom - 22))
    screen.set_clip(old_clip)
    return area


__all__ = [
    "DTYPE_COLORS",
    "AdapterOptions",
    "GraphAdapter",
    "PygameStyle",
    "StreamlitContainer",
    "compute_boxes",
    "field_to_pil",
    "gradio_register_grapheditor",
    "image_to_pil",
    "layout_graph",
    "output_thumbnail_surface",
    "payload_to_pil",
    "port_anchors",
    "port_color",
    "pygame_adjust_selected",
    "pygame_draw_inspector",
    "pygame_node_at",
    "pygame_register_grapheditor",
    "streamlit_register_grapheditor",
]
