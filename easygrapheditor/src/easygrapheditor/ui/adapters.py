"""UI adapters: backend-neutral visualisation over an executed graph.

Design (dependency injection): the caller owns the host object and hands it
to us — we never create windows/servers/blocks ourselves here:

* gradio:      ``gradio_register_grapheditor(blocks_or_tab, source, ...)``
* streamlit:   ``streamlit_register_grapheditor(source, container=st_tab, ...)``
* pygame:      ``pygame_register_grapheditor(screen, source, ...)``

``source`` is a ``GraphAdapter`` (or anything it accepts: ``EditorState`` /
``Graph``). Standalone entry points that *do* own the loop/server live in
``gradio_app.py`` / ``streamlit_app.py`` / ``pygame_app.py`` and delegate here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..engine.cache import Cache
from ..engine.execute import Executor, RunReport
from ..engine.graph import SUBWORKFLOW_TYPE_ID, Graph
from ..engine.nodes import canonical_kind, get_node
from ..engine.types import Field, Image
from .canvas import EditorState
from .inspector import inspect_node
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


# ------------------------------------------------------------------- adapter
@dataclass
class AdapterOptions:
    title: str = "Graph Editor"
    show_images: bool = True
    show_table: bool = True
    inspector: bool = True  # param editors / loop controls / expand buttons
    run_label: str = "▶ Run"


class GraphAdapter:
    """Backend-neutral view-model: owns execution + summarisation, not display."""

    def __init__(self, state: EditorState, cache: Cache | None = None) -> None:
        self.state = state
        self.cache = cache or Cache()
        self.executor = Executor(state.graph, cache=self.cache)
        self.report: RunReport | None = state.run_report

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
    def run(self) -> RunReport:
        report = self.executor.run_blocking()
        self.report = report
        self.state.run_report = report
        return report

    def clear_cache(self) -> None:
        """Forget cached node outputs (next run recomputes everything)."""
        self.cache.clear()

    def mark_dirty(self) -> None:
        """Invalidate the last report after a structural edit."""
        self.report = None
        self.state.run_report = None

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
        for nid, inst in self.state.graph.nodes.items():
            ndef = get_node(inst.type_id)
            rep = self.report.per_node.get(nid) if self.report else None
            subflow = len(inst.params.get("nodes", [])) if inst.type_id == SUBWORKFLOW_TYPE_ID else 0
            rows.append(
                {
                    "node": nid,
                    "title": ndef.title if ndef else inst.type_id,
                    "type": inst.type_id,
                    "status": rep.status if rep else "not run",
                    "ms": round(rep.ms, 1) if rep else 0.0,
                    "cached": bool(rep and rep.cache_hit),
                    "error": rep.error if rep and rep.error else "",
                    "loop": self.loop_of(nid),
                    "subflow": subflow,
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
        detail["instance_params"] = dict(inst.params)
        if inst.type_id == SUBWORKFLOW_TYPE_ID:
            from ..engine.subworkflows import describe_subworkflow

            detail["subflow"] = describe_subworkflow(inst)
        return detail

    def expand_subworkflow(self, nid: str) -> list[str]:
        """Inline a subworkflow node; returns restored ids (re-runs nothing)."""
        from ..engine.subworkflows import expand_subworkflow_node

        restored = expand_subworkflow_node(self.state.graph, nid)
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
            lines.append("_Not run yet — press Run._")
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
            err = f" — {row['error']}" if row["error"] else ""
            lines.append(f"- {flag} `{row['title']}` ({row['node']}) — {row['status']}{extra}{badges} · {row['ms']} ms{err}")
        return "\n".join(lines)


# -------------------------------------------------------------------- layout
def layout_graph(graph: Graph) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Depth (column) per node from topo order + stable (x, y) unit positions.

    Returns (positions, depths) with x = depth, y = index within depth.
    Falls back to insertion order when the graph has a cycle.
    """
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
    pos = {nid: (float(depth[nid]), float(i)) for col, members in columns.items() for i, nid in enumerate(members)}
    return pos, depth


# ------------------------------------------------- gradio (host-provided)
def gradio_register_grapheditor(host: Any, source: GraphAdapter | EditorState | Graph, **opts: Any) -> dict[str, Any]:
    """Mount a LIVE node editor into a caller-provided Gradio container.

    ``host`` is your ``gr.Blocks`` / ``gr.Tab`` / ``gr.Row`` / ``gr.Column`` —
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
        add_node_at,
        compatible_inputs,
        connect,
        disconnect,
        link_labels,
        node_type_choices,
        remove_node,
        render_canvas_image,
    )

    o = AdapterOptions(**{k: v for k, v in opts.items() if k in AdapterOptions.__dataclass_fields__})
    adapter = GraphAdapter.from_any(source)
    types = node_type_choices()

    def _refresh() -> tuple[str, Any, list, list]:
        return (
            adapter.summarize(o.title),
            render_canvas_image(adapter.state.graph, adapter.report, adapter.executor.outputs),
            [img for _, img in adapter.output_images()],
            adapter.node_rows(),
        )

    def _refresh_full() -> tuple:
        """_refresh + fresh node/link picker choices (structure may have changed)."""
        s, c, g, t = _refresh()
        ids = sorted(adapter.state.graph.nodes)
        links = link_labels(adapter.state.graph)
        return (s, c, g, t, gr.Dropdown(choices=ids), gr.Dropdown(choices=ids),
                gr.Dropdown(choices=links), gr.Dropdown(choices=ids))

    with host:
        gr.Markdown(f"## {o.title}")
        summary = gr.Markdown(adapter.summarize(o.title))
        with gr.Row():
            run_btn = gr.Button(o.run_label, variant="primary")
            live_box = gr.Checkbox(value=False, label="Live (auto-run on change)")
            clear_btn = gr.Button("Clear cache + rerun")
        canvas = gr.Image(value=render_canvas_image(adapter.state.graph, adapter.report, adapter.executor.outputs), label="Canvas")
        gallery = gr.Gallery(value=[img for _, img in adapter.output_images()], label="Previews", visible=o.show_images)
        table = gr.JSON(value=adapter.node_rows(), label="Nodes", visible=o.show_table)
        notice = gr.Markdown("", visible=False)

        def _refresh_ids() -> list[str]:
            return sorted(adapter.state.graph.nodes)

        # -- handlers (wired to buttons after all components exist) --------
        def _on_run(_live: bool = False) -> tuple:
            adapter.run()
            return _refresh_full()

        def _maybe_live(live: bool) -> tuple:
            if live:
                adapter.run()
            return _refresh_full()

        def _on_clear() -> tuple:
            adapter.clear_cache()
            adapter.run()
            return _refresh_full()

        with gr.Accordion("Library: add node", open=False):
            type_drop = gr.Dropdown(choices=[label for _, label in types], label="Node type")
            add_btn = gr.Button("Add node", size="sm")

            def _on_add(label: str | None, live: bool) -> tuple:
                if not label:
                    s = _refresh_full()
                    return (*s, gr.Markdown("Pick a node type first.", visible=True))
                type_id = next(t for t, lab in types if lab == label)
                inst = add_node_at(adapter.state.graph, type_id, pos=(0.0, 0.0))
                adapter.mark_dirty()
                s = _maybe_live(live)
                note = f"Added `{inst.id}` — reload the UI to refresh the inspector."
                return (*s, gr.Markdown(note, visible=True))

        with gr.Accordion("Links: connect / disconnect", open=False):
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
                    link = connect(adapter.state.graph, a or "", ap or "", b or "", bp or "")
                    adapter.mark_dirty()
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
                            disconnect(adapter.state.graph, link.from_node, link.from_port, link.to_node, link.to_port)
                            break
                    adapter.mark_dirty()
                return (*_maybe_live(live), gr.Markdown("", visible=False))

        with gr.Accordion("Delete node", open=False):
            del_drop = gr.Dropdown(choices=_refresh_ids(), label="Node")
            del_btn = gr.Button("Delete node + incident links", size="sm")

            def _on_delete(nid: str | None, live: bool) -> tuple:
                if nid and nid in adapter.state.graph.nodes:
                    remove_node(adapter.state.graph, nid)
                    adapter.mark_dirty()
                return (*_maybe_live(live), gr.Markdown("", visible=False))

        # -- wire buttons (every action refreshes views + picker choices) --
        outs_full = [summary, canvas, gallery, table, from_node, to_node, unlink_drop, del_drop]
        outs_note = [*outs_full, notice]
        run_btn.click(_on_run, inputs=[live_box], outputs=outs_full)
        clear_btn.click(_on_clear, outputs=outs_full)
        add_btn.click(_on_add, inputs=[type_drop, live_box], outputs=outs_note)
        link_btn.click(_on_link, inputs=[from_node, from_port, to_node, to_port, live_box], outputs=outs_note)
        unlink_btn.click(_on_unlink, inputs=[unlink_drop, live_box], outputs=outs_note)
        del_btn.click(_on_delete, inputs=[del_drop, live_box], outputs=outs_note)

        if o.inspector:
            with gr.Accordion("Inspector: node params (Apply writes back)", open=False):
                for nid, inst in list(adapter.state.graph.nodes.items()):
                    ndef = get_node(inst.type_id)
                    if ndef is None:
                        continue
                    with gr.Accordion(f"{ndef.title} (`{nid}`)", open=False):
                        comps = [gradio_param_component(gr, p, inst.params.get(p.key, p.default)) for p in ndef.params]

                        def _on_apply(*values: Any, _nid: str = nid, _ndef: Any = ndef) -> tuple:
                            live = bool(values[-1])
                            for pdef, raw in zip(_ndef.params, values[:-1]):
                                adapter.state.graph.nodes[_nid].params[pdef.key] = cast_param_value(pdef, raw)
                            return _maybe_live(live)

                        gr.Button("Apply params", size="sm").click(_on_apply, inputs=[*comps, live_box], outputs=outs_full)
                        if inst.type_id == SUBWORKFLOW_TYPE_ID:
                            expand_btn = gr.Button("Expand subworkflow inline", size="sm")

                            def _on_expand(_x: Any = None, _nid: str = nid) -> tuple:
                                restored = adapter.expand_subworkflow(_nid)
                                s = _refresh_full()
                                note = f"Expanded into {len(restored)} nodes — reload the UI to refresh the inspector."
                                return (*s, gr.Markdown(note, visible=True))

                            expand_btn.click(_on_expand, outputs=[*outs_full, notice])

            if adapter.state.graph.loops:
                with gr.Accordion("Loops: max_iterations (Apply writes back)", open=False):
                    for i, loop in enumerate(adapter.state.graph.loops):
                        max_box = gr.Number(value=loop.max_iterations, label=f"{loop.name} · max_iterations", precision=0)

                        def _on_loop_apply(v: float, live: bool, _i: int = i) -> tuple:
                            adapter.state.graph.loops[_i].max_iterations = max(1, int(v))
                            return _maybe_live(live)

                        gr.Button(f"Apply {loop.name}", size="sm").click(_on_loop_apply, inputs=[max_box, live_box], outputs=outs_full)
    return {"summary": summary, "canvas": canvas, "gallery": gallery, "table": table, "run": run_btn,
            "live": live_box, "clear": clear_btn, "notice": notice, "adapter": adapter}


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
        add_node_at,
        compatible_inputs,
        connect,
        disconnect,
        link_labels,
        node_type_choices,
        remove_node,
        render_canvas_image,
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

    live = False
    if container.button(o.run_label):
        adapter.run()
    with container.expander("Toolbar: live mode + cache", expanded=False):
        live = bool(container.checkbox("Live (auto-run on change)", value=False, key="ege:live"))
        if container.button("Clear cache + rerun", key="ege:clear"):
            adapter.clear_cache()
            adapter.run()
    if o.inspector:
        with container.expander("Library: add node", expanded=False):
            labels = [label for _, label in node_type_choices()]
            picked = container.selectbox("Node type", labels, key="ege:add:type")
            if container.button("Add node", key="ege:add:go") and picked:
                lookup = {label: t for t, label in node_type_choices()}
                inst = add_node_at(adapter.state.graph, lookup[picked], pos=(0.0, 0.0))
                adapter.mark_dirty()
                if live:
                    adapter.run()
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
                    link = connect(adapter.state.graph, from_nid or "", from_pt or "", to_nid or "", to_pt or "")
                    adapter.mark_dirty()
                    if live:
                        adapter.run()
                    container.markdown(f"Connected `{link.from_node}.{link.from_port} → {link.to_node}.{link.to_port}`.")
                except ValueError as e:
                    container.markdown(f"Cannot connect: {e}")
                _rerun()
            existing = link_labels(adapter.state.graph)
            doomed = container.selectbox("Existing link", existing, key="ege:unlink") if existing else None
            if existing and container.button("Disconnect", key="ege:unlink:go") and doomed:
                for link in adapter.state.graph.links:
                    if f"{link.from_node}.{link.from_port} -> {link.to_node}.{link.to_port}" == doomed:
                        disconnect(adapter.state.graph, link.from_node, link.from_port, link.to_node, link.to_port)
                        break
                adapter.mark_dirty()
                if live:
                    adapter.run()
                _rerun()
        with container.expander("Delete node", expanded=False):
            doomed = container.selectbox("Node", ids, key="ege:delete") if ids else None
            if ids and container.button("Delete node + incident links", key="ege:delete:go") and doomed:
                remove_node(adapter.state.graph, doomed)
                adapter.mark_dirty()
                if live:
                    adapter.run()
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
                        container.markdown(f"_Subworkflow: {detail['subflow']['nodes']} nodes — {', '.join(detail['subflow']['titles'][:8])}_")
                        if container.button("Expand subworkflow inline", key=f"ege:expand:{nid}"):
                            adapter.expand_subworkflow(nid)
                            _rerun()
                    for pdef in ndef.params:
                        key = f"ege:{nid}:{pdef.key}"
                        current = inst.params.get(pdef.key, pdef.default)
                        new_val = streamlit_param_widget(container, pdef, current, key)
                        casted = cast_param_value(pdef, new_val)
                        if casted != current:
                            inst.params[pdef.key] = casted
                            if live:
                                adapter.run()
        if adapter.state.graph.loops:
            with container.expander("Loops: max_iterations", expanded=False):
                for loop in adapter.state.graph.loops:
                    loop.max_iterations = max(
                        1, int(container.number_input(f"{loop.name} · max_iterations", value=int(loop.max_iterations), step=1, key=f"ege:loop:{loop.name}"))
                    )
    container.image(render_canvas_image(adapter.state.graph, adapter.report, adapter.executor.outputs), caption="Canvas")
    container.markdown(adapter.summarize(o.title))
    if o.show_images:
        for label, img in adapter.output_images():
            container.image(img, caption=label, use_container_width=True)
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
    bg: tuple[int, int, int] = (18, 20, 26)
    panel: tuple[int, int, int] = (32, 36, 46)
    edge: tuple[int, int, int] = (90, 140, 200)
    text: tuple[int, int, int] = (230, 230, 235)
    dim: tuple[int, int, int] = (150, 155, 165)
    ok: tuple[int, int, int] = (90, 200, 130)
    err: tuple[int, int, int] = (230, 110, 110)
    idle: tuple[int, int, int] = (140, 140, 150)
    node_w: int = 190
    node_h: int = 52
    col_gap: int = 60
    row_gap: int = 18


def _make_font(px: int) -> tuple[str, Any]:
    """Best-effort font: pygame.font, else pygame.freetype, else blank boxes.

    (Some builds, e.g. pygame 2.6.1 on cp314 without SDL_ttf, ship no font
    extension — degrade instead of crashing.)
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
    except Exception:  # noqa: BLE001 - fall through to blank
        return ("none", None)


def _render_text(kind_font: tuple[str, Any], text: str, color: tuple[int, int, int]) -> Any:
    import pygame

    kind, f = kind_font
    if kind == "font":
        return f.render(text, True, color)
    if kind == "freetype":
        surf, _rect = f.render(text, color)
        return surf
    surf = pygame.Surface((max(8, len(text) * 7), 14))
    return surf


def compute_boxes(graph: Graph, area: Any, style: PygameStyle, offsets: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    """Node id -> pygame.Rect. ``offsets`` adds session drag displacement (px)."""
    import pygame

    from .editing import layout_boxes_px

    plain = layout_boxes_px(graph, area.w, area.h, style.node_w, style.node_h, style.col_gap, style.row_gap,
                            top=area.y + 36, left=area.x + 16)
    boxes, _pos = plain
    out = {}
    for nid, (x, y, w, h) in boxes.items():
        dx, dy = (offsets or {}).get(nid, (0.0, 0.0))
        out[nid] = pygame.Rect(int(x + dx), int(y + dy), w, h)
    return out


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
    """Side panel: selected node params (label=value [kind]), loop + subflow detail."""
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
        for p in detail.get("params", [])[:10]:
            line(f"{p['label']}={p['value']} [{p['kind']}]", small, st.dim)
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
    thumbnails: bool = True,
) -> Any:
    """Draw graph visualisation onto a caller-provided pygame Surface.

    ``screen`` is your display/subsurface — we paint inside ``rect`` (default:
    the whole surface inset by 12px) and return the dirty rect. ``offsets``
    applies session drag displacement (defaults to the state's drag_offsets);
    ``thumbnails`` toggles live Field/Image previews. No event loop is owned
    here; see ``pygame_app.run_pygame_viewer`` for the standalone live
    editor loop.
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
    font = _make_font(15)
    small = _make_font(12)

    screen.fill(st.bg, area)
    pygame.draw.rect(screen, st.panel, area, border_radius=8)
    screen.blit(_render_text(font, title, st.text), (area.x + 12, area.y + 8))

    boxes = compute_boxes(graph, area, st, offsets=offsets)
    for link in graph.links:
        if link.from_node in boxes and link.to_node in boxes:
            a, b = boxes[link.from_node], boxes[link.to_node]
            pygame.draw.line(screen, st.edge, a.midright, b.midleft, 2)

    thumbs: dict[str, Any] = {}
    if thumbnails:
        for nid, port_map in adapter.executor.outputs.items():
            for value in port_map.values():
                surf = output_thumbnail_surface(value)
                if surf is not None:
                    thumbs[nid] = surf
                    break

    selected = set(adapter.state.selection)
    for nid, inst in graph.nodes.items():
        ndef = get_node(inst.type_id)
        rep = adapter.report.per_node.get(nid) if adapter.report else None
        color = st.idle if rep is None else (st.ok if rep.status in ("ok", "cached") else st.err)
        r = boxes[nid]
        pygame.draw.rect(screen, (45, 50, 62), r, border_radius=6)
        pygame.draw.rect(screen, color, r, 2, border_radius=6)
        if nid in selected:
            pygame.draw.rect(screen, (240, 240, 245), r, 4, border_radius=6)
        badges = ""
        loop = adapter.loop_of(nid)
        if loop:
            badges += f" [loop:{loop}]"
        if inst.type_id == SUBWORKFLOW_TYPE_ID:
            badges += f" [sub:{len(inst.params.get('nodes', []))}]"
        screen.blit(_render_text(font, (ndef.title if ndef else inst.type_id)[:22], st.text), (r.x + 8, r.y + 6))
        sub = f"{rep.status} {rep.ms:.0f}ms" if rep else "not run"
        screen.blit(_render_text(small, f"{sub}{badges}"[:34], st.dim), (r.x + 8, r.y + 28))
        if nid in thumbs:
            screen.blit(thumbs[nid], (r.right + 4, r.y - 2))

    hint = _render_text(small, f"{len(graph.nodes)} nodes · {len(graph.links)} links · click select · [R] run · [Q] quit", st.dim)
    screen.blit(hint, (area.x + 12, area.bottom - 22))
    return area


__all__ = [
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
    "pygame_adjust_selected",
    "pygame_draw_inspector",
    "pygame_node_at",
    "pygame_register_grapheditor",
    "streamlit_register_grapheditor",
]
