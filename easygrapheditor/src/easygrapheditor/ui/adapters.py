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
from ..engine.graph import Graph
from ..engine.nodes import get_node
from ..engine.types import Field, Image
from .canvas import EditorState

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

    @property
    def ok(self) -> bool:
        return self.report.ok() if self.report is not None else False

    # -- inspection --------------------------------------------------
    def node_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for nid, inst in self.state.graph.nodes.items():
            ndef = get_node(inst.type_id)
            rep = self.report.per_node.get(nid) if self.report else None
            rows.append(
                {
                    "node": nid,
                    "title": ndef.title if ndef else inst.type_id,
                    "type": inst.type_id,
                    "status": rep.status if rep else "not run",
                    "ms": round(rep.ms, 1) if rep else 0.0,
                    "cached": bool(rep and rep.cache_hit),
                    "error": rep.error if rep and rep.error else "",
                }
            )
        return rows

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
        lines = [f"## {title}", f"{len(g.nodes)} nodes · {len(g.links)} links"]
        if self.report is None:
            lines.append("_Not run yet — press Run._")
            return "\n\n".join(lines)
        ok_n = sum(1 for r in self.report.per_node.values() if r.status in ("ok", "cached"))
        lines.append(f"Run: **{'ok' if self.report.ok() else 'errors'}** · {self.report.ms:.1f} ms · {ok_n}/{len(self.report.per_node)} ok")
        for row in self.node_rows():
            flag = "✓" if row["status"] in ("ok", "cached") else ("✗" if row["status"] == "error" else "·")
            extra = " (cached)" if row["cached"] else ""
            err = f" — {row['error']}" if row["error"] else ""
            lines.append(f"- {flag} `{row['title']}` ({row['node']}) — {row['status']}{extra} · {row['ms']} ms{err}")
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
    """Mount graph visualisation into a caller-provided Gradio container.

    ``host`` is your ``gr.Blocks`` / ``gr.Tab`` / ``gr.Row`` / ``gr.Column`` —
    we add our components inside it (e.g. your submenu screen) and return
    component refs ``{"summary", "gallery", "table", "run", "adapter"}``.
    """
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError("Install the gradio extra: pip install 'easygrapheditor[gradio]'") from e

    o = AdapterOptions(**{k: v for k, v in opts.items() if k in AdapterOptions.__dataclass_fields__})
    adapter = GraphAdapter.from_any(source)

    with host:
        gr.Markdown(f"## {o.title}")
        summary = gr.Markdown(adapter.summarize(o.title))
        gallery = gr.Gallery(value=[img for _, img in adapter.output_images()], label="Outputs", visible=o.show_images)
        table = gr.JSON(value=adapter.node_rows(), label="Nodes", visible=o.show_table)
        run_btn = gr.Button(o.run_label)

        def _on_run() -> tuple[str, list, list]:
            adapter.run()
            return (
                adapter.summarize(o.title),
                [img for _, img in adapter.output_images()],
                adapter.node_rows(),
            )

        run_btn.click(_on_run, outputs=[summary, gallery, table])
    return {"summary": summary, "gallery": gallery, "table": table, "run": run_btn, "adapter": adapter}


# ----------------------------------------------- streamlit (host-provided)
class StreamlitContainer(Protocol):
    def markdown(self, body: str, *a: Any, **k: Any) -> Any: ...
    def image(self, image: Any, *a: Any, **k: Any) -> Any: ...
    def json(self, data: Any, *a: Any, **k: Any) -> Any: ...
    def button(self, label: str, *a: Any, **k: Any) -> bool: ...


def streamlit_register_grapheditor(
    source: GraphAdapter | EditorState | Graph, container: StreamlitContainer | None = None, **opts: Any
) -> GraphAdapter:
    """Render graph visualisation into a caller-provided Streamlit container.

    ``container`` is your tab/sidebar/expander (``st.tabs(...)[0]``,
    ``st.sidebar``, ``st.expander(...)`` …); defaults to the ``streamlit``
    module itself, i.e. the current page. Returns the adapter.
    """
    if container is None:
        try:
            import streamlit as st
        except ImportError as e:
            raise ImportError("Install the streamlit extra: pip install 'easygrapheditor[streamlit]'") from e
        container = st
    o = AdapterOptions(**{k: v for k, v in opts.items() if k in AdapterOptions.__dataclass_fields__})
    adapter = GraphAdapter.from_any(source)
    if container.button(o.run_label):
        adapter.run()
    container.markdown(adapter.summarize(o.title))
    if o.show_images:
        for label, img in adapter.output_images():
            container.image(img, caption=label, use_container_width=True)
    if o.show_table:
        container.json(adapter.node_rows())
    return adapter


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


def pygame_register_grapheditor(
    screen: Any,
    source: GraphAdapter | EditorState | Graph,
    rect: Any | None = None,
    style: PygameStyle | None = None,
    title: str = "Graph Editor",
) -> Any:
    """Draw graph visualisation onto a caller-provided pygame Surface.

    ``screen`` is your display/subsurface — we paint inside ``rect`` (default:
    the whole surface inset by 12px) and return the dirty rect. No event loop
    is owned here; see ``pygame_app.run_pygame_viewer`` for a standalone loop.
    """
    try:
        import pygame
    except ImportError as e:
        raise ImportError("Install the pygame extra: pip install 'easygrapheditor[pygame]'") from e

    st = style or PygameStyle()
    adapter = GraphAdapter.from_any(source)
    graph = adapter.state.graph
    area = rect or screen.get_rect().inflate(-24, -24)
    font = _make_font(15)
    small = _make_font(12)

    screen.fill(st.bg, area)
    pygame.draw.rect(screen, st.panel, area, border_radius=8)
    screen.blit(_render_text(font, title, st.text), (area.x + 12, area.y + 8))

    pos, _depth = layout_graph(graph)
    max_col = max((int(x) for x, _ in pos.values()), default=0)
    max_row = max((int(y) for _, y in pos.values()), default=0)
    grid_x0, grid_y0 = area.x + 16, area.y + 36
    avail_w = max(area.w - 32, st.node_w)
    avail_h = max(area.h - 60, st.node_h)
    step_x = min(st.node_w + st.col_gap, avail_w / max(max_col + 1, 1))
    step_y = min(st.node_h + st.row_gap, avail_h / max(max_row + 1, 1))

    def box(nid: str) -> Any:
        x, y = pos[nid]
        return pygame.Rect(int(grid_x0 + x * step_x), int(grid_y0 + y * step_y), st.node_w, st.node_h)

    boxes = {nid: box(nid) for nid in graph.nodes}
    for link in graph.links:
        if link.from_node in boxes and link.to_node in boxes:
            a, b = boxes[link.from_node], boxes[link.to_node]
            pygame.draw.line(screen, st.edge, a.midright, b.midleft, 2)

    for nid, inst in graph.nodes.items():
        ndef = get_node(inst.type_id)
        rep = adapter.report.per_node.get(nid) if adapter.report else None
        color = st.idle if rep is None else (st.ok if rep.status in ("ok", "cached") else st.err)
        r = boxes[nid]
        pygame.draw.rect(screen, (45, 50, 62), r, border_radius=6)
        pygame.draw.rect(screen, color, r, 2, border_radius=6)
        screen.blit(_render_text(font, (ndef.title if ndef else inst.type_id)[:24], st.text), (r.x + 8, r.y + 6))
        sub = f"{rep.status} {rep.ms:.0f}ms" if rep else "not run"
        screen.blit(_render_text(small, f"{sub}", st.dim), (r.x + 8, r.y + 28))

    hint = _render_text(small, f"{len(graph.nodes)} nodes · {len(graph.links)} links · [R] run · [Q] quit", st.dim)
    screen.blit(hint, (area.x + 12, area.bottom - 22))
    return area


__all__ = [
    "AdapterOptions",
    "GraphAdapter",
    "PygameStyle",
    "StreamlitContainer",
    "field_to_pil",
    "gradio_register_grapheditor",
    "image_to_pil",
    "layout_graph",
    "payload_to_pil",
    "pygame_register_grapheditor",
    "streamlit_register_grapheditor",
]
