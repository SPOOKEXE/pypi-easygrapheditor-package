"""Backend-neutral graph editing: add/remove/connect + canvas snapshot.

All mutations are plain functions over ``Graph`` so every UI backend
(gradio / streamlit / pygame) shares validation and behavior:

* links are type-checked (``can_connect``, subworkflow-aware),
* duplicates rejected,
* cycles rejected — except ``control.accumulate.next`` feedback links,
  which carry previous-iteration values by design (loop feedback must go
  through an accumulate node).
"""

from __future__ import annotations

from typing import Any

from ..engine.graph import Graph, Link, NodeInstance
from ..engine.loops import is_feedback_link
from ..engine.nodes import NODE_REGISTRY, get_node
from ..engine.types import can_connect


def node_type_choices() -> list[tuple[str, str]]:
    """(type_id, 'Title [Category]') sorted for picker dropdowns."""
    return sorted(
        ((n.type_id, f"{n.title} [{n.category}]") for n in NODE_REGISTRY.values()),
        key=lambda t: t[1].lower(),
    )


def add_node_at(graph: Graph, type_id: str, pos: tuple[float, float] = (0.0, 0.0), params: dict | None = None) -> NodeInstance:
    if type_id not in NODE_REGISTRY:
        raise ValueError(f"Unknown node type: {type_id}")
    ndef = NODE_REGISTRY[type_id]
    merged = {p.key: p.default for p in ndef.params}
    merged.update(params or {})
    return graph.add_node(type_id, params=merged, pos=pos)


def remove_node(graph: Graph, nid: str) -> None:
    """Delete a node, its incident links, and any loop memberships."""
    if nid not in graph.nodes:
        raise ValueError(f"Unknown node: {nid}")
    graph.links = [l for l in graph.links if l.from_node != nid and l.to_node != nid]
    for loop in graph.loops:
        if nid in loop.body:
            loop.body = [m for m in loop.body if m != nid]
    del graph.nodes[nid]


def _would_cycle(graph: Graph, extra: Link) -> bool:
    type_of = lambda nid: graph.nodes[nid].type_id if nid in graph.nodes else None
    if is_feedback_link(extra, type_of):
        return False
    probe = Graph()
    probe.nodes = dict(graph.nodes)
    probe.links = [*graph.links, extra]
    try:
        probe.topo_order()
        return False
    except ValueError:
        return True


def connect(graph: Graph, from_node: str, from_port: str, to_node: str, to_port: str) -> Link:
    """Link two ports. Raises ValueError with a human-readable reason."""
    if from_node not in graph.nodes:
        raise ValueError(f"Unknown node: {from_node}")
    if to_node not in graph.nodes:
        raise ValueError(f"Unknown node: {to_node}")
    out_dt = graph._port_dtype(from_node, from_port, "out")
    if out_dt is None:
        raise ValueError(f"{from_node} has no output port '{from_port}'")
    in_dt = graph._port_dtype(to_node, to_port, "in")
    if in_dt is None:
        raise ValueError(f"{to_node} has no input port '{to_port}'")
    if not can_connect(out_dt, in_dt):
        raise ValueError(f"Type mismatch: {out_dt} -> {in_dt}")
    if any(l.from_node == from_node and l.from_port == from_port and l.to_node == to_node and l.to_port == to_port for l in graph.links):
        raise ValueError("Link already exists")
    link = Link(from_node, from_port, to_node, to_port)
    if _would_cycle(graph, link):
        raise ValueError("Link would create a cycle (loop feedback must go through control.accumulate)")
    graph.links.append(link)
    return link


def disconnect(graph: Graph, from_node: str, from_port: str, to_node: str, to_port: str) -> bool:
    """Remove a link. Returns True when one was removed."""
    before = len(graph.links)
    graph.links = [l for l in graph.links if not (
        l.from_node == from_node and l.from_port == from_port and l.to_node == to_node and l.to_port == to_port)]
    return len(graph.links) < before


def compatible_inputs(graph: Graph, from_node: str, from_port: str) -> list[tuple[str, str]]:
    """All (node, input-port) pairs that ``from_node.from_port`` may legally feed."""
    out_dt = graph._port_dtype(from_node, from_port, "out")
    if out_dt is None:
        return []
    found: list[tuple[str, str]] = []
    for nid, inst in graph.nodes.items():
        if nid == from_node:
            continue
        if inst.type_id == "core.subworkflow":
            for m in inst.params.get("inputs", []):
                if can_connect(out_dt, str(m.get("dtype", "data.ANY"))):
                    found.append((nid, m["key"]))
            continue
        ndef = get_node(inst.type_id)
        for p in ndef.inputs if ndef else []:
            if can_connect(out_dt, p.dtype):
                found.append((nid, p.key))
    return found


def link_labels(graph: Graph) -> list[str]:
    return [f"{l.from_node}.{l.from_port} -> {l.to_node}.{l.to_port}" for l in graph.links]


# ------------------------------------------------------------- canvas image
def layout_boxes_px(
    graph: Graph, width: int, height: int, node_w: int = 190, node_h: int = 52, col_gap: int = 60, row_gap: int = 18,
    top: int = 12, left: int = 12,
) -> tuple[dict[str, tuple[int, int, int, int]], Any]:
    """Pixel boxes shared by the PIL snapshot (pygame wraps these as Rects)."""
    from .adapters import layout_graph

    pos, _depth = layout_graph(graph)
    max_col = max((int(x) for x, _ in pos.values()), default=0)
    max_row = max((int(y) for _, y in pos.values()), default=0)
    step_x = min(node_w + col_gap, max(width - 24, node_w) / max(max_col + 1, 1))
    step_y = min(node_h + row_gap, max(height - 24, node_h) / max(max_row + 1, 1))
    boxes = {
        nid: (int(left + x * step_x), int(top + y * step_y), node_w, node_h)
        for nid, (x, y) in pos.items()
    }
    return boxes, pos


def render_canvas_image(
    graph: Graph,
    report: Any = None,
    outputs: dict[str, dict[str, Any]] | None = None,
    width: int = 1100,
    height: int = 620,
) -> Any:
    """Server-side canvas snapshot (dark grid look) as a PIL image.

    Shows boxes with title/status/timing, edges, loop/subflow badges, and
    Field/Image thumbnails. Used by the gradio/streamlit live editors.
    """
    from PIL import Image as PILImage
    from PIL import ImageDraw

    from .adapters import payload_to_pil

    outputs = outputs or {}
    img = PILImage.new("RGB", (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(img)
    # grid dots
    for gx in range(0, width, 28):
        for gy in range(0, height, 28):
            draw.point((gx, gy), fill=(38, 42, 54))
    boxes, _pos = layout_boxes_px(graph, width, height)
    loops_of = {nid: loop.name for loop in graph.loops for nid in loop.body}

    def edge_color() -> tuple[int, int, int]:
        return (90, 140, 200)

    for link in graph.links:
        if link.from_node in boxes and link.to_node in boxes:
            ax, ay, aw, ah = boxes[link.from_node]
            bx, by, _bw, bh = boxes[link.to_node]
            draw.line([(ax + aw, ay + ah // 2), (bx, by + bh // 2)], fill=edge_color(), width=2)

    for nid, inst in graph.nodes.items():
        if nid not in boxes:
            continue
        ndef = get_node(inst.type_id)
        rep = report.per_node.get(nid) if report else None
        x, y, w, h = boxes[nid]
        border = (140, 140, 150) if rep is None else ((90, 200, 130) if rep.status in ("ok", "cached") else (230, 110, 110))
        draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=(45, 50, 62), outline=border, width=2)
        title = (ndef.title if ndef else inst.type_id)[:24]
        draw.text((x + 8, y + 5), title, fill=(230, 230, 235))
        sub = f"{rep.status} {rep.ms:.0f}ms" if rep else "not run"
        badges = f" [loop:{loops_of[nid]}]" if nid in loops_of else ""
        if inst.type_id == "core.subworkflow":
            badges += f" [sub:{len(inst.params.get('nodes', []))}]"
        draw.text((x + 8, y + 24), f"{sub}{badges}"[:34], fill=(150, 155, 165))
        # thumbnails under the box
        thumbs = [payload_to_pil(v) for v in outputs.get(nid, {}).values()]
        thumbs = [t for t in thumbs if t is not None][:3]
        for i, thumb in enumerate(thumbs):
            small = thumb.convert("RGB").resize((56, 56))
            img.paste(small, (x + i * 60, y + h + 4))
    return img


__all__ = [
    "add_node_at",
    "compatible_inputs",
    "connect",
    "disconnect",
    "layout_boxes_px",
    "link_labels",
    "node_type_choices",
    "remove_node",
    "render_canvas_image",
]
