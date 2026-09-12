"""Backend-neutral graph editing: add/remove/connect + canvas snapshot.

All mutations are plain functions over ``Graph`` so every UI backend
(gradio / streamlit / pygame) shares validation and behavior:

* links are type-checked (``can_connect``, subworkflow-aware),
* duplicates rejected,
* cycles rejected, except ``control.accumulate.next`` feedback links,
  which carry previous-iteration values by design (loop feedback must go
  through an accumulate node).
"""

from __future__ import annotations

import copy
from typing import Any

from ..engine.graph import Graph, Link, NodeInstance
from ..engine.loops import is_feedback_link
from ..engine.nodes import NODE_REGISTRY, get_node
from ..engine.types import can_connect


def node_type_choices(allowed: set[str] | None = None) -> list[tuple[str, str]]:
    """(type_id, 'Title [Category]') sorted for picker dropdowns."""
    return sorted(
        ((n.type_id, f"{n.title} [{n.category}]") for n in NODE_REGISTRY.values() if allowed is None or n.type_id in allowed),
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
    """Delete a node, its incident links, and any loop/group memberships."""
    if nid not in graph.nodes:
        raise ValueError(f"Unknown node: {nid}")
    graph.links = [l for l in graph.links if l.from_node != nid and l.to_node != nid]
    for loop in graph.loops:
        if nid in loop.body:
            loop.body = [m for m in loop.body if m != nid]
    graph.prune_groups({nid})
    del graph.nodes[nid]


def remove_nodes(graph: Graph, nids: list[str]) -> None:
    """Delete several nodes (shared prune pass)."""
    missing = [nid for nid in nids if nid not in graph.nodes]
    if missing:
        raise ValueError(f"Unknown nodes: {missing}")
    gone = set(nids)
    graph.links = [l for l in graph.links if l.from_node not in gone and l.to_node not in gone]
    for loop in graph.loops:
        loop.body = [m for m in loop.body if m not in gone]
    graph.prune_groups(gone)
    for nid in nids:
        del graph.nodes[nid]


def group_nodes(graph: Graph, nids: list[str], title: str, color: str = "slate") -> Any:
    """Create a visual group over ``nids`` (2+ members, like ComfyUI groups)."""

    missing = [nid for nid in nids if nid not in graph.nodes]
    if missing:
        raise ValueError(f"Unknown nodes: {missing}")
    if len(set(nids)) < 2:
        raise ValueError("Select at least 2 nodes to group")
    if color not in GROUP_COLORS:
        raise ValueError(f"Unknown group color '{color}' (choose: {', '.join(sorted(GROUP_COLORS))})")
    already = {nid: g.name for g in graph.groups for nid in g.nodes if nid in set(nids)}
    if already:
        raise ValueError(f"Nodes already grouped: {already}")
    return graph.add_group(title, list(dict.fromkeys(nids)), color)


def ungroup(graph: Graph, name: str) -> list[str]:
    """Dissolve a group; returns the freed node ids."""
    for i, group in enumerate(graph.groups):
        if group.name == name:
            freed = list(group.nodes)
            del graph.groups[i]
            return freed
    raise ValueError(f"Unknown group: {name}")


def copy_selection(graph: Graph, nids: list[str]) -> dict[str, Any]:
    """Serialize nodes + internal links for clipboard paste (loops not carried)."""
    from dataclasses import asdict

    missing = [nid for nid in nids if nid not in graph.nodes]
    if missing:
        raise ValueError(f"Unknown nodes: {missing}")
    sel = set(nids)
    return {
        "nodes": [asdict(graph.nodes[nid]) for nid in nids],
        "links": [asdict(l) for l in graph.links if l.from_node in sel and l.to_node in sel],
    }


def paste_clipboard(graph: Graph, clip: dict[str, Any], delta: tuple[float, float] = (40.0, 40.0)) -> list[str]:
    """Paste a clipboard: fresh ids, remapped internal links, offset positions."""
    remap: dict[str, str] = {}
    for node_dict in clip.get("nodes", []):
        inst = graph.add_node(
            node_dict["type_id"],
            params=copy.deepcopy(node_dict.get("params", {})),
            pos=(node_dict.get("pos", (0.0, 0.0))[0] + delta[0], node_dict.get("pos", (0.0, 0.0))[1] + delta[1]),
        )
        remap[node_dict["id"]] = inst.id
    for link_dict in clip.get("links", []):
        if link_dict["from_node"] in remap and link_dict["to_node"] in remap:
            graph.links.append(Link(remap[link_dict["from_node"]], link_dict["from_port"], remap[link_dict["to_node"]], link_dict["to_port"]))
    return [remap[d["id"]] for d in clip.get("nodes", []) if d["id"] in remap]


#: Group color names -> RGB (shared by PIL snapshot + pygame).
GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    "slate": (120, 135, 160),
    "blue": (90, 140, 200),
    "green": (90, 200, 130),
    "amber": (220, 175, 90),
    "red": (220, 120, 120),
    "purple": (170, 130, 220),
    "teal": (90, 200, 190),
}


def port_value_preview(value: Any, limit: int = 80) -> str:
    """One-line human preview of a live port value (for tooltips/inspectors)."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    if isinstance(value, str):
        text = value.replace("\n", " ")
        return text[:limit] + ("…" if len(text) > limit else "") or "none"
    dataclass_data = getattr(value, "data", None)
    if dataclass_data is not None:
        import numpy as np

        if isinstance(dataclass_data, np.ndarray):
            kind = type(value).__name__.lower()
            try:
                mean = float(dataclass_data.mean())
            except (TypeError, ValueError):
                mean = None
            extra = f" μ={mean:.3f}" if mean is not None else ""
            return f"{kind} {tuple(dataclass_data.shape)}{extra}"
    if isinstance(value, dict):
        keys = ",".join(list(value)[:4])
        return f"{{{keys}}}{'…' if len(value) > 4 else ''}"
    text = str(value).replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


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
    positioned = any(inst.pos != (0.0, 0.0) for inst in graph.nodes.values())
    if positioned:
        boxes = {nid: (int(left + x), int(top + y), node_w, node_h) for nid, (x, y) in pos.items()}
        return boxes, pos
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
    img = PILImage.new("RGB", (width, height), (0, 0, 0))
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

    # group boxes behind member nodes
    for i, group in enumerate(graph.groups):
        members = [boxes[nid] for nid in group.nodes if nid in boxes]
        if not members:
            continue
        gx0 = min(b[0] for b in members) - 14
        gy0 = min(b[1] for b in members) - 30
        gx1 = max(b[0] + b[2] for b in members) + 14
        gy1 = max(b[1] + b[3] for b in members) + 70  # room for thumbnails
        color = GROUP_COLORS.get(group.color, GROUP_COLORS["slate"])
        draw.rounded_rectangle([gx0, gy0, gx1, gy1], radius=10, outline=color, width=2)
        draw.text((gx0 + 10, gy0 + 6), group.title[:36], fill=color)

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
    "GROUP_COLORS",
    "add_node_at",
    "compatible_inputs",
    "connect",
    "copy_selection",
    "disconnect",
    "group_nodes",
    "layout_boxes_px",
    "link_labels",
    "node_type_choices",
    "paste_clipboard",
    "port_value_preview",
    "remove_node",
    "remove_nodes",
    "render_canvas_image",
    "ungroup",
]
