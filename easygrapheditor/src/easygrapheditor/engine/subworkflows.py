"""Subworkflows: combine many nodes into one (and expand back).

A ``core.subworkflow`` node carries its subgraph in params::

    params = {
        "label": "Adder",
        "nodes": [NodeInstance dicts...],   # ids preserved, positions relative
        "links": [Link dicts...],           # internal links only
        "inputs": [{key, dtype, label, inner_node, inner_port}],
        "outputs": [{key, dtype, label, inner_node, inner_port}],
    }

Boundary links are rewired to the new node; the executor runs the inner
graph recursively (see execute.py, ``max_recursions`` default 1000).
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

from .graph import SUBWORKFLOW_TYPE_ID, Graph, Link, NodeInstance
from .nodes import NODE_REGISTRY

#: Default cap for subworkflow nesting depth (also the executor default).
DEFAULT_MAX_RECURSIONS = 1000


def _port_dtype_of(node: NodeInstance, port_key: str, direction: str) -> str:
    """Read a port dtype, including the explicit maps of nested subflows."""
    if node.type_id == SUBWORKFLOW_TYPE_ID:
        maps = node.params.get("inputs" if direction == "in" else "outputs", [])
        for mapping in maps:
            if mapping.get("key") == port_key:
                return str(mapping.get("dtype", "data.ANY"))
        return "data.ANY"
    ndef = NODE_REGISTRY.get(node.type_id)
    if ndef is None:
        return "data.ANY"
    for port in ndef.outputs if direction == "out" else ndef.inputs:
        if port.key == port_key:
            return port.dtype
    return "data.ANY"


def _port_required_of(node: NodeInstance, port_key: str) -> bool:
    if node.type_id == SUBWORKFLOW_TYPE_ID:
        return next(
            (
                bool(mapping.get("required", True))
                for mapping in node.params.get("inputs", [])
                if mapping.get("key") == port_key
            ),
            True,
        )
    ndef = NODE_REGISTRY.get(node.type_id)
    if ndef is None:
        return True
    return next((port.required for port in ndef.inputs if port.key == port_key), True)


def _unique(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def combine_nodes(
    graph: Graph, node_ids: list[str], label: str, new_pos: tuple[float, float] | None = None
) -> NodeInstance:
    """Fold ``node_ids`` into one subworkflow node. Returns the new node."""
    selected = set(node_ids)
    missing = [nid for nid in selected if nid not in graph.nodes]
    if missing:
        raise ValueError(f"combine_nodes: unknown nodes {missing}")
    if len(selected) < 1:
        raise ValueError("combine_nodes: select at least one node")
    loop_hits = [loop.name for loop in graph.loops if selected.intersection(loop.body)]
    if loop_hits:
        raise ValueError(
            f"combine_nodes cannot fold loop members ({', '.join(loop_hits)}); expand the loop body first"
        )

    inner_nodes = [copy.deepcopy(graph.nodes[nid]) for nid in node_ids]
    xs = [n.pos[0] for n in inner_nodes]
    ys = [n.pos[1] for n in inner_nodes]
    ox, oy = (min(xs), min(ys)) if xs else (0.0, 0.0)
    for node in inner_nodes:
        node.pos = (node.pos[0] - ox, node.pos[1] - oy)

    internal = [l for l in graph.links if l.from_node in selected and l.to_node in selected]
    incoming = [l for l in graph.links if l.from_node not in selected and l.to_node in selected]
    outgoing = [l for l in graph.links if l.from_node in selected and l.to_node not in selected]

    inputs: list[dict[str, Any]] = []
    taken: set[str] = set()
    for link in incoming:
        inner = graph.nodes[link.to_node]
        key = _unique(link.to_port, taken)
        taken.add(key)
        inputs.append(
            {
                "key": key,
                "dtype": _port_dtype_of(inner, link.to_port, "in"),
                "label": link.to_port,
                "required": _port_required_of(inner, link.to_port),
                "inner_node": link.to_node,
                "inner_port": link.to_port,
            }
        )
    outputs: list[dict[str, Any]] = []
    taken = set()
    for link in outgoing:
        inner = graph.nodes[link.from_node]
        key = _unique(link.from_port, taken)
        taken.add(key)
        outputs.append(
            {
                "key": key,
                "dtype": _port_dtype_of(inner, link.from_port, "out"),
                "label": link.from_port,
                "inner_node": link.from_node,
                "inner_port": link.from_port,
            }
        )

    # Retain group definitions so expand is a genuine visual round-trip even
    # when a group straddles the selection boundary.
    saved_groups = [asdict(group) for group in graph.groups if selected.intersection(group.nodes)]
    # Drop selected nodes + every touching link, then add the combined node.
    keep = [l for l in graph.links if l.from_node not in selected and l.to_node not in selected]
    graph.links = keep
    for nid in selected:
        del graph.nodes[nid]
    graph.prune_groups(selected)
    sub = graph.add_node(
        SUBWORKFLOW_TYPE_ID,
        params={
            "label": label,
            "nodes": [asdict(n) for n in inner_nodes],
            "links": [asdict(l) for l in internal],
            "inputs": inputs,
            "outputs": outputs,
            "groups": saved_groups,
        },
        pos=new_pos if new_pos is not None else (ox, oy),
    )
    # Rewire boundary links through the new node (match by inner endpoint).
    for link, mapping in zip(incoming, inputs):
        graph.add_link(link.from_node, link.from_port, sub.id, mapping["key"])
    for link, mapping in zip(outgoing, outputs):
        graph.add_link(sub.id, mapping["key"], link.to_node, link.to_port)
    return sub


def build_inner_graph(inst: NodeInstance) -> Graph:
    """Rebuild the inner graph of a subworkflow node (positions relative)."""
    if inst.type_id != SUBWORKFLOW_TYPE_ID:
        raise ValueError(f"node '{inst.id}' is not a subworkflow")
    inner = Graph()
    for node_dict in inst.params.get("nodes", []):
        node_dict = dict(node_dict)
        node_dict["pos"] = tuple(node_dict.get("pos", (0.0, 0.0)))
        inner.nodes[node_dict["id"]] = NodeInstance(
            **{
                k: node_dict[k]
                for k in ("id", "type_id", "params", "pos", "collapsed")
                if k in node_dict
            }
        )
    for link_dict in inst.params.get("links", []):
        inner.links.append(Link(**link_dict))
    return inner


def describe_subworkflow(inst: NodeInstance) -> dict[str, Any]:
    """Hover/inspector summary: label, counts, port maps, inner node titles."""
    from .nodes import get_node

    if inst.type_id != SUBWORKFLOW_TYPE_ID:
        raise ValueError(f"node '{inst.id}' is not a subworkflow")
    titles = []
    for node_dict in inst.params.get("nodes", []):
        ndef = get_node(node_dict.get("type_id", ""))
        titles.append(ndef.title if ndef else node_dict.get("type_id", "?"))
    return {
        "label": inst.params.get("label", "Subworkflow"),
        "nodes": len(inst.params.get("nodes", [])),
        "links": len(inst.params.get("links", [])),
        "inputs": [
            (m["key"], m["dtype"], m["inner_node"], m["inner_port"])
            for m in inst.params.get("inputs", [])
        ],
        "outputs": [
            (m["key"], m["dtype"], m["inner_node"], m["inner_port"])
            for m in inst.params.get("outputs", [])
        ],
        "titles": titles,
    }


def expand_subworkflow_node(graph: Graph, node_id: str) -> list[str]:
    """Inline a subworkflow node back into flat nodes. Returns restored ids."""
    inst = graph.nodes.get(node_id)
    if inst is None:
        raise ValueError(f"expand: unknown node '{node_id}'")
    if inst.type_id != SUBWORKFLOW_TYPE_ID:
        raise ValueError(f"expand: node '{node_id}' is not a subworkflow")

    # Re-home inner nodes (restore absolute positions, remap id clashes).
    remap: dict[str, str] = {}
    taken = set(graph.nodes) - {node_id}
    for node_dict in inst.params.get("nodes", []):
        old = node_dict["id"]
        new = old if old not in taken else _unique(old, taken)
        remap[old] = new
        taken.add(new)
    for node_dict in inst.params.get("nodes", []):
        restored = NodeInstance(
            id=remap[node_dict["id"]],
            type_id=node_dict["type_id"],
            params=copy.deepcopy(node_dict.get("params", {})),
            pos=(
                node_dict.get("pos", (0, 0))[0] + inst.pos[0],
                node_dict.get("pos", (0, 0))[1] + inst.pos[1],
            ),
            collapsed=bool(node_dict.get("collapsed", False)),
        )
        graph.nodes[restored.id] = restored
    for link_dict in inst.params.get("links", []):
        graph.links.append(
            Link(
                remap[link_dict["from_node"]],
                link_dict["from_port"],
                remap[link_dict["to_node"]],
                link_dict["to_port"],
            )
        )

    # Rewire boundary links back to inner endpoints.
    in_map = {
        m["key"]: (remap[m["inner_node"]], m["inner_port"]) for m in inst.params.get("inputs", [])
    }
    out_map = {
        m["key"]: (remap[m["inner_node"]], m["inner_port"]) for m in inst.params.get("outputs", [])
    }
    rewired: list[Link] = []
    for link in graph.links:
        if link.to_node == node_id and link.to_port in in_map:
            nid, port = in_map[link.to_port]
            rewired.append(Link(link.from_node, link.from_port, nid, port))
        elif link.from_node == node_id and link.from_port in out_map:
            nid, port = out_map[link.from_port]
            rewired.append(Link(nid, port, link.to_node, link.to_port))
        elif link.from_node != node_id and link.to_node != node_id:
            rewired.append(link)
    graph.links = rewired
    del graph.nodes[node_id]
    # Restore group membership captured when the subworkflow was created.
    saved_groups = inst.params.get("groups", [])
    existing = {group.name: group for group in graph.groups}
    for group_dict in saved_groups:
        name = str(group_dict.get("name", "group"))
        members = [remap.get(member, member) for member in group_dict.get("nodes", [])]
        members = [member for member in members if member in graph.nodes]
        if name in existing:
            group = existing[name]
            group.nodes = list(dict.fromkeys([*group.nodes, *members]))
        else:
            from .graph import GroupDef

            graph.groups.append(
                GroupDef(
                    name=name,
                    title=str(group_dict.get("title", "Group")),
                    nodes=members,
                    color=str(group_dict.get("color", "slate")),
                )
            )
    return [remap[d["id"]] for d in inst.params.get("nodes", [])]
