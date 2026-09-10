"""Graph model: nodes, links, validation, (de)serialization. See editor.md §4.3."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .nodes import NODE_REGISTRY
from .types import can_connect

GRAPH_VERSION = 1


@dataclass
class NodeInstance:
    id: str
    type_id: str
    params: dict[str, Any] = field(default_factory=dict)
    pos: tuple[float, float] = (0.0, 0.0)
    collapsed: bool = False


@dataclass
class Link:
    from_node: str
    from_port: str
    to_node: str
    to_port: str


@dataclass
class ValidationError:
    node_id: str | None
    message: str


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeInstance] = {}
        self.links: list[Link] = []

    def add_node(self, type_id: str, params: dict[str, Any] | None = None,
                 pos: tuple[float, float] = (0.0, 0.0)) -> NodeInstance:
        nid = f"{type_id.split('.')[-1]}_{uuid.uuid4().hex[:6]}"
        inst = NodeInstance(id=nid, type_id=type_id, params=params or {}, pos=pos)
        self.nodes[nid] = inst
        return inst

    def add_link(self, from_node: str, from_port: str, to_node: str, to_port: str) -> Link:
        link = Link(from_node, from_port, to_node, to_port)
        self.links.append(link)
        return link

    def _port_dtype(self, node_id: str, port_key: str, direction: str) -> str | None:
        inst = self.nodes.get(node_id)
        if inst is None:
            return None
        ndef = NODE_REGISTRY.get(inst.type_id)
        if ndef is None:
            return None
        ports = ndef.outputs if direction == "out" else ndef.inputs
        for p in ports:
            if p.key == port_key:
                return p.dtype
        return None

    def validate(self) -> list[ValidationError]:
        errors: list[ValidationError] = []
        for nid, inst in self.nodes.items():
            if inst.type_id not in NODE_REGISTRY:
                errors.append(ValidationError(nid, f"Unknown node type: {inst.type_id}"))
        for link in self.links:
            if link.from_node not in self.nodes or link.to_node not in self.nodes:
                errors.append(ValidationError(None, f"Dangling link: {link}"))
                continue
            out_dt = self._port_dtype(link.from_node, link.from_port, "out")
            in_dt = self._port_dtype(link.to_node, link.to_port, "in")
            if out_dt is None or in_dt is None:
                errors.append(ValidationError(link.to_node, f"Unknown port in link: {link}"))
            elif not can_connect(out_dt, in_dt):
                errors.append(ValidationError(link.to_node, f"Type mismatch {out_dt} -> {in_dt}: {link}"))
        # Cycle detection (DFS).
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for link in self.links:
            if link.from_node in adj and link.to_node in adj:
                adj[link.from_node].append(link.to_node)
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(n: str, stack: list[str]) -> bool:
            visiting.add(n)
            for m in adj[n]:
                if m in visiting:
                    errors.append(ValidationError(m, f"Cycle detected: {' -> '.join([*stack, m])}"))
                    return True
                if m not in visited and dfs(m, [*stack, m]):
                    return True
            visiting.discard(n)
            visited.add(n)
            return False

        for nid in self.nodes:
            if nid not in visited:
                dfs(nid, [nid])
        return errors

    def topo_order(self) -> list[str]:
        indeg = {nid: 0 for nid in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for link in self.links:
            adj[link.from_node].append(link.to_node)
            indeg[link.to_node] += 1
        queue = [nid for nid, d in indeg.items() if d == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if len(order) != len(self.nodes):
            raise ValueError("Graph has a cycle; cannot topologically sort.")
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": GRAPH_VERSION,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "links": [asdict(link) for link in self.links],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Graph:
        g = cls()
        for nd in d.get("nodes", []):
            nd = dict(nd)
            nd["pos"] = tuple(nd.get("pos", (0.0, 0.0)))
            g.nodes[nd["id"]] = NodeInstance(**{k: nd[k] for k in ("id", "type_id", "params", "pos", "collapsed") if k in nd})
        for ld in d.get("links", []):
            g.links.append(Link(**ld))
        return g

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> Graph:
        return cls.from_dict(json.loads(s))
