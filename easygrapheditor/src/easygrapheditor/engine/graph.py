"""Graph model: nodes, links, validation, (de)serialization. See editor.md §4.3."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .loops import DEFAULT_MAX_ITERATIONS, LoopDef, is_feedback_link, validate_loops
from .nodes import NODE_REGISTRY
from .types import can_connect

GRAPH_VERSION = 1
SUBWORKFLOW_TYPE_ID = "core.subworkflow"  # defined here to avoid import cycles


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


@dataclass
class GroupDef:
    """Visual group (ComfyUI-style): move together, titled box, no execution semantics."""

    name: str
    title: str
    nodes: list[str] = field(default_factory=list)
    color: str = "slate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupDef:
        return cls(
            name=str(d.get("name", "group")),
            title=str(d.get("title", "Group")),
            nodes=list(d.get("nodes", [])),
            color=str(d.get("color", "slate")),
        )


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeInstance] = {}
        self.links: list[Link] = []
        self.loops: list[LoopDef] = []
        self.groups: list[GroupDef] = []

    def add_loop(
        self,
        name: str,
        body: list[str],
        condition: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> LoopDef:
        loop = LoopDef(
            name=name, body=list(body), condition=condition, max_iterations=max_iterations
        )
        self.loops.append(loop)
        return loop

    def add_group(self, title: str, nodes: list[str], color: str = "slate") -> GroupDef:
        group = GroupDef(
            name=f"group_{len(self.groups) + 1}", title=title, nodes=list(nodes), color=color
        )
        self.groups.append(group)
        return group

    def group_of(self, nid: str) -> GroupDef | None:
        for group in self.groups:
            if nid in group.nodes:
                return group
        return None

    def prune_groups(self, removed: set[str]) -> None:
        """Drop deleted nodes from groups; drop groups left with <2 nodes."""
        for group in self.groups:
            group.nodes = [nid for nid in group.nodes if nid not in removed]
        self.groups = [g for g in self.groups if len(g.nodes) >= 2]

    def add_node(
        self,
        type_id: str,
        params: dict[str, Any] | None = None,
        pos: tuple[float, float] = (0.0, 0.0),
    ) -> NodeInstance:
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
        if inst.type_id == SUBWORKFLOW_TYPE_ID:
            maps = inst.params.get("inputs" if direction == "in" else "outputs", [])
            for mapping in maps:
                if mapping.get("key") == port_key:
                    return str(mapping.get("dtype", "data.ANY"))
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
            if inst.type_id == SUBWORKFLOW_TYPE_ID:
                errors.extend(self._validate_inner(nid, inst, depth=0))
            elif inst.type_id not in NODE_REGISTRY:
                errors.append(ValidationError(nid, f"Unknown node type: {inst.type_id}"))
        incoming: dict[tuple[str, str], int] = {}
        for link in self.links:
            if link.from_node not in self.nodes or link.to_node not in self.nodes:
                errors.append(ValidationError(None, f"Dangling link: {link}"))
                continue
            out_dt = self._port_dtype(link.from_node, link.from_port, "out")
            in_dt = self._port_dtype(link.to_node, link.to_port, "in")
            if out_dt is None or in_dt is None:
                errors.append(ValidationError(link.to_node, f"Unknown port in link: {link}"))
            elif not can_connect(out_dt, in_dt):
                errors.append(
                    ValidationError(link.to_node, f"Type mismatch {out_dt} -> {in_dt}: {link}")
                )
            endpoint = (link.to_node, link.to_port)
            incoming[endpoint] = incoming.get(endpoint, 0) + 1
        for (nid, port), count in incoming.items():
            if count > 1:
                errors.append(
                    ValidationError(
                        nid, f"Input '{port}' has {count} incoming links; fan-in is not supported"
                    )
                )
        for nid, inst in self.nodes.items():
            if inst.type_id == SUBWORKFLOW_TYPE_ID:
                for mapping in inst.params.get("inputs", []):
                    if mapping.get("required", True) and (nid, mapping.get("key")) not in incoming:
                        errors.append(
                            ValidationError(
                                nid,
                                f"Required subworkflow input '{mapping.get('key')}' is not connected",
                            )
                        )
                continue
            ndef = NODE_REGISTRY.get(inst.type_id)
            if ndef is None:
                continue
            for port in ndef.inputs:
                if port.required and (nid, port.key) not in incoming:
                    errors.append(
                        ValidationError(nid, f"Required input '{port.key}' is not connected")
                    )
        for group in self.groups:
            for nid in group.nodes:
                if nid not in self.nodes:
                    errors.append(
                        ValidationError(
                            nid, f"group '{group.name}' references missing node '{nid}'"
                        )
                    )
        # Cycle detection (DFS). Feedback links (accumulate.next) are exempt:
        # they carry previous-iteration values, never current-pass data.
        type_of = lambda nid: self.nodes[nid].type_id if nid in self.nodes else None
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for link in self.links:
            if (
                link.from_node in adj
                and link.to_node in adj
                and not is_feedback_link(link, type_of)
            ):
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
        errors.extend(
            validate_loops(self, lambda nid: self.nodes[nid].type_id if nid in self.nodes else None)
        )
        return errors

    def _validate_inner(
        self, outer_id: str, inst: NodeInstance, depth: int
    ) -> list[ValidationError]:
        """Recursively validate a subworkflow's inner nodes/links (depth-guarded)."""
        errors: list[ValidationError] = []
        if depth > 64:
            return [ValidationError(outer_id, "subworkflow nesting too deep to validate")]
        inner_nodes = {n["id"]: n for n in inst.params.get("nodes", []) if "id" in n}

        def inner_dtype(node_dict: dict, port_key: str, direction: str) -> str | None:
            if node_dict.get("type_id") == SUBWORKFLOW_TYPE_ID:
                maps = node_dict.get("params", {}).get(
                    "inputs" if direction == "in" else "outputs", []
                )
                for mapping in maps:
                    if mapping.get("key") == port_key:
                        return str(mapping.get("dtype", "data.ANY"))
                return None
            ndef = NODE_REGISTRY.get(node_dict.get("type_id", ""))
            if ndef is None:
                return None
            for port in ndef.outputs if direction == "out" else ndef.inputs:
                if port.key == port_key:
                    return port.dtype
            return None

        def inner_inputs(node_dict: dict) -> list[tuple[str, bool]]:
            """Required input contract for ordinary and nested subflow nodes."""
            if node_dict.get("type_id") == SUBWORKFLOW_TYPE_ID:
                return [
                    (str(mapping.get("key", "")), bool(mapping.get("required", True)))
                    for mapping in node_dict.get("params", {}).get("inputs", [])
                ]
            ndef = NODE_REGISTRY.get(node_dict.get("type_id", ""))
            return [] if ndef is None else [(port.key, port.required) for port in ndef.inputs]

        for inner_id, node_dict in inner_nodes.items():
            tid = node_dict.get("type_id", "")
            if tid == SUBWORKFLOW_TYPE_ID:
                nested = NodeInstance(
                    id=inner_id,
                    type_id=tid,
                    params=node_dict.get("params", {}),
                    pos=tuple(node_dict.get("pos", (0.0, 0.0))),
                )
                errors.extend(self._validate_inner(f"{outer_id}/{inner_id}", nested, depth + 1))
            elif tid not in NODE_REGISTRY:
                errors.append(
                    ValidationError(
                        outer_id, f"subworkflow '{outer_id}' has unknown inner type: {tid}"
                    )
                )
        for link_dict in inst.params.get("links", []):
            src, dst = (
                inner_nodes.get(link_dict.get("from_node", "")),
                inner_nodes.get(link_dict.get("to_node", "")),
            )
            if src is None or dst is None:
                errors.append(
                    ValidationError(
                        outer_id, f"subworkflow '{outer_id}' has dangling inner link: {link_dict}"
                    )
                )
                continue
            out_dt = inner_dtype(src, link_dict.get("from_port", ""), "out")
            in_dt = inner_dtype(dst, link_dict.get("to_port", ""), "in")
            if out_dt is None or in_dt is None:
                errors.append(
                    ValidationError(
                        outer_id, f"subworkflow '{outer_id}' has unknown inner port: {link_dict}"
                    )
                )
            elif not can_connect(out_dt, in_dt):
                errors.append(
                    ValidationError(
                        outer_id, f"subworkflow '{outer_id}' type mismatch {out_dt} -> {in_dt}"
                    )
                )
        mapped_inputs = {
            (mapping.get("inner_node"), mapping.get("inner_port"))
            for mapping in inst.params.get("inputs", [])
        }
        incoming: dict[tuple[str, str], int] = {}
        for link_dict in inst.params.get("links", []):
            endpoint = (link_dict.get("to_node"), link_dict.get("to_port"))
            incoming[endpoint] = incoming.get(endpoint, 0) + 1
        for endpoint, count in incoming.items():
            if count > 1:
                errors.append(
                    ValidationError(
                        outer_id,
                        f"subworkflow '{outer_id}' input '{endpoint[1]}' has duplicate fan-in",
                    )
                )
        for inner_id, node_dict in inner_nodes.items():
            for port_key, required in inner_inputs(node_dict):
                endpoint = (inner_id, port_key)
                if required and endpoint not in incoming and endpoint not in mapped_inputs:
                    errors.append(
                        ValidationError(
                            outer_id,
                            f"subworkflow '{outer_id}' required input '{inner_id}.{port_key}' is not connected",
                        )
                    )
        # Inner cycles are not visible to the outer graph's DFS.
        adj: dict[str, list[str]] = {nid: [] for nid in inner_nodes}
        for link_dict in inst.params.get("links", []):
            src, dst = link_dict.get("from_node"), link_dict.get("to_node")
            if src in adj and dst in adj:

                class _Link:
                    from_node = src
                    to_node = dst
                    from_port = link_dict.get("from_port")
                    to_port = link_dict.get("to_port")

                if not is_feedback_link(
                    _Link(), lambda node_id: inner_nodes[node_id].get("type_id")
                ):
                    adj[src].append(dst)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(nid: str) -> bool:
            visiting.add(nid)
            for nxt in adj[nid]:
                if nxt in visiting:
                    errors.append(
                        ValidationError(outer_id, f"subworkflow '{outer_id}' has an inner cycle")
                    )
                    return True
                if nxt not in visited and visit(nxt):
                    return True
            visiting.remove(nid)
            visited.add(nid)
            return False

        for inner_id in inner_nodes:
            if inner_id not in visited:
                visit(inner_id)
        # Port maps must point at real inner endpoints and faithfully retain
        # the dtype when a nested subflow is folded into another one.
        for direction, mappings in (
            ("in", inst.params.get("inputs", [])),
            ("out", inst.params.get("outputs", [])),
        ):
            for mapping in mappings:
                inner_node = mapping.get("inner_node")
                if inner_node not in inner_nodes:
                    errors.append(
                        ValidationError(
                            outer_id, f"subworkflow '{outer_id}' maps missing node: {mapping}"
                        )
                    )
                    continue
                actual = inner_dtype(
                    inner_nodes[inner_node], mapping.get("inner_port", ""), direction
                )
                if actual is None:
                    errors.append(
                        ValidationError(
                            outer_id, f"subworkflow '{outer_id}' maps missing port: {mapping}"
                        )
                    )
                elif str(mapping.get("dtype", actual)) != actual:
                    errors.append(
                        ValidationError(
                            outer_id,
                            f"subworkflow '{outer_id}' map dtype {mapping.get('dtype')} does not match {actual}",
                        )
                    )
        return errors

    def topo_order(self) -> list[str]:
        type_of = lambda nid: self.nodes[nid].type_id if nid in self.nodes else None
        indeg = {nid: 0 for nid in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for link in self.links:
            if is_feedback_link(link, type_of):
                continue
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
            "loops": [loop.to_dict() for loop in self.loops],
            "groups": [group.to_dict() for group in self.groups],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Graph:
        g = cls()
        for nd in d.get("nodes", []):
            nd = dict(nd)
            nd["pos"] = tuple(nd.get("pos", (0.0, 0.0)))
            g.nodes[nd["id"]] = NodeInstance(
                **{k: nd[k] for k in ("id", "type_id", "params", "pos", "collapsed") if k in nd}
            )
        for ld in d.get("links", []):
            g.links.append(Link(**ld))
        for loop_dict in d.get("loops", []):
            g.loops.append(LoopDef.from_dict(loop_dict))
        for group_dict in d.get("groups", []):
            g.groups.append(GroupDef.from_dict(group_dict))
        return g

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> Graph:
        return cls.from_dict(json.loads(s))
