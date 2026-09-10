"""Loop support: repeat a node body until an EndConditionNode finishes.

Model (cycle-free by construction — validation still forbids link cycles):
* A loop declares ``body`` (node ids run each iteration) + ``condition``
  (an ``control.end_condition`` node id whose ``done`` output ends the loop).
* Loop-carried state uses ``control.accumulate`` (previous-iteration value)
  and ``control.counter`` (iteration index) nodes inside the body.
* ``max_iterations`` caps every loop (default 1000); hitting it is an error.

Links from body nodes to the outside world carry final-iteration values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_MAX_ITERATIONS = 1000
END_CONDITION_TYPE_ID = "control.end_condition"

#: Links into this (node type, port) carry previous-iteration values, so they
#: are exempt from cycle detection and ordering everywhere. (Literal here —
#: importing nodes_control from engine modules would risk import cycles.)
FEEDBACK_NODE_TYPE = "control.accumulate"
FEEDBACK_PORT = "next"


def is_feedback_link(link: Any, type_of: Any) -> bool:
    """True if ``link`` feeds an accumulate node (loop-carried state)."""
    try:
        return link.to_port == FEEDBACK_PORT and type_of(link.to_node) == FEEDBACK_NODE_TYPE
    except Exception:  # noqa: BLE001 - missing nodes simply aren't feedback
        return False


@dataclass
class LoopDef:
    name: str
    body: list[str] = field(default_factory=list)
    condition: str = ""
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LoopDef:
        return cls(
            name=str(d.get("name", "loop")),
            body=list(d.get("body", [])),
            condition=str(d.get("condition", "")),
            max_iterations=int(d.get("max_iterations", DEFAULT_MAX_ITERATIONS)),
        )


def validate_loops(graph: Any, node_type_of: Any) -> list[Any]:
    """Check loop defs against a graph. ``node_type_of(nid)`` -> type_id|None."""
    from .graph import ValidationError

    errors: list[ValidationError] = []
    seen: dict[str, str] = {}  # node id -> loop name (bodies must not overlap in v1)
    for loop in graph.loops:
        if not loop.body:
            errors.append(ValidationError(None, f"loop '{loop.name}' has an empty body"))
        if loop.max_iterations < 1:
            errors.append(ValidationError(None, f"loop '{loop.name}' needs max_iterations >= 1"))
        for nid in loop.body:
            if node_type_of(nid) is None:
                errors.append(ValidationError(nid, f"loop '{loop.name}' references unknown node '{nid}'"))
            elif nid in seen:
                errors.append(ValidationError(nid, f"node '{nid}' in two loops ('{seen[nid]}', '{loop.name}'): nested loops are v1-unsupported"))
            else:
                seen[nid] = loop.name
        if not loop.condition:
            errors.append(ValidationError(None, f"loop '{loop.name}' has no condition node"))
        elif loop.condition not in loop.body:
            errors.append(ValidationError(loop.condition, f"loop '{loop.name}' condition '{loop.condition}' must be inside the body"))
        elif node_type_of(loop.condition) != END_CONDITION_TYPE_ID:
            errors.append(
                ValidationError(
                    loop.condition,
                    f"loop '{loop.name}' condition must be a {END_CONDITION_TYPE_ID} node",
                )
            )
    return errors


def loop_body_order(node_ids: list[str], links: list[Any], type_of: Any = None) -> list[str]:
    """Topo order restricted to the loop body (internal links only).

    Feedback links (``control.accumulate.next``) are not ordering constraints:
    they always read the previous iteration.
    """
    body = set(node_ids)
    type_of = type_of or (lambda _nid: None)
    indeg = {nid: 0 for nid in node_ids}
    adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for link in links:
        if link.from_node in body and link.to_node in body and not is_feedback_link(link, type_of):
            adj[link.from_node].append(link.to_node)
            indeg[link.to_node] += 1
    queue = [nid for nid in node_ids if indeg[nid] == 0]
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for nxt in adj[nid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(node_ids):
        raise ValueError("Loop body has a cycle; bodies must stay acyclic (use control.accumulate for feedback).")
    return order
