"""Inspector stub (see editor.md §2.6)."""

from __future__ import annotations

from typing import Any

from ..engine.nodes import NODE_REGISTRY


def inspect_node(type_id: str) -> dict[str, Any]:
    ndef = NODE_REGISTRY.get(type_id)
    if ndef is None:
        return {"error": f"Unknown node: {type_id}"}
    return {
        "type_id": ndef.type_id,
        "title": ndef.title,
        "category": ndef.category,
        "description": ndef.description,
        "inputs": [(p.key, p.dtype) for p in ndef.inputs],
        "outputs": [(p.key, p.dtype) for p in ndef.outputs],
        "params": [(p.key, p.kind, p.default) for p in ndef.params],
    }
