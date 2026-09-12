"""Inspector and Types panel view models."""

from __future__ import annotations

from typing import Any

from ..engine.nodes import NODE_REGISTRY
from ..engine.types import list_types


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


def inspect_types() -> list[dict[str, Any]]:
    """Type rows with producer and consumer counts for the Types panel."""
    rows: list[dict[str, Any]] = []
    for dtype in list_types():
        producers = [node.title for node in NODE_REGISTRY.values() if any(p.dtype == dtype.id for p in node.outputs)]
        consumers = [node.title for node in NODE_REGISTRY.values() if any(p.dtype == dtype.id for p in node.inputs)]
        rows.append({"id": dtype.id, "label": dtype.label, "description": dtype.description,
                     "preview": dtype.preview, "producers": producers, "consumers": consumers,
                     "producer_count": len(producers), "consumer_count": len(consumers)})
    return rows
