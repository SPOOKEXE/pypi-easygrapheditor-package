"""Searchable library view models shared by all frontends."""

from __future__ import annotations

from ..engine.nodes import NODE_REGISTRY, NodeDef


def categories() -> dict[str, list[str]]:
    """Category tree suitable for a narrow left library rail."""
    tree: dict[str, list[str]] = {}
    for node in NODE_REGISTRY.values():
        tree.setdefault(node.category, []).append(node.type_id)
    return {key: sorted(value) for key, value in sorted(tree.items())}


def node_detail(type_id: str) -> dict:
    """Library detail pane data, including port tables and badges."""
    node = NODE_REGISTRY.get(type_id)
    if node is None:
        return {"error": f"Unknown node: {type_id}"}
    return {
        "type_id": node.type_id,
        "title": node.title,
        "category": node.category,
        "description": node.description,
        "badges": list(node.badges),
        "inputs": [{"key": p.key, "label": p.label, "type": p.dtype} for p in node.inputs],
        "outputs": [{"key": p.key, "label": p.label, "type": p.dtype} for p in node.outputs],
    }


def search_nodes(query: str = "", category: str | None = None, allowed: set[str] | None = None) -> list[NodeDef]:
    q = query.lower().strip()
    out: list[NodeDef] = []
    for ndef in NODE_REGISTRY.values():
        if allowed is not None and ndef.type_id not in allowed:
            continue
        if category and ndef.category != category:
            continue
        haystack = " ".join((ndef.title, ndef.type_id, ndef.category, ndef.description, *ndef.badges)).lower()
        # Token matching is a modest fuzzy search that keeps zero dependencies.
        if not q or all(token in haystack for token in q.split()):
            out.append(ndef)
    return sorted(out, key=lambda n: (n.category, n.title))
