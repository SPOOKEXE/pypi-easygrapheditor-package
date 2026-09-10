"""Library search stub (see editor.md §2.5)."""

from __future__ import annotations

from ..engine.nodes import NODE_REGISTRY, NodeDef


def search_nodes(query: str = "", category: str | None = None) -> list[NodeDef]:
    q = query.lower().strip()
    out: list[NodeDef] = []
    for ndef in NODE_REGISTRY.values():
        if category and ndef.category != category:
            continue
        if not q or q in ndef.title.lower() or q in ndef.type_id.lower() or q in ndef.description.lower():
            out.append(ndef)
    return sorted(out, key=lambda n: (n.category, n.title))
