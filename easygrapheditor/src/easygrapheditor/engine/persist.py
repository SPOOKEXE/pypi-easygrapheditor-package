"""Persistence: *.ege.json save/load. See editor.md §6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .graph import GRAPH_VERSION, Graph


def save_graph(graph: Graph, path: str | Path, view: dict[str, Any] | None = None) -> Path:
    p = Path(path)
    payload = {**graph.to_dict(), "editor": {"view": view or {}}}
    p.write_text(__import__("json").dumps(payload, indent=2))
    return p


def load_graph(path: str | Path) -> tuple[Graph, dict[str, Any]]:
    p = Path(path)
    import json

    d = json.loads(p.read_text())
    assert d.get("version", GRAPH_VERSION) == GRAPH_VERSION, "Unsupported graph version"
    view = (d.get("editor") or {}).get("view", {})
    return Graph.from_dict(d), view


def import_comfy_workflow_api(d: dict[str, Any]) -> Graph:
    """Best-effort stub: wrap unknown Comfy nodes as errors (see editor.md §6)."""
    raise NotImplementedError("Comfy import is stubbed for M5.")
