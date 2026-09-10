"""UI backend protocol. See editor.md §5."""

from __future__ import annotations

from typing import Protocol

from ..engine.graph import Graph


class EditorBackend(Protocol):
    def render(self, graph: Graph) -> None:
        ...

    def run(self) -> None:
        ...
