"""Shared canvas view-model stub."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine.execute import RunReport
from ..engine.graph import Graph


@dataclass
class Viewport:
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


@dataclass
class EditorState:
    graph: Graph
    selection: list[str] = field(default_factory=list)
    viewport: Viewport = field(default_factory=Viewport)
    run_report: RunReport | None = None
    errors: list[str] = field(default_factory=list)
    live: bool = True
    drag_offsets: dict[str, tuple[float, float]] = field(default_factory=dict)  # session node displacement (px)
