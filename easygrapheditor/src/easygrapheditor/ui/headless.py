"""Headless runner (no UI, for tests/CI)."""

from __future__ import annotations

from ..engine.cache import Cache
from ..engine.execute import Executor, RunReport
from ..engine.graph import Graph


def run_headless(graph: Graph, cache: Cache | None = None) -> RunReport:
    return Executor(graph, cache=cache).run_blocking()
