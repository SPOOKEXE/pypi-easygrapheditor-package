"""Staged Arithmetic demo (mirrors plans/ 15-41-28 / 15-41-35).

Two Staged Task nodes -> Arithmetic(add). Run: uv run easygrapheditor/examples/staged_arithmetic_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from easygrapheditor.engine import Cache, Executor, Graph


def build() -> tuple[Graph, dict[str, str]]:
    """Build the demo graph. Returns (graph, refs) with node ids (also used by view_demo.py)."""
    g = Graph()
    left = g.add_node("simulate.staged_task", params={"seconds": 0.4, "label": "left"}, pos=(0, 0))
    right = g.add_node("simulate.staged_task", params={"seconds": 0.6, "label": "right"}, pos=(0, 120))
    arith = g.add_node("maths.arithmetic", params={"operation": "add", "absolute": False}, pos=(300, 60))
    g.add_link(left.id, "out", arith.id, "a")
    g.add_link(right.id, "out", arith.id, "b")
    return g, {"left": left.id, "right": right.id, "arith": arith.id}


def main() -> None:
    g, refs = build()
    print("validate:", g.validate())
    cache = Cache()
    ex = Executor(g, cache=cache)
    report = ex.run_blocking()
    print(f"run {report.ms:.1f}ms ok={report.ok()} cache={cache.summary()}")
    for nid, rep in report.per_node.items():
        print(f"  {nid}: {rep.status} {rep.ms:.1f}ms stages={rep.stages} {rep.error or ''}")
    print("arith out:", ex.outputs.get(refs["arith"]))

    # Second run should be cached (0.0ms per node).
    report2 = ex.run_blocking()
    print(f"second run cache={cache.summary()} ok={report2.ok()}")
    for nid, rep in report2.per_node.items():
        print(f"  {nid}: {rep.status} {rep.ms:.1f}ms")


if __name__ == "__main__":
    main()
