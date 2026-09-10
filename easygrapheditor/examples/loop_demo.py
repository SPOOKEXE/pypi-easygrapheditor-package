"""Loop demo: count 1 + accumulate until the EndConditionNode finishes.

Graph: number(1) + accumulate/current -> arithmetic -> accumulate/next,
with control.counter available and control.end_condition (threshold 5)
ending the loop. Final accumulate.current == 5.0.
Run: uv run easygrapheditor/examples/loop_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from easygrapheditor.engine import Cache, Executor, Graph


def build(target: float = 5.0) -> tuple[Graph, dict[str, str]]:
    """Build the demo graph. Returns (graph, refs) with node ids (also used by view_demo.py)."""
    g = Graph()
    one = g.add_node("input.number", params={"value": 1.0}, pos=(0, 40))
    counter = g.add_node("control.counter", params={}, pos=(0, 160))
    add = g.add_node("maths.arithmetic", params={"operation": "add", "absolute": False}, pos=(240, 80))
    acc = g.add_node("control.accumulate", params={"initial": 0.0}, pos=(480, 80))
    cond = g.add_node(
        "control.end_condition",
        params={"mode": "threshold", "target": target, "tolerance": 0.0},
        pos=(720, 80),
    )
    g.add_link(one.id, "out", add.id, "a")
    g.add_link(acc.id, "current", add.id, "b")
    g.add_link(add.id, "out", acc.id, "next")
    g.add_link(acc.id, "current", cond.id, "value")
    g.add_loop("count-to-target", [one.id, counter.id, add.id, acc.id, cond.id], cond.id)
    return g, {"acc": acc.id, "cond": cond.id, "counter": counter.id}


def main() -> None:
    g, refs = build()
    print("validate:", g.validate())
    cache = Cache()
    ex = Executor(g, cache=cache)
    report = ex.run_blocking()
    print(f"run {report.ms:.1f}ms ok={report.ok()} cache={cache.summary()}")
    for nid, rep in report.per_node.items():
        print(f"  {nid}: {rep.status} {rep.ms:.1f}ms {rep.error or ''}")
    print("final accumulate:", ex.outputs[refs["acc"]])
    assert report.ok() and ex.outputs[refs["acc"]]["current"] == 5.0


if __name__ == "__main__":
    main()
