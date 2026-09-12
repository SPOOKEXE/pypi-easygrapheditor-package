"""Subworkflow demo: combine nodes into one, run it, expand it back.

Builds number(2) + number(3) -> arithmetic -> readout, folds the first
three nodes into an "Adder" subworkflow (input-intermediate-output), runs
the folded graph (readout still 5.0), then expands back and re-runs.
Run: uv run easygrapheditor/examples/subflow_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_runner import run_demo
from easygrapheditor.engine import (
    Executor,
    Graph,
    combine_nodes,
    describe_subworkflow,
    expand_subworkflow_node,
)


def build() -> tuple[Graph, dict[str, str]]:
    """Build the flat demo graph (also used by view_demo.py)."""
    g = Graph()
    n1 = g.add_node("input.number", params={"value": 2.0}, pos=(0, 0))
    n2 = g.add_node("input.number", params={"value": 3.0}, pos=(0, 120))
    ad = g.add_node("maths.arithmetic", params={"operation": "add", "absolute": False}, pos=(240, 60))
    ro = g.add_node("output.readout", params={}, pos=(480, 60))
    g.add_link(n1.id, "out", ad.id, "a")
    g.add_link(n2.id, "out", ad.id, "b")
    g.add_link(ad.id, "out", ro.id, "value_in")
    return g, {"n1": n1.id, "n2": n2.id, "ad": ad.id, "ro": ro.id}


def readout_of(g: Graph, ro_id: str) -> float:
    ex = Executor(g)
    report = ex.run_blocking()
    assert report.ok(), {k: v.error for k, v in report.per_node.items()}
    return ex.outputs[ro_id]["value"]


def main() -> None:
    g, refs = build()
    print("flat readout:", readout_of(g, refs["ro"]))

    sub = combine_nodes(g, [refs["n1"], refs["n2"], refs["ad"]], "Adder")
    print("combined:", describe_subworkflow(sub))
    print("folded validate:", g.validate())
    print("folded readout:", readout_of(g, refs["ro"]))

    restored = expand_subworkflow_node(g, sub.id)
    print(f"expanded {len(restored)} nodes, validate:", g.validate())
    print("expanded readout:", readout_of(g, refs["ro"]))


if __name__ == "__main__":
    run_demo(build=build, cli_main=main, title="Subworkflow", description=__doc__ or "Subworkflow demo", script_path=__file__)
