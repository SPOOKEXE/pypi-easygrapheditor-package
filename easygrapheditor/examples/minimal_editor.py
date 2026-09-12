"""Minimal editor: constant Number -> Readout. Run: uv run easygrapheditor/examples/minimal_editor.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_runner import run_demo
from easygrapheditor.engine import Graph

import easygrapheditor as ege


def build() -> Graph:
    """Build the demo graph (also used by view_demo.py / UI adapters)."""
    g = Graph()
    g.add_node("input.number", params={"value": 2.0}, pos=(0, 0))
    g.add_node("simulate.staged_task", params={"seconds": 0.2, "label": "left"}, pos=(200, 0))
    return g


def main() -> None:
    print(f"easygrapheditor {ege.__version__}")
    print("Registered nodes:")
    for n in ege.list_nodes():
        print(f"  - {n.type_id} [{n.category}]")

    ed = ege.Editor(title="Minimal")
    ed.graph = build()
    print("validate:", ed.graph.validate())
    report = ed.run_headless()
    for nid, rep in report.per_node.items():
        print(f"{nid}: {rep.status} {rep.ms:.1f}ms out={ed.graph and ''}{rep.error or ''}")
    print("outputs:", ege.Executor(ed.graph).outputs if hasattr(ege.Executor(ed.graph), "outputs") else "")


if __name__ == "__main__":
    run_demo(build=build, cli_main=main, title="Minimal", description=__doc__ or "Minimal demo", script_path=__file__)
