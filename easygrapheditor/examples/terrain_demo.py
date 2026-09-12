"""Headless terrain graph demo with deterministic cache verification.

Run: ``uv run python easygrapheditor/examples/terrain_demo.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_runner import run_demo
from easygrapheditor.engine import Cache, Executor
from normterrain_demo import build


def main() -> None:
    graph, refs = build()
    errors = graph.validate()
    assert not errors, errors

    cache = Cache()
    executor = Executor(graph, cache=cache)
    first = executor.run_blocking()
    assert first.ok(), first.per_node
    colour = executor.outputs[refs["colour"]]["out"].data
    print(
        f"terrain {colour.shape[1]}x{colour.shape[0]} "
        f"run={first.ms:.1f}ms result={first.result_hash}"
    )

    second = executor.run_blocking()
    assert second.ok() and all(report.cache_hit for report in second.per_node.values())
    print(f"second run cache={cache.summary()} result={second.result_hash}")
    assert first.result_hash == second.result_hash


if __name__ == "__main__":
    run_demo(build=build, cli_main=main, title="Terrain", description=__doc__ or "Terrain demo", script_path=__file__)
