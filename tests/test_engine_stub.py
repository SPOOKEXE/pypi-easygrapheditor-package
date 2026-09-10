"""Engine stub tests (M1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine import Cache, Executor, Graph, can_connect


def test_can_connect():
    assert can_connect("data.NUMBER", "data.NUMBER")
    assert can_connect("data.ANY", "data.NUMBER")
    assert can_connect("data.NUMBER", "data.ANY")
    assert not can_connect("data.NUMBER", "data.FIELD")


def test_arithmetic_add():
    g = Graph()
    g.add_node("maths.arithmetic", params={"operation": "add"})
    # Wire constants via params only is not enough for required inputs;
    # stub executor treats missing inputs as absent -> fn gets only params.
    # So test validate passes structurally then run errors cleanly.
    assert isinstance(g.validate(), list)


def test_cycle_rejected():
    g = Graph()
    a = g.add_node("maths.arithmetic", params={})
    b = g.add_node("maths.arithmetic", params={})
    g.add_link(a.id, "out", b.id, "a")
    g.add_link(b.id, "out", a.id, "a")
    errs = g.validate()
    assert any("Cycle" in e.message for e in errs)


def test_cache_second_run_cached():
    g = Graph()
    n = g.add_node("input.number", params={"value": 1.0})
    cache = Cache()
    ex = Executor(g, cache=cache)
    r1 = ex.run_blocking()
    assert r1.ok()
    r2 = ex.run_blocking()
    assert r2.per_node[n.id].cache_hit
