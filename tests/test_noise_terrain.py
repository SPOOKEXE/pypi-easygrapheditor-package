"""Terrain pipeline tests: shapes, determinism, port keys."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine import Executor, Graph


def _tiny_chain(seed=7.0):
    g = Graph()
    a = g.add_node("generate.noise", params={"resolution": 16, "frequency": 2.0, "seed": seed, "octaves": 2, "gain": 0.5})
    b = g.add_node("generate.ridged", params={"resolution": 16, "frequency": 2.0, "seed": seed, "octaves": 2, "gain": 0.5})
    w = g.add_node("filter.domain_warp", params={"amount": 4.0})
    c = g.add_node("filter.combine", params={"amount": 0.5, "mode": "maximum"})
    e = g.add_node("simulate.erode", params={"thermal_passes": 2.0, "talus": 0.01, "spin_passes": 1.0})
    col = g.add_node("colour.colourise", params={"palette": "alpine", "sea_level": 0.32, "relief": False})
    g.add_link(a.id, "out", w.id, "field_in")
    g.add_link(b.id, "out", w.id, "warp_by")
    g.add_link(w.id, "out", c.id, "a")
    g.add_link(b.id, "out", c.id, "b")
    g.add_link(c.id, "out", e.id, "field_in")
    g.add_link(e.id, "out", col.id, "field_in")
    return g, e, col


def test_terrain_shapes_and_ranges():
    g, e, col = _tiny_chain()
    assert g.validate() == []
    ex = Executor(g)
    rep = ex.run_blocking()
    assert rep.ok(), {k: v.error for k, v in rep.per_node.items()}
    f = ex.outputs[e.id]["out"].data
    img = ex.outputs[col.id]["out"].data
    assert f.shape == (16, 16) and f.min() >= 0 and f.max() <= 1
    assert img.shape == (16, 16, 3) and img.dtype.name == "uint8"


def test_noise_deterministic_and_cached():
    g, e, _ = _tiny_chain(seed=3.0)
    ex = Executor(g)
    ex.run_blocking()
    first = ex.outputs[e.id]["out"].data.copy()
    rep2 = ex.run_blocking()
    assert all(r.cache_hit for r in rep2.per_node.values())
    assert (ex.outputs[e.id]["out"].data == first).all()


def test_readout_uses_port_key():
    g = Graph()
    n = g.add_node("input.number", params={"value": 5.0})
    r = g.add_node("output.readout", params={})
    g.add_link(n.id, "out", r.id, "value_in")
    ex = Executor(g)
    assert ex.run_blocking().ok()
    assert ex.outputs[r.id]["value"] == 5.0
