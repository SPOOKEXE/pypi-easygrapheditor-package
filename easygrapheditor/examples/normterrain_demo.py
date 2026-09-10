"""Normalized terrain demo: Noise -> Erode -> Terrace -> Colourise.

Same engine as noiseterrain but the eroded field is min-max normalized to
0..1 before preview, so outputs are comparable across seeds. Exercises the
terrace/threshold filter nodes.
Run: uv run easygrapheditor/examples/normterrain_demo.py
Outputs PNGs + graph JSON into easygrapheditor/examples/output/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from easygrapheditor.engine import Cache, Executor, Graph
from PIL import Image as PILImage

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def build() -> tuple[Graph, dict[str, str]]:
    """Build the demo graph. Returns (graph, refs) with node ids (also used by view_demo.py)."""
    g = Graph()
    n_noise = g.add_node("generate.noise", params={"resolution": 64, "frequency": 6.0, "seed": 11.0, "octaves": 4, "gain": 0.5}, pos=(0, 0))
    n_erode = g.add_node("simulate.erode", params={"thermal_passes": 10.0, "talus": 0.02, "spin_passes": 5.0}, pos=(240, 0))
    n_terrace = g.add_node("filter.terrace", params={"steps": 6.0}, pos=(480, 0))
    n_colour = g.add_node("colour.colourise", params={"palette": "alpine", "sea_level": 0.4, "relief": True}, pos=(720, 0))
    n_read = g.add_node("output.readout", params={}, pos=(960, 0))

    g.add_link(n_noise.id, "out", n_erode.id, "field_in")
    g.add_link(n_erode.id, "out", n_terrace.id, "field_in")
    g.add_link(n_terrace.id, "out", n_colour.id, "field_in")
    g.add_link(n_colour.id, "out", n_read.id, "value_in")
    return g, {"erode": n_erode.id, "terrace": n_terrace.id, "colour": n_colour.id, "readout": n_read.id}


def main() -> None:
    g, refs = build()
    errs = g.validate()
    print("validate:", errs)
    assert not errs, errs

    cache = Cache()
    ex = Executor(g, cache=cache)
    report = ex.run_blocking()
    print(f"run {report.ms:.1f}ms ok={report.ok()} cache={cache.summary()}")
    for nid, rep in report.per_node.items():
        print(f"  {nid}: {rep.status} {rep.ms:.1f}ms {rep.error or ''}")

    raw = ex.outputs[refs["erode"]]["out"].data
    normed = _normalize(raw)
    terraced = ex.outputs[refs["terrace"]]["out"].data
    colour = ex.outputs[refs["colour"]]["out"].data
    print(f"eroded  min={raw.min():.3f} max={raw.max():.3f} -> normalized min={normed.min():.3f} max={normed.max():.3f}")
    print(f"terraced levels={len(np.unique(terraced))} readout={ex.outputs[refs['readout']]['value']:.3f}")

    p_norm = OUT / "normterrain_field.png"
    PILImage.fromarray((normed * 255).astype(np.uint8), mode="L").save(p_norm)
    p_colour = OUT / "normterrain_colour.png"
    PILImage.fromarray(colour).save(p_colour)
    p_graph = OUT / "normterrain_graph.ege.json"
    p_graph.write_text(g.to_json())
    print(f"saved: {p_norm}\n       {p_colour}\n       {p_graph}")

    report2 = ex.run_blocking()
    print(f"second run cache={cache.summary()} ok={report2.ok()}")
    assert all(r.cache_hit for r in report2.per_node.values()), "expected full cache hit"


if __name__ == "__main__":
    main()
