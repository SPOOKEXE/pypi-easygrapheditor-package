"""Full noise terrain demo: Noise + Ridged -> Warp -> Combine -> Erode -> Colourise.

Mirrors plans/15-41-09. Run: uv run easygrapheditor/examples/noise_terrain_demo.py
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


def save_field_grey(name: str, arr: np.ndarray) -> Path:
    p = OUT / name
    img = PILImage.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    img.save(p)
    return p


def build() -> tuple[Graph, dict[str, str]]:
    """Build the demo graph. Returns (graph, refs) with node ids (also used by view_demo.py)."""
    g = Graph()
    n_noise = g.add_node("generate.noise", params={"resolution": 64, "frequency": 4.0, "seed": 7.0, "octaves": 4, "gain": 0.5}, pos=(0, 0))
    n_ridged = g.add_node("generate.ridged", params={"resolution": 64, "frequency": 4.0, "seed": 7.0, "octaves": 5, "gain": 0.5}, pos=(0, 160))
    n_warp = g.add_node("filter.domain_warp", params={"amount": 12.0}, pos=(260, 80))
    n_combine = g.add_node("filter.combine", params={"amount": 0.5, "mode": "maximum"}, pos=(480, 80))
    n_erode = g.add_node("simulate.erode", params={"thermal_passes": 30.0, "talus": 0.01, "spin_passes": 20.0}, pos=(700, 80))
    n_slope = g.add_node("filter.slope", params={}, pos=(920, 0))
    n_colour = g.add_node("colour.colourise", params={"palette": "alpine", "sea_level": 0.32, "relief": False}, pos=(920, 160))
    n_read = g.add_node("output.readout", params={}, pos=(1140, 80))

    g.add_link(n_noise.id, "out", n_warp.id, "field_in")
    g.add_link(n_ridged.id, "out", n_warp.id, "warp_by")
    g.add_link(n_warp.id, "out", n_combine.id, "a")
    g.add_link(n_ridged.id, "out", n_combine.id, "b")
    g.add_link(n_combine.id, "out", n_erode.id, "field_in")
    g.add_link(n_erode.id, "out", n_slope.id, "field_in")
    g.add_link(n_erode.id, "out", n_colour.id, "field_in")
    g.add_link(n_colour.id, "out", n_read.id, "value_in")
    return g, {"erode": n_erode.id, "slope": n_slope.id, "colour": n_colour.id, "readout": n_read.id}


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

    field = ex.outputs[refs["erode"]]["out"].data
    colour = ex.outputs[refs["colour"]]["out"].data
    slope = ex.outputs[refs["slope"]]["out"].data
    mean_h = ex.outputs[refs["readout"]]["value"]
    print(f"field 64x64 mean={field.mean():.3f} min={field.min():.3f} max={field.max():.3f} readout={mean_h:.3f}")

    p_field = save_field_grey("terrain_field.png", field)
    p_slope = save_field_grey("terrain_slope.png", slope)
    p_colour = OUT / "terrain_colour.png"
    PILImage.fromarray(colour).save(p_colour)
    p_graph = OUT / "terrain_graph.ege.json"
    p_graph.write_text(g.to_json())
    print(f"saved: {p_field}\n       {p_slope}\n       {p_colour}\n       {p_graph}")

    # Second run proves caching (all cached, ~0ms).
    report2 = ex.run_blocking()
    print(f"second run cache={cache.summary()} ok={report2.ok()}")
    assert all(r.cache_hit for r in report2.per_node.values()), "expected full cache hit"


if __name__ == "__main__":
    main()
