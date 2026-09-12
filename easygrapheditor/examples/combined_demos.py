"""Combined demos: every demo graph tiled into one grid-arranged mega-graph.

Each demo's nodes keep their links/loops (ids namespaced per demo) and are
offset into grid cells; a single Executor runs the whole thing once. All
Field/Image outputs are saved plus assembled into a contact sheet.
Run: uv run easygrapheditor/examples/combined_demos.py
"""

import copy
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # demo modules live beside this file

from demo_runner import run_demo
from easygrapheditor.engine import Executor, Graph
from easygrapheditor.engine.graph import Link, NodeInstance
from easygrapheditor.engine.loops import LoopDef
from easygrapheditor.ui.adapters import payload_to_pil

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

DEMOS: list[tuple[str, str, str]] = [
    ("minimal", "minimal_editor", "Minimal"),
    ("stagedarith", "staged_arithmetic_demo", "Staged Arithmetic"),
    ("noiseterrain", "noise_terrain_demo", "Noise Terrain"),
    ("normterrain", "normterrain_demo", "Normalized Terrain"),
    ("aitrace", "ai_trace_factory_demo", "AI Trace Factory"),
    ("loop", "loop_demo", "Loop"),
    ("subflow", "subflow_demo", "Subworkflow"),
]

COLS = 3
PAD_X = 420
PAD_Y = 320


def _load_demo_graphs() -> list[tuple[str, str, Graph]]:
    loaded = []
    for name, module_name, title in DEMOS:
        module = importlib.import_module(module_name)
        built = module.build()
        graph = built[0] if isinstance(built, tuple) else built
        loaded.append((name, title, graph))
    return loaded


def build() -> tuple[Graph, dict[str, dict[str, str]]]:
    """Tile all demo graphs into one grid graph. Returns (graph, refs)."""
    loaded = _load_demo_graphs()

    # Uniform cell stride from the largest demo bounding box.
    widths, heights = [], []
    for _name, _title, graph in loaded:
        xs = [n.pos[0] for n in graph.nodes.values()]
        ys = [n.pos[1] for n in graph.nodes.values()]
        widths.append(max(xs) - min(xs))
        heights.append(max(ys) - min(ys))
    stride_x = max(widths) + PAD_X
    stride_y = max(heights) + PAD_Y

    merged = Graph()
    refs: dict[str, dict[str, str]] = {}
    for idx, (name, title, graph) in enumerate(loaded):
        xs = [n.pos[0] for n in graph.nodes.values()]
        ys = [n.pos[1] for n in graph.nodes.values()]
        ox = (idx % COLS) * stride_x - min(xs)
        oy = (idx // COLS) * stride_y - min(ys)
        remap = {nid: f"{name}__{nid}" for nid in graph.nodes}
        cell_refs: dict[str, str] = {}
        for nid, inst in graph.nodes.items():
            merged.nodes[remap[nid]] = NodeInstance(
                id=remap[nid],
                type_id=inst.type_id,
                params=copy.deepcopy(inst.params),
                pos=(inst.pos[0] + ox, inst.pos[1] + oy),
                collapsed=inst.collapsed,
            )
            cell_refs[nid] = remap[nid]
        for link in graph.links:
            merged.links.append(Link(remap[link.from_node], link.from_port, remap[link.to_node], link.to_port))
        for loop in graph.loops:
            merged.loops.append(
                LoopDef(
                    name=f"{name}/{loop.name}",
                    body=[remap[nid] for nid in loop.body],
                    condition=remap[loop.condition],
                    max_iterations=loop.max_iterations,
                )
            )
        # Keep the AI demo's side effects inside the combined output dir.
        for nid, inst in merged.nodes.items():
            if nid.startswith(f"{name}__") and inst.type_id == "ai.trace_dataset":
                inst.params["path"] = str(OUT / "combined_traces.jsonl")
        refs[name] = {"title": title, **cell_refs}
    return merged, refs


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower() or "out"


def save_contact_sheet(cells: list[tuple[str, str, object]], path: Path, cols: int = COLS) -> Path:
    """Assemble every output thumbnail into a labelled grid PNG."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    thumb, bar, pad = 256, 30, 12
    items = cells or [("empty", "no outputs", PILImage.new("RGB", (thumb, thumb), (40, 44, 54)))]
    rows = (len(items) + cols - 1) // cols
    sheet = PILImage.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + bar + pad) + pad + bar), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 6), f"Combined demos: {len(items)} outputs", fill=(230, 230, 235))
    for idx, (demo, label, img) in enumerate(items):
        thumb_img = img.convert("RGB").resize((thumb, thumb))
        cx = pad + (idx % cols) * (thumb + pad)
        cy = pad + bar + (idx // cols) * (thumb + bar + pad)
        sheet.paste(thumb_img, (cx, cy))
        draw.text((cx + 4, cy + thumb + 4), f"{demo}: {label}"[:38], fill=(150, 155, 165))
    sheet.save(path)
    return path


def main() -> None:
    combined_traces = OUT / "combined_traces.jsonl"
    combined_traces.unlink(missing_ok=True)

    graph, refs = build()
    errors = graph.validate()
    print("validate:", errors)
    assert not errors, errors

    ex = Executor(graph)
    report = ex.run_blocking()
    print(f"run {report.ms:.1f}ms ok={report.ok()} nodes={len(graph.nodes)} links={len(graph.links)} loops={len(graph.loops)}")

    titles = {name: refs[name].pop("title") for name in refs}
    for name in [n for n, _, _ in DEMOS]:
        rows = {k: v for k, v in report.per_node.items() if k.startswith(f"{name}__")}
        bad = {k: v.error for k, v in rows.items() if v.status == "error"}
        print(f"  [{titles[name]}] {len(rows)} nodes ok={not bad}" + (f" ERRORS={bad}" if bad else ""))
    assert report.ok(), {k: v.error for k, v in report.per_node.items() if v.status == "error"}

    from easygrapheditor.engine.nodes import get_node

    cells: list[tuple[str, str, object]] = []
    for name in [n for n, _, _ in DEMOS]:
        for nid, port_map in ex.outputs.items():
            if not nid.startswith(f"{name}__"):
                continue
            inst = graph.nodes[nid]
            ndef = get_node(inst.type_id)
            label = ndef.title if ndef else nid
            for port, value in port_map.items():
                img = payload_to_pil(value)
                if img is None:
                    continue
                img.save(OUT / f"combined_{name}_{_slug(label)}_{_slug(port)}.png")
                cells.append((titles[name], f"{label} · {port}", img))

    grid_path = save_contact_sheet(cells, OUT / "combined_grid.png")
    graph_path = OUT / "combined_graph.ege.json"
    graph_path.write_text(graph.to_json())
    print(f"saved: {grid_path}\n       {graph_path}")
    print(f"outputs: {len(cells)} images")


if __name__ == "__main__":
    run_demo(build=build, cli_main=main, title="Combined Demos", description=__doc__ or "Combined demos", script_path=__file__)
