"""Launch any demo graph in a UI backend.

Usage:
    uv run python easygrapheditor/examples/view_demo.py --demo minimal --ui
    uv run python easygrapheditor/examples/view_demo.py --demo noiseterrain --ui --backend gradio

Prefer ./scripts/run-demo.sh, which wraps this with --demo/--ui/--list.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # demo modules live beside this file

DEMOS: dict[str, tuple[str, str]] = {
    "minimal": ("minimal_editor", "Minimal"),
    "stagedarith": ("staged_arithmetic_demo", "Staged Arithmetic"),
    "noiseterrain": ("noise_terrain_demo", "Noise Terrain"),
    "normterrain": ("normterrain_demo", "Normalized Terrain"),
    "aitrace": ("ai_trace_factory_demo", "AI Trace Factory"),
    "loop": ("loop_demo", "Loop: count to target"),
    "subflow": ("subflow_demo", "Subworkflow: combine + expand"),
    "combined": ("combined_demos", "Combined: all demos in one grid"),
}


def load_demo(name: str):  # type: ignore[no-untyped-def]
    try:
        module_name, _title = DEMOS[name]
    except KeyError:
        raise SystemExit(f"Unknown demo {name!r}. Choose from: {', '.join(sorted(DEMOS))}") from None
    return importlib.import_module(module_name), _title


def main(argv: list[str] | None = None) -> None:
    from demo_runner import build_parser, launch_ui

    ap = build_parser("Run or view an EasyGraphEditor demo graph.")
    ap.add_argument("--demo", default="minimal", choices=sorted(DEMOS), help="Demo graph to load.")
    args, _unknown = ap.parse_known_args(argv)

    module, title = load_demo(args.demo)
    if not args.ui:
        module.main()
        return
    built = module.build()
    graph = built[0] if isinstance(built, tuple) else built
    launch_ui(
        graph,
        title=title,
        backend=args.backend,
        mode="read-only" if args.read_only else args.mode,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
