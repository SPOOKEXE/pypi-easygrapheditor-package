"""Launch any demo graph in a UI backend.

Usage:
    uv run python easygrapheditor/examples/view_demo.py --demo minimal --ui pygame
    uv run python easygrapheditor/examples/view_demo.py --demo noiseterrain --ui gradio
    streamlit run easygrapheditor/examples/view_demo.py -- --demo stagedarith --ui streamlit

Prefer ./scripts/run-demo.sh, which wraps this with --demo/--ui/--list.
"""

import argparse
import importlib
import os
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


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:  # noqa: BLE001 - old streamlit or import failure
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            return get_script_run_ctx() is not None
        except Exception:  # noqa: BLE001
            return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="View an easygrapheditor demo graph in a UI backend.")
    ap.add_argument("--demo", default="minimal", choices=sorted(DEMOS), help="Demo graph to load.")
    ap.add_argument("--ui", default="pygame", choices=["pygame", "gradio", "streamlit"], help="Backend to show it in.")
    # parse_known_args: under `streamlit run` stray server flags can leak into
    # sys.argv — never let that kill the page with SystemExit.
    args, _unknown = ap.parse_known_args(argv)

    module, title = load_demo(args.demo)
    built = module.build()
    graph = built[0] if isinstance(built, tuple) else built

    from easygrapheditor.ui.adapters import GraphAdapter

    adapter = GraphAdapter.from_any(graph)
    adapter.run()
    print(adapter.summarize(title))

    if args.ui == "pygame":
        from easygrapheditor.ui.pygame_app import run_pygame_viewer

        run_pygame_viewer(adapter, title=f"{title} (easygrapheditor)")
    elif args.ui == "gradio":
        from easygrapheditor.ui.gradio_app import build_gradio_app

        build_gradio_app(adapter, title=title).launch()
    elif args.ui == "streamlit":
        if not _in_streamlit():
            here = str(Path(__file__).resolve())
            print("Re-launching under `streamlit run` …")
            os.execvp("uv", ["uv", "run", "streamlit", "run", here, "--", "--demo", args.demo, "--ui", "streamlit"])
        from easygrapheditor.ui.streamlit_app import build_streamlit_app

        build_streamlit_app(adapter, title=title)


if __name__ == "__main__":
    main()
