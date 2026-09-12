"""Shared command-line launcher for EasyGraphEditor examples."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

BackendId = Literal["pygame", "gradio", "streamlit"]
UiMode = Literal["live", "read-only"]


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Open the graph visualiser instead of running the CLI path.",
    )
    parser.add_argument(
        "--backend",
        choices=("pygame", "gradio", "streamlit"),
        default="pygame",
        help="UI backend used with --ui (default: pygame).",
    )
    parser.add_argument(
        "--mode",
        choices=("live", "read-only"),
        default="live",
        help="Allow graph editing or show a read-only running graph.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Alias for --mode read-only.",
    )
    return parser


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:  # noqa: BLE001
        return False


def launch_ui(
    graph: Any,
    *,
    title: str,
    backend: BackendId,
    mode: UiMode,
    script_path: str | Path,
) -> Any:
    """Run ``graph`` in one UI host, preserving the same editing policy."""
    from easygrapheditor.ui.adapters import GraphAdapter

    editable = mode == "live"
    adapter = GraphAdapter.from_any(graph)

    if backend == "pygame":
        from easygrapheditor.ui.pygame_app import run_pygame_viewer

        adapter.start_run()
        return run_pygame_viewer(adapter, title=title, editable=editable, autorun=False)
    if backend == "gradio":
        from easygrapheditor.ui.gradio_app import build_gradio_app

        adapter.run()
        return build_gradio_app(adapter, title=title, editable=editable).launch()
    if not _in_streamlit():
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(script_path).resolve()),
            "--",
            "--ui",
            "--backend",
            "streamlit",
            "--mode",
            mode,
        ]
        os.execv(sys.executable, command)
    import streamlit as st
    from easygrapheditor.ui.streamlit_app import build_streamlit_app

    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """<style>
        .stApp, [data-testid="stAppViewContainer"] { background:#000; color:#fff; }
        .block-container { max-width:none; padding:0.5rem 0.75rem 1rem; }
        [data-testid="stHeader"] { background:#000; }
        </style>""",
        unsafe_allow_html=True,
    )

    session_key = f"ege:demo:{Path(script_path).stem}"
    adapter = st.session_state.get(session_key) or adapter
    st.session_state[session_key] = adapter
    if adapter.report is None and not adapter.state.is_running:
        adapter.start_run()
    return build_streamlit_app(
        adapter,
        title=title,
        editable=editable,
        autorun=False,
        session_key=session_key,
    )


def run_demo(
    *,
    build: Callable[[], Any],
    cli_main: Callable[[], None],
    title: str,
    description: str,
    script_path: str | Path,
    argv: list[str] | None = None,
) -> Any:
    """Choose the unchanged CLI path or a requested UI backend."""
    args = build_parser(description).parse_args(argv)
    if not args.ui:
        return cli_main()
    built = build()
    graph = built[0] if isinstance(built, tuple) else built
    mode: UiMode = "read-only" if args.read_only else args.mode
    return launch_ui(
        graph,
        title=title,
        backend=args.backend,
        mode=mode,
        script_path=script_path,
    )


__all__ = ["BackendId", "UiMode", "build_parser", "launch_ui", "run_demo"]
