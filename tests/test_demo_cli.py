"""Demo command-line and read-only UI launch contracts."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "easygrapheditor" / "examples"
sys.path.insert(0, str(EXAMPLES))

import demo_runner
from easygrapheditor.engine import Graph
from easygrapheditor.ui.canvas import EditorState
from easygrapheditor.ui.web_canvas import editor_canvas_html


def test_demo_parser_defaults_to_cli_and_pygame_live() -> None:
    parser = demo_runner.build_parser("demo")
    cli = parser.parse_args([])
    assert not cli.ui
    assert cli.backend == "pygame"
    assert cli.mode == "live"

    ui = parser.parse_args(["--ui", "--backend", "gradio", "--read-only"])
    assert ui.ui and ui.backend == "gradio" and ui.read_only


def test_run_demo_keeps_cli_and_ui_paths_separate(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    graph = Graph()
    calls: list[object] = []

    def cli_main() -> None:
        calls.append("cli")

    def fake_launch(candidate, **options):  # type: ignore[no-untyped-def]
        calls.append((candidate, options))
        return "opened"

    monkeypatch.setattr(demo_runner, "launch_ui", fake_launch)
    demo_runner.run_demo(
        build=lambda: (graph, {}),
        cli_main=cli_main,
        title="Demo",
        description="demo",
        script_path=__file__,
        argv=[],
    )
    assert calls == ["cli"]

    calls.clear()
    result = demo_runner.run_demo(
        build=lambda: (graph, {}),
        cli_main=cli_main,
        title="Demo",
        description="demo",
        script_path=__file__,
        argv=["--ui", "--backend", "streamlit", "--mode", "read-only"],
    )
    assert result == "opened"
    candidate, options = calls[0]
    assert candidate is graph
    assert options["backend"] == "streamlit"
    assert options["mode"] == "read-only"


def test_read_only_canvas_hides_mutating_controls() -> None:
    state = EditorState(Graph())
    markup = editor_canvas_html(state, editable=False)
    assert '"editable": false' in markup.replace("&quot;", '"')
    assert 'data-action="run"' in markup
    assert 'data-action="add"' not in markup
    assert 'data-action="param"' not in markup

