"""Adapter + DI register tests (gradio/pygame real, streamlit via fake container)."""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine import Graph
from easygrapheditor.ui.adapters import (
    GraphAdapter,
    gradio_register_grapheditor,
    layout_graph,
    pygame_register_grapheditor,
    streamlit_register_grapheditor,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeContainer:
    """Stand-in for a streamlit tab/sidebar: records calls, never imports streamlit."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def markdown(self, body: str, *a, **k) -> None:
        self.calls.append(("markdown", body))

    def image(self, image, *a, **k) -> None:
        self.calls.append(("image", k.get("caption", "")))

    def json(self, data, *a, **k) -> None:
        self.calls.append(("json", data))

    def button(self, label: str, *a, **k) -> bool:
        self.calls.append(("button", label))
        return False

    def expander(self, label: str, *a, **k):
        self.calls.append(("expander", label))
        return self

    def slider(self, label: str, *a, **k):
        self.calls.append(("slider", label))
        return k.get("value", 0)

    def number_input(self, label: str, *a, **k):
        self.calls.append(("number_input", label))
        return k.get("value", 0)

    def text_input(self, label: str, *a, **k):
        self.calls.append(("text_input", label))
        return k.get("value", "")

    def text_area(self, label: str, *a, **k):
        self.calls.append(("text_area", label))
        return k.get("value", "")

    def selectbox(self, label: str, options, *a, **k):
        self.calls.append(("selectbox", label))
        return options[k.get("index", 0)] if options else None

    def checkbox(self, label: str, *a, **k):
        self.calls.append(("checkbox", label))
        return k.get("value", False)

    def file_uploader(self, label: str, *a, **k):
        self.calls.append(("file_uploader", label))

    def __enter__(self):
        return self

    def __exit__(self, *a) -> bool:
        return False


def _tiny_graph() -> Graph:
    g = Graph()
    n = g.add_node("input.number", params={"value": 2.0})
    r = g.add_node("output.readout", params={})
    g.add_link(n.id, "out", r.id, "value_in")
    return g


def test_adapter_run_and_summarize():
    adapter = GraphAdapter.from_any(_tiny_graph())
    report = adapter.run()
    assert report.ok() and adapter.ok
    text = adapter.summarize("Tiny")
    assert "Tiny" in text and "Readout" in text and "ok" in text
    assert len(adapter.node_rows()) == 2
    assert adapter.output_images() == []  # number graphs have no thumbnails


def test_adapter_output_images():
    g = Graph()
    g.add_node("generate.noise", params={"resolution": 16, "frequency": 2.0, "seed": 1.0, "octaves": 1, "gain": 0.5})
    adapter = GraphAdapter.from_any(g)
    assert adapter.run().ok()
    images = adapter.output_images()
    assert len(images) == 1 and images[0][1].size == (16, 16)


def test_layout_graph_depths():
    g = Graph()
    a = g.add_node("input.number", params={"value": 1.0})
    b = g.add_node("output.readout", params={})
    c = g.add_node("output.readout", params={})
    g.add_link(a.id, "out", b.id, "value_in")
    g.add_link(b.id, "value", c.id, "value_in")
    _pos, depth = layout_graph(g)
    assert (depth[a.id], depth[b.id], depth[c.id]) == (0, 1, 2)


def test_gradio_register_builds():
    import gradio as gr

    adapter = GraphAdapter.from_any(_tiny_graph())
    adapter.run()
    with gr.Blocks() as blocks:
        refs = gradio_register_grapheditor(blocks, adapter, title="Tiny")
    assert {"summary", "canvas", "gallery", "table", "run", "live", "clear", "notice", "adapter"} <= set(refs)


def test_streamlit_register_fake_container():
    fake = FakeContainer()
    adapter = streamlit_register_grapheditor(_tiny_graph(), container=fake, title="Tiny")
    kinds = [k for k, _ in fake.calls]
    assert kinds[0] == "button"  # run button first, so it can re-execute before render
    assert "markdown" in kinds and "json" in kinds
    assert adapter.report is None  # button not pressed in fake

    fake2 = FakeContainer()
    g = Graph()
    g.add_node("generate.noise", params={"resolution": 8, "frequency": 2.0, "seed": 1.0, "octaves": 1, "gain": 0.5})
    adapter2 = GraphAdapter.from_any(g)
    adapter2.run()
    streamlit_register_grapheditor(adapter2, container=fake2)
    assert any(k == "image" for k, _ in fake2.calls)


def test_pygame_register_draws():
    import pygame

    pygame.init()
    try:
        screen = pygame.Surface((800, 600))
        adapter = GraphAdapter.from_any(_tiny_graph())
        adapter.run()
        dirty = pygame_register_grapheditor(screen, adapter, title="Tiny")
        assert dirty.width > 0 and dirty.height > 0
    finally:
        pygame.quit()


def test_pygame_viewer_quits_headless():
    import pygame
    from easygrapheditor.ui.pygame_app import run_pygame

    pygame.init()
    try:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        adapter = run_pygame(GraphAdapter.from_any(_tiny_graph()), size=(640, 480), max_frames=5)
        assert adapter.report is not None and adapter.ok
    finally:
        pygame.quit()


def _minimal_page() -> None:
    """Streamlit page body, fully self-contained (see test_live._loop_page)."""
    import sys
    from pathlib import Path

    import easygrapheditor

    pkg = Path(easygrapheditor.__file__).resolve()
    sys.argv = ["view_demo.py", "--demo", "minimal", "--ui", "streamlit"]
    sys.path.insert(0, str(pkg.parents[1]))  # .../easygrapheditor/src
    sys.path.insert(0, str(pkg.parents[2] / "examples"))  # .../easygrapheditor/examples
    import view_demo

    view_demo.main()


def test_streamlit_view_demo_renders():
    """Full page render via streamlit's headless AppTest (no browser needed)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_minimal_page, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    assert len(at.markdown) >= 1 and len(at.button) >= 1


def test_run_demo_lists():
    proc = subprocess.run([str(ROOT / "scripts" / "run-demo.sh"), "--list"], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0
    names = {line.split()[0] for line in proc.stdout.strip().splitlines()}
    assert {"minimal", "noiseterrain", "stagedarith", "normterrain", "aitrace", "terrain"} <= names
