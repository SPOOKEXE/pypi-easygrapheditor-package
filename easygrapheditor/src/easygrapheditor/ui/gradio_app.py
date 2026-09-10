"""Gradio backend: standalone app builder on top of DI adapters.

For embedding into your own UI, call
:func:`easygrapheditor.ui.adapters.gradio_register_grapheditor` with your
``gr.Blocks`` / ``gr.Tab`` directly.
"""

from __future__ import annotations

from typing import Any

from .adapters import GraphAdapter, gradio_register_grapheditor
from .canvas import EditorState


def build_gradio_app(state: EditorState | GraphAdapter, title: str = "Graph Editor", **opts: Any):  # type: ignore[no-untyped-def]
    """Build a standalone Gradio app visualising ``state``. Caller launches it."""
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError("Install the gradio extra: pip install 'easygrapheditor[gradio]'") from e
    blocks = gr.Blocks(title=title)
    gradio_register_grapheditor(blocks, state, title=title, **opts)
    return blocks
