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
    theme = gr.themes.Base().set(
        body_background_fill="#000000",
        body_text_color="#ffffff",
        background_fill_primary="#000000",
        background_fill_secondary="#0c0c0c",
        block_background_fill="#000000",
        panel_background_fill="#000000",
        input_background_fill="#000000",
        button_secondary_background_fill="#111111",
        button_secondary_text_color="#ffffff",
        border_color_primary="#303030",
        block_border_color="#303030",
    )
    blocks = gr.Blocks(title=title, theme=theme)
    gradio_register_grapheditor(blocks, state, title=title, **opts)
    return blocks
