"""Streamlit backend: page renderer on top of DI adapters.

For embedding into your own app, call
:func:`easygrapheditor.ui.adapters.streamlit_register_grapheditor` with your
tab/sidebar/expander container directly.
"""

from __future__ import annotations

from typing import Any

from .adapters import GraphAdapter, StreamlitContainer, streamlit_register_grapheditor
from .canvas import EditorState


def build_streamlit_app(
    state: EditorState | GraphAdapter,
    container: StreamlitContainer | None = None,
    autorun: bool = True,
    **opts: Any,
) -> GraphAdapter:
    """Render the graph into ``container`` (default: current page).

    With ``autorun`` (default) the graph executes on first render so visitors
    see outputs immediately; the Run button re-executes.
    """
    adapter = GraphAdapter.from_any(state)
    if autorun and adapter.report is None:
        adapter.run()
    return streamlit_register_grapheditor(adapter, container=container, **opts)
