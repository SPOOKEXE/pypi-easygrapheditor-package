# Changelog

## 0.2.0

- Added a shared command-based `EditorState` with undo/redo, console history,
  tabs, settings, snap, collapse, fit, viewport persistence, and live-run
  scheduling.
- Added a shared true-black HTML/SVG canvas surface for Gradio and Streamlit.
- Preserved pygame editing while committing dragged positions into saved graphs.
- Expanded library, inspector, Types, toolbar, and status view models.
- Updated `Editor` with scoped nodes, persisted view save/load, and an optional
  headless graph path.
- Made Gradio, Streamlit, and pygame optional lazy extras.
- Added one command-line contract to every demo: headless by default, `--ui`
  for visual mode, `--backend` for host choice, and live or read-only modes.
