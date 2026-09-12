# EasyGraphEditor

EasyGraphEditor is a Python-first DAG engine with an embeddable node editor.
Declare nodes in Python, run them headlessly in tests, then open the same graph
in Gradio, Streamlit, or pygame.

```bash
pip install easygrapheditor
pip install 'easygrapheditor[gradio]'
```

```python
import easygrapheditor as ege

@ege.node(category="Generate", type_id="demo.double")
def double(value: ege.Number) -> ege.Number:
    return value * 2

editor = ege.Editor(nodes=[double, *ege.list_nodes()], title="My graph")
number = editor.add_node("input.number", pos=(40, 60))
result = editor.add_node("demo.double", pos=(320, 60))
editor.graph.add_link(number.id, "out", result.id, "value")
editor.to_gradio().launch()
```

## Backends

- `Editor(...).to_gradio()` builds a local interactive HTML/SVG canvas plus
  server-backed library, inspector, toolbar, previews, and snapshots.
- `Editor(...).to_streamlit()` mounts the same local canvas surface and
  server-backed controls in a Streamlit page.
- `Editor(...).to_pygame()` opens the immediate-mode desktop editor.
- `Editor(...).run_headless()` runs the graph without any optional UI package.

Optional packages are lazy. Importing `easygrapheditor` only needs NumPy and
Pillow. Calling an unavailable backend explains which extra to install.
The interactive web bridge requires `gradio>=6.26` or `streamlit>=1.63`;
pygame uses `pygame>=2.5`.

Passing `nodes=None` exposes the full registered library. Passing a list scopes
the editor library exactly to those decorated functions or `NodeDef` values.
`Editor(nodes=[])` intentionally exposes no addable nodes.

The embedded SVG sends the same JSON action protocol to Gradio and Streamlit,
so canvas selections, drags, viewport changes, and controls persist through
the shared command history.

## Editing and persistence

Every backend works through `EditorState`, which owns selection, viewport,
snap-to-grid, collapsed nodes, tabs, settings, console entries, and undo/redo.
Web settings persist in browser local storage; desktop settings persist in
`~/.easygrapheditor.json`. Settings cover node style, pricing, developer
details, token precision, and error visibility.
Node positions live on the graph and are saved with it. UI-only state is saved
under `editor.view`, so headless graph diffs remain clean.

```python
editor.state.live = True
editor.state.selection = [result.id]
editor.state.snap_selection()
editor.save("workflow.ege.json")

loaded = ege.Editor(nodes=[double, *ege.list_nodes()])
loaded.load("workflow.ege.json")
report = loaded.run_headless()
assert report.ok()
```

## Demos and checks

```bash
./scripts/run-demo.sh --list
./scripts/run-demo.sh --demo noiseterrain
./scripts/run-demo.sh --demo noiseterrain --ui
./scripts/run-demo.sh --demo stagedarith --ui --backend gradio
./scripts/run-demo.sh --demo normterrain --ui --backend streamlit --mode read-only
uv run pytest -q
uv run ruff check easygrapheditor/src tests
```

The included terrain, staged-task, loop, and subworkflow demos are in
[`easygrapheditor/examples`](easygrapheditor/examples).

Every demo accepts the same flags. Without `--ui`, its normal command-line
path processes the graph. With `--ui`, pygame opens by default. Select another
host with `--backend pygame|gradio|streamlit`. UI mode defaults to editable
live mode; use `--mode read-only` or `--read-only` for a running visualisation
that cannot change the graph.
