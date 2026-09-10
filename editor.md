# EasyGraphEditor — Technical Specification

> Goal: ComfyUI-class node graph editor as a Gradio-style Python package.
> Source intent: `plans/PLAN.md` — clone ComfyUI node editor UX, ship as `pip install` package with adapters for Gradio, Streamlit, Pygame, others.
> References: `plans/*.png` (3x ComfyUI, 7x `node-graph-template` terrain demo).

## 1. Product Overview

### 1.1 What we are building
A Python library that lets a developer declare nodes in pure Python and instantly get:

1. A reactive DAG execution **engine** (async, typed, cached).
2. An embeddable node **editor UI** (canvas + library + inspector + toolbar).
3. One-line integrations: `Gradio`, `Streamlit`, `Pygame`, headless.

```python
import easygrapheditor as ege

@ege.node(category="Generate", color="blue")
def noise(frequency: ege.Number = 4.0, seed: ege.Number = 7.0) -> ege.Field:
    ...

app = ege.Editor(nodes=[noise, ridged, domain_warp, combine, erode, slope, colourise])
app.to_gradio()    # or .to_streamlit(), .to_pygame(), .run_headless(graph_json)
```

### 1.2 Non-goals for v1
- No hosted SaaS, no user accounts, no marketplace.
- No full ComfyUI backend compatibility (no checkpoints/samplers by default).
- No collaborative editing.

## 2. Feature Inventory (derived from screenshots)

### 2.1 Canvas / Graph editor
- [x] Infinite dark grid canvas, pan/zoom (zoom % in status bar, e.g. `95%`, `fit` button).
- [x] Draggable nodes, bezier links with port-color coding (blue=FIELD, yellow=NUMBER, pink=IMAGE).
- [x] Multi-select, group/ungroup, collapse/uncollapse, compress/expand, snap-to-grid.
- [x] Minimap + viewport controls (bottom-right in ComfyUI shot `15-39-12.png`).
- [x] Multi-tab workflows: `video_minimax_h3...`, `Unsaved Workflow (5)`, `+` new tab.
- [x] Top run bar: `Extensions`, `Run`, queue count (`1`), `0 active`, error indicator.
- [x] Inline node previews (heightmap thumbnail, colour map thumbnail).
- [x] Group nodes (blue `Use Image Size` container in ComfyUI shot).

### 2.2 Node model
Each node card (both ComfyUI + template):

- Header: `title` + status badge: `cached` (grey/green) / `working` (purple pulsing).
- Body variants:
  - Param widgets: slider (`Frequency 4.00`), dropdown (`aspect_ratio 1:1 (Square)`, `Palette alpine`, `Operation add/subtract`), toggle (`Absolute`, `Relief`), text/file (`transparent_rgb_gaming_mouse.png`, `choose file to upload`), large prompt text area (MiniMax prompt), numeric steppers (`duration 5.0`, `noise_seed 168866841893410`).
  - Readouts/thumbnails.
  - Note node (`Size Settings Reference` megapixel table).
- Ports:
  - Inputs left, outputs right. Named + typed: `In: FIELD`, `Mask: FIELD`, `Out: IMAGE`, `A/B: NUMBER`, `Done: NUMBER`, `After: ANY`, `IMAGE/MASK/VIDEO`.
  - Multi-edge fan-out supported (e.g. `Ridged.Out` -> two inputs).
- Node states: `idle / working (fetching/planning/thinking/writing/checking) / cached / error / dirty`.
- Per-node timing display: `2003 ms`, `Seconds 2.00`.

Required node catalog for parity tests:
- Generate: `Noise`, `Ridged`.
- Filter: `Domain Warp`, `Terrace`, `Slope`, `Threshold`, `Combine`, `Erode`.
- Colour: `Colourise`.
- Simulate: `Staged Task` (async demo), `Arithmetic (add/subtract/abs)`.
- Input/Output: `Number`, `Readout`, `Note`.
- Comfy-parity examples: `Load Image`, `Preview Image`, `Save Image (Advanced)`, `CLIP Text Encode`, `VAE Encode/Decode`, `KSampler`, `Load Checkpoint`, `Load LoRA`, `Empty Latent Image`, `Video Combine`, `Image to Video`, `Save Video`, `Resolution Selector`, `Get Image Size`.

### 2.3 Type system
From `15-42-41.png` Types tab:

- A port carries a string type ID. Link legal iff `a == b` or either is wildcard.
- Core types v1:
  - `data.NUMBER` — one scalar float. No thumbnail, read value on hover/inspector.
  - `data.FIELD` — square grid of heights `0..1` (e.g. `64x64`). Thumbnail-able.
  - `data.IMAGE` — square grid of colours (`rgb 64^2`). Thumbnail-able.
  - `data.ANY` — wildcard (`Staged Task.After`).
  - Extension pattern: `IMAGE`, `MASK`, `VIDEO`, `MODEL`, `CONDITIONING`, `LATENT` as string IDs for Comfy-style graphs.
- Inspector must document: type ID, description, producer/consumer node list with counts (`Number · 7 nodes`, `Field · 8 nodes`).

### 2.4 Execution engine
From `15-41-28.png` → `15-41-35.png` → `15-42-15.png`:

- DAG evaluation, topological order, cycle rejection.
- Async nodes: `Staged Task` exposes stages `planning → fetching → thinking → writing → checking → Done`. UI polls/completes `5/5 steps · cached`.
- Cache: content-hash on `(node_id, node_type, params, input_hashes)`. Badges `cached` / `cache hit`, `cache 858/869`, `forget cache` button. Cache hit avoids recompute (`run 0.0 ms`).
- Modes: `live` (auto re-run on dirty, checkbox in toolbar) vs `build` (manual Run).
- Cancellation + error propagation: `Show errors tab in side panel` setting.
- Determinism: seed params (`Seed 7.00`, `noise_seed`).

### 2.5 Library / Add-node dialog
From `15-39-25.png`, `15-41-53.png`:

- Fuzzy search `Add a node...` / `filter nodes`.
- Facets: `Blueprints / Comfy / Partner / Extensions / Input / Output` + category tree: `Input > Number/Maths/Arithmetic`, `Generate > Noise/Ridged`, `Filter > Domain Warp/...`, `Colour > Colourise`, `Simulate > Erode/Staged Task`, `Output > Readout/Note`.
- Each row: `name`, `category path`, `description`, badges (`Comfy`, `Essentials`).
- Detail pane: description + `INPUTS` / `OUTPUTS` table with types.

### 2.6 Inspector panel
From `15-42-15.png`, `15-42-31.png`:

- Tabs: `Library | Inspector | Types`.
- For selection: node type ID (`task.staged`, `image.colourise`), cache state, one-line summary (`height becomes a picture visualisation`), full-size output preview, `inputs` thumbnails, `outputs` thumbnails, `result` scalar (`Done 1.0000`), `parameters` editable fields, `ports` list (`in After data.ANY unconnected`, `out Done Number`).
- Empty state: hint text (`Point at a row to pick it out on the canvas`).

### 2.7 Toolbar / Status / Settings
- Toolbar: `File | undo | redo | group | ungroup | compress | expand | collapse | uncollapse | snap | live | build | fit | forget cache`.
- Status bar: `zoom % · nodes N · links N · selected N · cache hits/total · run ms · live`.
- Settings modal (`15-40-04.png`):
  - `Nodes 2.0 Modern Node Design` toggle.
  - `API Nodes: Show pricing badge`.
  - `Dev Mode: API save, etc.`
  - `Edit Token Weight: Ctrl+up/down precision (0.05 slider)`.
  - `Error System: Show errors tab`.
  - Per-pack sections (`Comfy`, `Lite Graph`, `Appearance`, `Mask Editor`, `KJNodes`, ...).
- Left rail (ComfyUI): `Assets, Nodes, Models, Workflows, Apps, NodesMa, Templates, Help, Console, Shortcuts, Settings` + perf `T: 0.00s, FPS: 60.61`.

## 3. Architecture

Split per `PLAN.md`: `engine` vs `ui`.

```
easygrapheditor/
  __init__.py          # Editor facade, to_gradio/to_streamlit/to_pygame
  engine/
    types.py           # DataType registry, NUMBER/FIELD/IMAGE/ANY + custom
    nodes.py           # @node decorator, NodeDef, PortDef, ParamDef, widgets
    graph.py           # Graph, NodeInstance, Link, validate(), (de)serialize
    execute.py         # Executor: topo sort, async, cache, cancel, events
    cache.py           # Hash + LRU/disk cache
    persist.py         # .ege.json save/load (Comfy-compatible subset)
  ui/
    base.py            # EditorBackend protocol (all adapters implement this)
    canvas.py          # Shared view-model: pan/zoom/selection/clipboard
    library.py         # Search + categories
    inspector.py       # Selection detail view-model
    gradio_app.py
    streamlit_app.py
    pygame_app.py
    headless.py
  nodes_builtin/       # Number/Arithmetic/Note + terrain demo nodes
  nodes_comfy_demo/    # Load/Preview/Save Image, Resolution Selector (mock impls)
```

Dependency rule: `engine` has zero UI deps (pure Python + numpy/Pillow). `ui/*` depends on `engine`, never vice versa.

## 4. Engine Specification

### 4.1 Type registry
```python
@dataclass(frozen=True)
class DataType:
    id: str            # "data.NUMBER"
    label: str         # "Number"
    description: str
    preview: Literal["none","spark","thumbnail"]
```

- `register_type()` allows custom types (`"IMAGE"`, `"VIDEO"`, `"MODEL"`).
- Compatibility: `can_connect(out: str, inp: str) -> bool = out==inp or out=="data.ANY" or inp=="data.ANY"`.

### 4.2 Node definition
```python
@dataclass
class ParamDef:
    key: str; label: str
    kind: Literal["slider","int","number","dropdown","toggle","text","multiline","file","seed"]
    default: Any; min: float|None; max: float|None; step: float|None
    options: list[str]|None; affects_hash: bool = True

@dataclass
class PortDef:
    key: str; label: str; dtype: str; direction: Literal["in","out"]

@dataclass
class NodeDef:
    type_id: str       # "image.noise"
    title: str         # "Noise"
    category: str      # "Generate"
    color: str         # header color
    description: str
    params: list[ParamDef]
    inputs: list[PortDef]
    outputs: list[PortDef]
    fn: Callable       # sync or async (progress callback supported)
    cacheable: bool = True
```

- `@node(...)` decorator infers `PortDef`s from type hints (`-> ege.Field` etc.) but explicit overrides allowed.
- Async contract: `async def fn(ctx: ExecCtx, ...)`: `ctx.report(stage: str, frac: float)` drives `Staged Task` progress UI.

### 4.3 Graph model
- `Graph { nodes, links, loops }`.
- `NodeInstance { id, type_id, params: dict, pos: (x,y), collapsed: bool }`.
- `LoopDef { name, body: [node ids], condition: node id, max_iterations = 1000 }` (see §4.6).
- Subworkflow nodes (`core.subworkflow`) carry `{label, nodes, links, inputs, outputs}` in params (see §4.7).
- `validate()`: unknown type, missing required input, type mismatch, cycle detection (**except** `control.accumulate.next` feedback links, which carry previous-iteration values), loop rules, recursive subworkflow checks. Returns structured errors for Errors tab.
- Serialize: `to_json() -> {version, nodes[{id,type,pos,params}], links[...], loops[...]}`. Must round-trip canvas positions. Import best-effort subset of ComfyUI `workflow_api.json`.

### 4.4 Executor
- `Executor(graph, cache, max_concurrency=4, max_recursions=1000)`.
- `async run(dirty_only=True, cancel_token) -> RunReport { per_node: {status, ms, cache_hit, error} }`.
- Dirty propagation: param change or upstream change marks downstream dirty.
- Hash: `sha1(type_id + canonical params + sorted input hashes)`. Loop-body passes bypass the cache (iteration-scoped values); subworkflow inners share it (pure).
- Loops run as supernodes in condensation order; recursion depth counts subworkflow levels and errors past `max_recursions`.
- Events emitted for UI: `node_started/stage/node_done/cache_hit/run_done/error`.
- Sync wrapper `run_blocking()` for Streamlit/simple scripts.

### 4.6 Loops
- Primitives (`Control` category): `control.counter` (iteration index), `control.accumulate` (loop-carried state: emits previous `next`, starts at `initial`), `control.end_condition` (EndConditionNode → `done` 1/0 via `counter` | `threshold` | `truthy` modes).
- `Graph.add_loop(name, body, condition, max_iterations=1000)`; bodies stay acyclic — feedback flows only through `accumulate.next`, which validation/ordering ignore.
- Cap exhaustion is a run error naming the loop and cap. Loop reports merge per-iteration stages/ms under the body node ids.

### 4.7 Subworkflows (combine-into-one)
- `combine_nodes(graph, ids, label)` folds a selection into one `core.subworkflow` node: internal links move inside (positions relativized), boundary links become mapped input/output ports, outside links rewire through it. `expand_subworkflow_node` inlines back (id-clash safe).
- Inner graphs validate + execute recursively (nesting allowed); `describe_subworkflow` feeds hover/inspector (`label`, port maps, inner titles).
- UI must badge subflow nodes (`[sub:N]`), list loop membership (`[loop:name]`), and offer inline Expand.

### 4.5 Built-in data payloads (v1)
- `Number = float`, `Field = np.ndarray[float32, HxW] 0..1 square`, `Image = np.ndarray[uint8, HxWx3] square`. Fixed `64` default res for demo, configurable via param (`Resolution 64`).
- Terrain demo functions are pure numpy: value-noise + ridged variant, domain warp, max-combine, thermal+spin erode (iterative), slope from gradient, alpine colour ramp with `sea_level`.

## 5. UI Specification (shared across adapters)

All backends implement `EditorBackend` operating on a shared `EditorState { graph, selection, viewport, run_report, errors }`.

- Canvas interactions: drag node, drag port-to-port (highlight legal targets, reject illegal with toast), marquee select, delete, duplicate, `group/ungroup`, `collapse/expand`, copy/paste JSON, `snap` toggle.
- Widgets map from canonical `ParamDef.kind` (`canonical_kind()`; legacy `slider/toggle/dropdown/multiline` alias to `float_slider/checkbox/select/textarea`): `float_slider`→continuous decimal slider, `step_slider`→stepped slider, `int`/`number`/`seed`→numerical inputs, `text`→input box, `textarea`→multi-line box, `select`→dropdown, `checkbox`→switch, `file`→uploader. Shared spec/cast/nudge helpers live in `ui/widgets.py`.
- Manipulation per backend: Gradio — per-node inspector accordions with kind-mapped inputs + Apply, Loops max_iterations editors, Expand buttons; Streamlit — per-node expanders writing straight back to params; pygame — click-select, `[`/`]` tweak numeric/slider (Shift = x10), `T` toggle, `D` cycle dropdown, `E` expand subworkflow, `R` re-run.
- Node card must show: status dot/badge, timing (`2003 ms`), thumbnails.
- Right panel tabs `Library|Inspector|Types` + left rail (`Nodes`, `Console/Errors`, `Settings`) on desktop backends; collapsible drawer on Gradio/Streamlit.
- Toolbar + status bar as §2.7. `Run` triggers `executor.run()`. `live` toggles auto-run on debounce (500ms). `forget cache` clears cache + marks all dirty.
- Settings persisted to `localStorage` (web) / `~/.easygrapheditor.json` (desktop).

Backend notes:
- `gradio_app.py`: Gradio `Blocks` with custom HTML canvas (SVG/Canvas JS) + Python callbacks for run/inspect. No `gradio` import at top-level package import.
- `streamlit_app.py`: `st_canvas`-style layout; rerun on state change; cache executor in `st.session_state`.
- `pygame_app.py`: immediate-mode desktop reference; same `EditorState`, 60fps, for offline/test.
- `headless.py`: no UI, for tests/CI.

## 6. Persistence & Interop

- Native: `*.ege.json` (versioned). Includes `editor.view {viewport, selection}` separate from `graph` so headless diffs are clean.
- ComfyUI: best-effort import of `workflow_api.json` → map known types, wrap unknown as `Unknown (passthrough)` error node. Export subset back where types align (`Load Image`, `Preview Image`, etc. as mock nodes in `nodes_comfy_demo/`).
- Assets: file params store relative `assets/` paths; uploader copies in.

## 7. Package / API Requirements (Gradio-style DX)

- `pip install easygrapheditor[gradio|streamlit|pygame|all]`. Extras lazy — missing extra raises helpful message.
- Top-level facade:
  - `Editor(nodes=[...], title=...)`
  - `.to_gradio()`, `.to_streamlit()`, `.to_pygame()`, `.run_headless(graph_path)`, `.save/load()`.
  - `@node` decorator + `Number/Field/Image/Any` hints + `Param()` helper for sliders/dropdowns.
- Versioning: semver, `__version__`, changelog.
- Quality gates: `pytest` (engine: type-check, cycles, cache hits, async stages, determinism with seed), `ruff`, example app boots in each backend.

## 8. Milestones

1. **M1 Engine core** — types, `@node`, graph validate/serialize, sync executor + cache + tests (Arithmetic + Staged Task mock).
2. **M2 Terrain demo parity** — numpy Noise/Ridged/Warp/Combine/Erode/Slope/Colourise/Readout, `live` re-run, timing + cache badges verified headless.
3. **M3 Pygame editor** — canvas, library search, inspector, toolbar/statusbar, save/load.
4. **M4 Gradio + Streamlit adapters** — same `EditorState`, widget mapping, thumbnails.
5. **M5 Comfy-parity pack + docs** — mock Load/Preview/Save/CLIP/KSampler/Checkpoint nodes, Comfy import, settings modal, README + PyPI publish.

## 9. Acceptance Criteria (v1)

- Recreate terrain graph from `15-41-09.png` in <2 min via Library search; `live` run shows thumbnails + `cached` on second run with `0.0 ms`.
- Two `Staged Task (2s/3s) → Arithmetic(add)` reproduces `15-41-35.png` timings ±200ms and correct sum; switching to `subtract` updates inspector as in `15-42-15.png`.
- `Types` tab lists all types with producer/consumer counts.
- `pytest -q` green; `Editor(...).to_gradio()` and `.to_streamlit()` boot without engine changes.
- Saved `*.ege.json` reloads with identical results hash.

## 10. Open Questions

1. Max field/image resolution for v1 (64 vs 256) — perf vs fidelity?
2. Disk cache location + eviction policy?
3. Comfy `workflow_api.json` import depth — how many node types to map for demo?
4. Canvas implementation for Gradio — custom component vs iframe + API?
