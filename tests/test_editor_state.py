"""Focused contracts for shared UI state and the public Editor facade."""

from __future__ import annotations

import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "easygrapheditor" / "src"))

from easygrapheditor.engine import Graph
from easygrapheditor.engine.persist import load_graph
from easygrapheditor.ui.adapters import GraphAdapter, decode_gradio_action, handle_gradio_event
from easygrapheditor.ui.canvas import EditorState
from easygrapheditor.ui.inspector import inspect_types
from easygrapheditor.ui.library import categories, node_detail, search_nodes
from easygrapheditor.ui.web_canvas import editor_canvas_html, streamlit_canvas_js

import easygrapheditor as ege


def test_command_history_restores_positions_and_collapse() -> None:
    graph = Graph()
    node = graph.add_node("input.number", params={"value": 1.0})
    state = EditorState(graph, selection=[node.id])
    state.move_nodes([node.id], 43, 17)
    assert graph.nodes[node.id].pos == (40.0, 20.0)
    state.toggle_collapsed(True)
    assert graph.nodes[node.id].collapsed and state.can_undo
    assert state.undo() and not graph.nodes[node.id].collapsed
    assert state.undo() and graph.nodes[node.id].pos == (0.0, 0.0)
    assert state.redo() and graph.nodes[node.id].pos == (40.0, 20.0)


def test_visual_node_moves_do_not_schedule_live_execution() -> None:
    graph = Graph()
    node = graph.add_node("input.number", params={"value": 1.0})
    state = EditorState(graph=graph, live=True)
    marker = object()
    state.run_report = marker  # type: ignore[assignment]

    state.move_nodes([node.id], 20, 0)

    assert state.run_report is marker
    assert not state.live_run_due(now=time.monotonic() + 10.0)


def test_group_and_parameter_commands_are_undoable() -> None:
    graph = Graph()
    left = graph.add_node("input.number", params={"value": 1.0})
    right = graph.add_node("input.number", params={"value": 2.0})
    state = EditorState(graph, selection=[left.id, right.id])
    state.group_selection("Pair")
    assert graph.groups and graph.groups[0].title == "Pair"
    state.set_param(left.id, "value", 9.0)
    assert graph.nodes[left.id].params["value"] == 9.0
    state.undo()
    assert graph.nodes[left.id].params["value"] == 1.0
    assert state.ungroup_selection() == [left.id, right.id]


def test_state_view_round_trip_and_debounced_live() -> None:
    graph = Graph()
    node = graph.add_node("input.number", params={"value": 1.0})
    state = EditorState(graph, selection=[node.id])
    state.viewport.zoom = 1.6
    state.settings["dev_mode"] = True
    view = state.view_dict()
    restored = EditorState(graph)
    restored.load_view(view)
    assert restored.selection == [node.id]
    assert restored.viewport.zoom == 1.6 and restored.settings["dev_mode"]
    restored.schedule_live_run(500)
    assert not restored.live_run_due(now=0.0)
    assert restored.live_run_due(now=time.monotonic() + 1.0)


def test_editor_saves_view_loads_it_and_scopes_library(tmp_path: Path) -> None:
    editor = ege.Editor(nodes=[ege.node(type_id="test.only", category="Test")(lambda: 1.0)])
    node = editor.add_node("test.only", pos=(25.0, 35.0))
    editor.state.selection = [node.id]
    editor.state.viewport.zoom = 1.3
    path = editor.save(tmp_path / "view.ege.json")
    loaded = ege.Editor(nodes=editor.nodes)
    loaded.load(path)
    assert loaded.graph.nodes[node.id].pos == (25.0, 35.0)
    assert loaded.state.selection == [node.id]
    assert loaded.state.viewport.zoom == 1.3
    assert loaded.state.allowed_node_types == {"test.only"}
    with pytest.raises(ValueError, match="scoped nodes"):
        loaded.add_node("input.number")
    _, saved_view = load_graph(path)
    assert saved_view["tabs"][0]["dirty"] is False


def test_empty_explicit_scope_and_any_export() -> None:
    editor = ege.Editor(nodes=[])
    assert editor.state.allowed_node_types == set()
    with pytest.raises(ValueError, match="scoped nodes"):
        editor.add_node("input.number")
    assert ege.Any is not None


def test_clear_cache_forces_a_recompute() -> None:
    graph = Graph()
    node = graph.add_node("input.number", params={"value": 3.0})
    adapter = GraphAdapter.from_any(graph)
    assert adapter.run().per_node[node.id].status == "ok"
    assert adapter.run().per_node[node.id].cache_hit
    adapter.clear_cache()
    report = adapter.run()
    assert report.per_node[node.id].status == "ok"
    assert not report.per_node[node.id].cache_hit


def test_adapter_mutation_is_undoable_and_debounced() -> None:
    graph = Graph()
    adapter = GraphAdapter.from_any(graph)
    node = adapter.mutate("add", lambda: graph.add_node("input.number", params={"value": 1.0}))
    assert node.id in graph.nodes and adapter.state.can_undo
    assert not adapter.consume_live_run()
    adapter.state._pending_live_at = 0.0
    adapter.consume_live_run()
    assert adapter.state.is_running or adapter.report is not None
    assert adapter.state.undo() and node.id not in graph.nodes


def test_background_run_exposes_working_stage_before_completion() -> None:
    graph = Graph()
    node = graph.add_node("simulate.staged_task", params={"seconds": 0.35, "label": "test"})
    adapter = GraphAdapter.from_any(graph)
    assert adapter.start_run()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        working = adapter.status()["working"].get(node.id, {})
        if adapter.state.is_running and working.get("stage"):
            break
        time.sleep(0.01)
    assert adapter.state.is_running
    assert adapter.status()["working"][node.id]["stage"]
    adapter.state.selection = [node.id]
    assert adapter.inspector_data()["status"] == "working"
    assert adapter.node_rows()[0]["stage"]
    while adapter.state.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not adapter.state.is_running and adapter.report is not None and adapter.report.ok()


def test_shared_web_action_protocol_mutates_editor_state() -> None:
    graph = Graph()
    left = graph.add_node("input.number", params={"value": 1.0})
    right = graph.add_node("output.readout", params={})
    adapter = GraphAdapter.from_any(graph)
    adapter.dispatch_web_action({"type": "select", "data": {"node": left.id}})
    assert adapter.state.selection == [left.id]
    adapter.dispatch_web_action({"type": "positions", "data": {"positions": {left.id: [37, 44]}}})
    assert graph.nodes[left.id].pos == (40.0, 40.0)
    adapter.dispatch_web_action({"type": "connect", "data": {"from_node": left.id, "from_port": "out", "to_node": right.id, "to_port": "value_in"}})
    assert graph.links
    adapter.dispatch_web_action({"type": "param", "data": {"node": left.id, "key": "value", "value": 7.0}})
    assert graph.nodes[left.id].params["value"] == 7.0
    adapter.dispatch_web_action({"type": "undo", "data": {}})
    assert graph.nodes[left.id].params["value"] == 1.0
    adapter.dispatch_web_action({"type": "viewport", "data": {"x": 12, "y": 4, "zoom": 1.4}})
    assert adapter.state.viewport.zoom == 1.4


def test_web_canvas_contains_gradio_and_streamlit_bridge_protocol() -> None:
    graph = Graph()
    html = editor_canvas_html(EditorState(graph))
    assert "trigger('click',action)" in html
    assert "setTriggerValue('action',action)" in html


def test_tabs_own_independent_graphs_and_views(tmp_path: Path) -> None:
    state = EditorState(Graph())
    first = state.graph.add_node("input.number", params={"value": 1.0})
    state.selection = [first.id]
    state.viewport.zoom = 1.3
    state.new_tab("Second")
    assert len(state.tabs) == 2 and not state.graph.nodes
    second = state.graph.add_node("input.number", params={"value": 2.0})
    state.switch_tab(0)
    assert first.id in state.graph.nodes and state.selection == [first.id]
    state.switch_tab(1)
    assert second.id in state.graph.nodes
    path = state.save_desktop_settings(tmp_path / "settings.json")
    restored = EditorState(Graph())
    restored.settings["dev_mode"] = True
    restored.save_desktop_settings(path)
    state.load_desktop_settings(path)
    assert path.exists()


def test_editor_facade_targets_active_tab_graph() -> None:
    editor = ege.Editor()
    first = editor.add_node("input.number")
    editor.state.new_tab("Second")
    second = editor.add_node("input.number")
    assert second.id in editor.graph.nodes and first.id not in editor.graph.nodes
    editor.state.switch_tab(0)
    assert first.id in editor.graph.nodes and second.id not in editor.graph.nodes


def test_gradio_event_decode_and_streamlit_module_source() -> None:
    import gradio as gr

    action = {"type": "select", "data": {"node": "x"}}
    assert decode_gradio_action(gr.EventData(None, {"value": action})) == action
    source = streamlit_canvas_js(EditorState(Graph()))
    assert source.startswith("export default function(component)")
    assert "component.setTriggerValue" in source


def test_gradio_event_handler_dispatches_eventdata_payload() -> None:
    import gradio as gr

    graph = Graph()
    node = graph.add_node("input.number", params={"value": 1.0})
    adapter = GraphAdapter.from_any(graph)
    status = handle_gradio_event(adapter, gr.EventData(None, {"value": {"type": "select", "data": {"node": node.id}}}))
    assert adapter.state.selection == [node.id] and status["selected"] == 1


def test_inspector_model_contains_runtime_connections_and_results() -> None:
    graph = Graph()
    source = graph.add_node("input.number", params={"value": 4.0})
    sink = graph.add_node("output.readout")
    graph.add_link(source.id, "out", sink.id, "value_in")
    adapter = GraphAdapter.from_any(graph)
    adapter.run()
    adapter.dispatch_web_action({"type": "select", "data": {"node": source.id}})
    detail = adapter.inspector_data()
    assert detail["status"] == "ok"
    assert detail["connections"]["outgoing"] == [{"node": sink.id, "port": "value_in"}]
    assert "out" in detail["results"] and "stages" in detail


def test_canvas_exposes_interactive_actions_and_named_ports() -> None:
    graph = Graph()
    graph.add_node("input.number", params={"value": 1.0})
    html = editor_canvas_html(EditorState(graph))
    for action in ("connect", "marquee", "delete", "group", "ungroup", "collapse", "param", "cache"):
        assert action in html
    assert "data-port" in html and "data-dir" in html and "data.NUMBER" in html
    assert "localStorage.setItem" in html and "disconnect" in html and "settings" in html


def test_web_actions_scope_add_cast_params_and_surface_errors() -> None:
    import gradio as gr

    graph = Graph()
    node = graph.add_node("input.number", params={"value": 1.0})
    adapter = GraphAdapter.from_any(EditorState(graph, allowed_node_types={"input.number"}))
    adapter.dispatch_web_action({"type": "param", "data": {"node": node.id, "key": "value", "value": "7.5"}})
    assert graph.nodes[node.id].params["value"] == 7.5
    event = gr.EventData(None, {"value": {"type": "add", "data": {"type_id": "output.readout"}}})
    status = handle_gradio_event(adapter, event)
    assert "error" in status and len(graph.nodes) == 1


def test_positioned_pygame_boxes_apply_active_drag_offsets() -> None:
    import pygame
    from easygrapheditor.ui.adapters import PygameStyle, compute_boxes

    graph = Graph()
    node = graph.add_node("input.number", pos=(60.0, 20.0))
    area = pygame.Rect(0, 0, 640, 400)
    normal = compute_boxes(graph, area, PygameStyle())[node.id]
    dragged = compute_boxes(graph, area, PygameStyle(), offsets={node.id: (30.0, 0.0)}, apply_offsets=True)[node.id]
    assert dragged.x - normal.x == 30


def test_staged_arithmetic_example_links_done_outputs_and_runs() -> None:
    path = Path(__file__).resolve().parents[1] / "easygrapheditor" / "examples" / "staged_arithmetic_demo.py"
    spec = spec_from_file_location("staged_arithmetic_demo", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    graph, refs = module.build()
    report = ege.Executor(graph).run_blocking()
    assert report.ok()
    assert graph.links[0].from_port == "done"
    assert refs["arith"] in graph.nodes


def test_library_types_and_shared_web_canvas_have_view_models() -> None:
    assert "Generate" in categories()
    assert node_detail("input.number")["outputs"]
    assert any(node.type_id == "input.number" for node in search_nodes("number"))
    assert any(row["id"] == "data.NUMBER" for row in inspect_types())
    graph = Graph()
    graph.add_node("input.number", params={"value": 1.0}, pos=(20, 30))
    html = editor_canvas_html(EditorState(graph))
    assert "<svg" in html and "background:#000" in html and "pointerdown" in html
