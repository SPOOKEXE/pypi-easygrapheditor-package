"""Focused regression coverage for engine scheduling, contracts and interop."""

from __future__ import annotations

import asyncio
import time

from easygrapheditor.engine import (
    Any as AnyHint,
)
from easygrapheditor.engine import (
    Cache,
    CancellationToken,
    Executor,
    Field,
    Graph,
    Image,
    Number,
    Param,
    combine_nodes,
    expand_subworkflow_node,
    export_comfy_workflow_api,
    get_node,
    import_comfy_workflow_api,
    load_graph,
    node,
    save_graph,
)


def test_postponed_annotations_infer_typed_ports() -> None:
    @node(type_id="test.postponed", category="Test")
    def postponed(field: Field, count: Number) -> Image:
        return field

    definition = get_node("test.postponed")
    assert definition is not None
    assert [port.dtype for port in definition.inputs] == ["data.FIELD", "data.NUMBER"]
    assert definition.outputs[0].dtype == "data.IMAGE"

    @node(type_id="test.any_hint", category="Test")
    def any_hint(value: AnyHint) -> AnyHint:
        return value

    wildcard = get_node("test.any_hint")
    assert wildcard is not None and wildcard.inputs[0].dtype == "data.ANY"


def test_validation_requires_inputs_and_rejects_duplicate_fanin_and_inner_cycle() -> None:
    graph = Graph()
    left = graph.add_node("input.number", {"value": 1})
    right = graph.add_node("input.number", {"value": 2})
    maths = graph.add_node("maths.arithmetic")
    graph.add_link(left.id, "out", maths.id, "a")
    graph.add_link(right.id, "out", maths.id, "a")
    errors = [error.message for error in graph.validate()]
    assert any("fan-in" in error for error in errors)
    assert any("Required input 'b'" in error for error in errors)

    graph = Graph()
    outer = graph.add_node("core.subworkflow")
    outer.params = {
        "nodes": [
            {"id": "a", "type_id": "maths.arithmetic", "params": {}},
            {"id": "b", "type_id": "maths.arithmetic", "params": {}},
        ],
        "links": [
            {"from_node": "a", "from_port": "out", "to_node": "b", "to_port": "a"},
            {"from_node": "b", "from_port": "out", "to_node": "a", "to_port": "a"},
        ],
        "inputs": [],
        "outputs": [],
    }
    assert any("inner cycle" in error.message for error in graph.validate())


def test_async_wave_events_and_cancellation() -> None:
    @node(type_id="test.slow_left", category="Test")
    async def slow_left(ctx) -> float:  # type: ignore[no-untyped-def]
        ctx.report("started", 0.1)
        await asyncio.sleep(0.05)
        return 2.0

    @node(type_id="test.slow_right", category="Test")
    async def slow_right(ctx) -> float:  # type: ignore[no-untyped-def]
        ctx.report("started", 0.1)
        await asyncio.sleep(0.05)
        return 3.0

    graph = Graph()
    left, right = graph.add_node("test.slow_left"), graph.add_node("test.slow_right")
    total = graph.add_node("maths.arithmetic", {"operation": "add"})
    graph.add_link(left.id, "out", total.id, "a")
    graph.add_link(right.id, "out", total.id, "b")
    executor = Executor(graph, max_concurrency=2)
    started = time.perf_counter()
    report = executor.run_blocking()
    assert report.ok() and time.perf_counter() - started < 0.09
    assert any(event["event"] == "stage" for event in executor.events)
    assert executor.events[-1]["event"] == "run_done"

    @node(type_id="test.cancellable", category="Test")
    async def cancellable() -> float:
        await asyncio.sleep(1)
        return 1.0

    graph = Graph()
    graph.add_node("test.cancellable")
    executor = Executor(graph)
    token = CancellationToken()

    async def cancel_run() -> str:
        task = asyncio.create_task(executor.run(cancel_token=token))
        await asyncio.sleep(0.02)
        token.cancel()
        return next(iter((await task).per_node.values())).status

    assert asyncio.run(cancel_run()) == "cancelled"


def test_dirty_descendants_and_downstream_error_skip() -> None:
    calls: list[str] = []

    @node(type_id="test.counted", category="Test")
    def counted(
        value: float = Param(1.0, affects_hash=True), cosmetic: str = Param("a", affects_hash=False)
    ) -> float:  # type: ignore[no-untyped-def]
        calls.append(f"source:{value}")
        return float(value)

    @node(type_id="test.bad", category="Test")
    def bad() -> float:
        raise RuntimeError("boom")

    graph = Graph()
    source = graph.add_node("test.counted", {"value": 1, "cosmetic": "a"})
    unrelated = graph.add_node("test.counted", {"value": 9})
    out = graph.add_node("output.readout")
    graph.add_link(source.id, "out", out.id, "value_in")
    executor = Executor(graph)
    assert executor.run_blocking().ok()
    calls.clear()
    source.params["value"] = 2
    report = executor.run_blocking(dirty_only=True)
    assert report.ok() and calls == ["source:2"]
    assert report.per_node[unrelated.id].cache_hit
    calls.clear()
    source.params["cosmetic"] = "b"
    assert executor.run_blocking(dirty_only=True).ok() and not calls

    graph = Graph()
    bad_node, good = graph.add_node("test.bad"), graph.add_node("input.number", {"value": 1})
    maths = graph.add_node("maths.arithmetic")
    graph.add_link(bad_node.id, "out", maths.id, "a")
    graph.add_link(good.id, "out", maths.id, "b")
    report = Executor(graph).run_blocking()
    assert report.per_node[bad_node.id].status == "error"
    assert report.per_node[maths.id].status == "skipped"


def test_cache_identity_structural_invalidation_and_reset() -> None:
    calls: list[str] = []

    @node(type_id="test.node_identity", category="Test")
    def node_identity(ctx) -> str:  # type: ignore[no-untyped-def]
        calls.append(ctx.node_id)
        return ctx.node_id

    graph = Graph()
    first, second = graph.add_node("test.node_identity"), graph.add_node("test.node_identity")
    executor = Executor(graph)
    assert executor.run_blocking().ok()
    assert executor.outputs[first.id]["out"] == first.id
    assert executor.outputs[second.id]["out"] == second.id
    report = executor.run_blocking()
    assert report.per_node[first.id].cache_hit and report.per_node[second.id].cache_hit

    @node(type_id="test.rewire_reader", category="Test")
    def rewire_reader(value: float) -> float:
        calls.append(f"reader:{value}")
        return value

    left = graph.add_node("input.number", {"value": 1})
    right = graph.add_node("input.number", {"value": 2})
    reader = graph.add_node("test.rewire_reader")
    graph.add_link(left.id, "out", reader.id, "value")
    assert executor.run_blocking().ok()
    calls.clear()
    graph.links = [link for link in graph.links if link.to_node != reader.id]
    graph.add_link(right.id, "out", reader.id, "value")
    report = executor.run_blocking(dirty_only=True)
    assert executor.outputs[reader.id]["out"] == 2.0
    assert calls == ["reader:2.0"] and not report.per_node[reader.id].cache_hit

    executor.force_dirty([reader.id])
    calls.clear()
    assert executor.run_blocking(dirty_only=True).ok()
    assert calls == ["reader:2.0"]
    executor.forget_cache()
    calls.clear()
    assert executor.run_blocking(dirty_only=True).ok()
    assert "reader:2.0" in calls

    del graph.nodes[left.id]
    graph.links = [
        link for link in graph.links if link.from_node != left.id and link.to_node != left.id
    ]
    assert executor.run_blocking().ok()
    assert left.id not in executor.outputs and left.id not in executor._param_fingerprints


def test_validation_reports_all_errors_and_nested_subflow_contracts() -> None:
    graph = Graph()
    one, two = graph.add_node("maths.arithmetic"), graph.add_node("maths.arithmetic")
    report = Executor(graph).run_blocking()
    assert len(report.per_node) == 4
    assert [node.node_id for node in report.per_node.values()].count(one.id) == 2
    assert [node.node_id for node in report.per_node.values()].count(two.id) == 2

    nested = {
        "nodes": [{"id": "math", "type_id": "maths.arithmetic", "params": {}}],
        "links": [],
        "inputs": [
            {"key": "a", "dtype": "data.NUMBER", "inner_node": "math", "inner_port": "a"},
            {"key": "b", "dtype": "data.NUMBER", "inner_node": "math", "inner_port": "b"},
        ],
        "outputs": [
            {"key": "out", "dtype": "data.NUMBER", "inner_node": "math", "inner_port": "out"}
        ],
    }
    graph = Graph()
    outer = graph.add_node("core.subworkflow")
    outer.params = {
        "nodes": [{"id": "nested", "type_id": "core.subworkflow", "params": nested}],
        "links": [],
        "inputs": [
            {"key": "input", "dtype": "data.FIELD", "inner_node": "nested", "inner_port": "a"}
        ],
        "outputs": [],
    }
    errors = [error.message for error in graph.validate()]
    assert any("map dtype data.FIELD does not match data.NUMBER" in error for error in errors)
    assert any("nested.b" in error and "required input" in error for error in errors)

    graph = Graph()
    left = graph.add_node("input.number", {"value": 1})
    right = graph.add_node("input.number", {"value": 2})
    maths = graph.add_node("maths.arithmetic")
    readout = graph.add_node("output.readout")
    graph.add_link(left.id, "out", maths.id, "a")
    graph.add_link(right.id, "out", maths.id, "b")
    graph.add_link(maths.id, "out", readout.id, "value_in")
    inner = combine_nodes(graph, [maths.id], "Inner")
    outer = combine_nodes(graph, [inner.id], "Outer")
    assert {mapping["dtype"] for mapping in outer.params["inputs"]} == {"data.NUMBER"}
    assert outer.params["outputs"][0]["dtype"] == "data.NUMBER"
    assert graph.validate() == []


def test_staged_task_has_optional_after_and_done_ports() -> None:
    staged = get_node("simulate.staged_task")
    assert staged is not None
    assert [(port.key, port.dtype, port.required) for port in staged.inputs] == [
        ("after", "data.ANY", False)
    ]
    assert [(port.key, port.dtype) for port in staged.outputs] == [("done", "data.NUMBER")]


def test_lru_disk_and_comfy_known_unknown_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = Cache(maxsize=2, disk_path=tmp_path)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert cache.get("b") is None and Cache(2, tmp_path).get("a") == 1

    graph = import_comfy_workflow_api(
        {
            "nodes": [
                {"id": 1, "type": "LoadImage", "widgets_values": ["demo.png"]},
                {"id": 2, "type": "PreviewImage"},
                {"id": 3, "type": "FutureNode"},
            ],
            "links": [[1, 1, 0, 2, 0, "IMAGE"]],
        }
    )
    assert [node.type_id for node in list(graph.nodes.values())[:2]] == [
        "comfy.load_image",
        "comfy.preview_image",
    ]
    assert list(graph.nodes.values())[-1].type_id.startswith("comfy.unknown.")
    assert Executor(graph).run_blocking().per_node[next(reversed(graph.nodes))].status == "error"
    exported = export_comfy_workflow_api(graph)
    assert exported["nodes"][0]["type"] == "LoadImage" and exported["links"]


def test_comfy_unknown_multiport_widgets_and_aligned_export() -> None:
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "Mystery Source",
                "outputs": [
                    {"name": "first", "type": "FLOAT"},
                    {"name": "second", "type": "IMAGE"},
                ],
            },
            {
                "id": 2,
                "type": "Mystery Sink",
                "inputs": [
                    {"name": "left", "type": "FLOAT"},
                    {"name": "right", "type": "IMAGE"},
                ],
            },
            {
                "id": 3,
                "type": "KSampler",
                "widgets_values": [7, "fixed", 31, 8.0, "euler", "normal", 1.0],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "FLOAT"], [2, 1, 1, 2, 1, "IMAGE"]],
    }
    graph = import_comfy_workflow_api(workflow)
    source, sink, sampler = graph.nodes.values()
    assert len(get_node(source.type_id).outputs) == 2  # type: ignore[union-attr]
    assert len(get_node(sink.type_id).inputs) == 2  # type: ignore[union-attr]
    assert len({(link.from_port, link.to_port) for link in graph.links}) == 2
    assert sampler.params["seed"] == 7 and sampler.params["steps"] == 31
    preserved = export_comfy_workflow_api(graph, include_unknown=True)
    restored = import_comfy_workflow_api(preserved)
    unknown_nodes = [
        node for node in restored.nodes.values() if node.type_id.startswith("comfy.unknown.")
    ]
    assert len(unknown_nodes) == 2 and len(restored.links) == 2
    sampler_row = next(row for row in preserved["nodes"] if row["type"] == "KSampler")
    assert sampler_row["widgets_values"][0] == 7 and sampler_row["widgets_values"][2] == 31

    known = import_comfy_workflow_api(
        {
            "nodes": [{"id": 1, "type": "LoadImage"}, {"id": 2, "type": "PreviewImage"}],
            "links": [[1, 1, 0, 2, 0, "IMAGE"]],
        }
    )
    known.add_node(source.type_id)
    aligned = export_comfy_workflow_api(known)
    assert len(aligned["nodes"]) == 2
    assert import_comfy_workflow_api(aligned).validate() == []


def test_native_assets_and_result_hash_survive_reload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    original = tmp_path / "outside.png"
    original.write_bytes(b"small image fixture")
    graph = Graph()
    image = graph.add_node("comfy.load_image", {"image": str(original)})
    target = tmp_path / "workflow.ege.json"
    save_graph(graph, target, view={"zoom": 1.25})
    loaded, view = load_graph(target)
    copied = loaded.nodes[image.id].params["image"]
    assert (
        copied.startswith("assets/") and (tmp_path / copied).read_bytes() == original.read_bytes()
    )
    assert view == {"zoom": 1.25}

    numbers = Graph()
    number = numbers.add_node("input.number", {"value": 7.0})
    numeric_path = tmp_path / "numbers.ege.json"
    first = Executor(numbers).run_blocking()
    save_graph(numbers, numeric_path)
    reloaded, _ = load_graph(numeric_path)
    second = Executor(reloaded).run_blocking()
    assert first.result_hash == first.results_hash == second.result_hash
    assert number.id in reloaded.nodes


def test_subworkflow_preserves_groups_through_native_roundtrip() -> None:
    graph = Graph()
    left = graph.add_node("input.number", {"value": 2}, pos=(10, 20))
    right = graph.add_node("input.number", {"value": 3}, pos=(30, 20))
    total = graph.add_node("maths.arithmetic", {"operation": "add"}, pos=(50, 20))
    graph.add_link(left.id, "out", total.id, "a")
    graph.add_link(right.id, "out", total.id, "b")
    graph.add_group("Inputs and sum", [left.id, right.id, total.id])
    combine_nodes(graph, [left.id, right.id, total.id], "Math")
    restored = Graph.from_json(graph.to_json())
    restored_subflow = next(
        node for node in restored.nodes.values() if node.type_id == "core.subworkflow"
    )
    restored_ids = expand_subworkflow_node(restored, restored_subflow.id)
    assert len(restored.groups) == 1
    assert set(restored.groups[0].nodes) == set(restored_ids)
    assert restored.validate() == []
