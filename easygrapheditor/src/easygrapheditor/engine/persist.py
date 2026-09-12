"""Native graph files, portable file assets, and a ComfyUI workflow subset."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from .cache import content_hash
from .graph import GRAPH_VERSION, Graph
from .nodes import NODE_REGISTRY, NodeDef, PortDef

COMFY_TO_EGE = {
    "LoadImage": "comfy.load_image",
    "PreviewImage": "comfy.preview_image",
    "SaveImage": "comfy.save_image_advanced",
    "SaveImageAdvanced": "comfy.save_image_advanced",
    "CLIPTextEncode": "comfy.clip_text_encode",
    "VAEEncode": "comfy.vae_encode",
    "VAEDecode": "comfy.vae_decode",
    "KSampler": "comfy.ksampler",
    "CheckpointLoaderSimple": "comfy.load_checkpoint",
    "LoraLoader": "comfy.load_lora",
    "EmptyLatentImage": "comfy.empty_latent_image",
    "VHS_VideoCombine": "comfy.video_combine",
    "VideoCombine": "comfy.video_combine",
    "ImageToVideo": "comfy.image_to_video",
    "SaveVideo": "comfy.save_video",
    "ResolutionSelector": "comfy.resolution_selector",
    "GetImageSize": "comfy.get_image_size",
}
EGE_TO_COMFY = {value: key for key, value in COMFY_TO_EGE.items()}

# Comfy stores widgets as positional values. These schemas ensure that a
# KSampler's third value remains `steps` instead of being misread as `steps`
# after `control_after_generate` was inserted upstream.
COMFY_WIDGET_KEYS: dict[str, list[str]] = {
    "LoadImage": ["image", "upload"],
    "SaveImage": ["filename_prefix"],
    "SaveImageAdvanced": ["filename_prefix"],
    "CLIPTextEncode": ["text"],
    "KSampler": [
        "seed",
        "control_after_generate",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
    ],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "VHS_VideoCombine": ["frame_rate", "loop_count", "filename_prefix", "format"],
    "VideoCombine": ["frame_rate", "loop_count", "filename_prefix", "format"],
    "SaveVideo": ["filename_prefix"],
    "ResolutionSelector": ["width", "height"],
}


def _asset_path(value: str, target: Path) -> Path | None:
    source = Path(value)
    candidates = [source] if source.is_absolute() else [target.parent / source, Path.cwd() / source]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _copy_asset(value: str, target: Path) -> str:
    source = _asset_path(value, target)
    if source is None:
        return value.replace("\\", "/")
    digest = hashlib.sha1(source.read_bytes()).hexdigest()[:12]
    name = f"{digest}{source.suffix.lower()}"
    assets = target.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    destination = assets / name
    if not destination.exists():
        shutil.copy2(source, destination)
    return (Path("assets") / name).as_posix()


def _portable_assets(nodes: list[dict[str, Any]], target: Path) -> None:
    for node in nodes:
        type_id = str(node.get("type_id", ""))
        params = node.get("params", {})
        if type_id == "core.subworkflow":
            _portable_assets(list(params.get("nodes", [])), target)
            continue
        definition = NODE_REGISTRY.get(type_id)
        if definition is None:
            continue
        for param in definition.params:
            if param.kind == "file" and isinstance(params.get(param.key), str):
                params[param.key] = _copy_asset(params[param.key], target)


def save_graph(graph: Graph, path: str | Path, view: dict[str, Any] | None = None) -> Path:
    """Save a graph and copy declared file parameters into ``assets/`` beside it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = graph.to_dict()
    _portable_assets(payload["nodes"], target)
    payload["editor"] = {"view": view or {}}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _unknown_function(**_kwargs: Any) -> Any:
    raise RuntimeError("Unsupported Comfy node cannot execute in EasyGraphEditor")


def _safe_key(value: str, fallback: str) -> str:
    key = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    return key or fallback


def _normal_dtype(value: Any) -> str:
    raw = str(value or "data.ANY")
    return {"FLOAT": "data.NUMBER", "INT": "data.NUMBER", "NUMBER": "data.NUMBER"}.get(
        raw,
        raw
        if raw
        in {
            "IMAGE",
            "MASK",
            "VIDEO",
            "MODEL",
            "CLIP",
            "VAE",
            "CONDITIONING",
            "LATENT",
            "data.NUMBER",
            "data.FIELD",
            "data.IMAGE",
            "data.ANY",
        }
        else "data.ANY",
    )


def _unknown_type(
    comfy_type: str, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> str:
    signature = content_hash({"class": comfy_type, "inputs": inputs, "outputs": outputs})[:12]
    type_id = f"comfy.unknown.{_safe_key(comfy_type, 'node').lower()}.{signature}"
    if type_id not in NODE_REGISTRY:
        NODE_REGISTRY[type_id] = NodeDef(
            type_id,
            f"Unknown ({comfy_type})",
            "Comfy",
            "red",
            "Unsupported Comfy node. Ports are retained for workflow round-tripping.",
            [],
            [
                PortDef(port["key"], port["label"], port["dtype"], "in", required=False)
                for port in inputs
            ],
            [PortDef(port["key"], port["label"], port["dtype"], "out") for port in outputs],
            _unknown_function,
            cacheable=False,
            badges=["Comfy", "Unknown"],
        )
    return type_id


def _unknown_ports(type_id: str) -> dict[str, list[dict[str, str]]]:
    definition = NODE_REGISTRY[type_id]
    return {
        "inputs": [
            {"key": port.key, "label": port.label, "dtype": port.dtype}
            for port in definition.inputs
        ],
        "outputs": [
            {"key": port.key, "label": port.label, "dtype": port.dtype}
            for port in definition.outputs
        ],
    }


def _restore_unknown_nodes(nodes: list[dict[str, Any]]) -> None:
    """Re-register dynamic unknown port shapes before native graph loading."""
    for node in nodes:
        params = node.get("params", {})
        if node.get("type_id", "").startswith("comfy.unknown."):
            ports = params.get("_comfy_ports", {})
            comfy_type = str(params.get("comfy_class_type", "Unknown"))
            type_id = _unknown_type(
                comfy_type, list(ports.get("inputs", [])), list(ports.get("outputs", []))
            )
            if type_id != node.get("type_id"):
                # Keep old native files usable even if their historic dynamic
                # identifier predates the current stable port-signature rule.
                NODE_REGISTRY[str(node["type_id"])] = replace(
                    NODE_REGISTRY[type_id], type_id=str(node["type_id"])
                )
        if node.get("type_id") == "core.subworkflow":
            _restore_unknown_nodes(list(params.get("nodes", [])))


def _ports_from_raw(
    raw: dict[str, Any], direction: str, observed: dict[int, str]
) -> list[dict[str, Any]]:
    source = raw.get("inputs" if direction == "in" else "outputs", [])
    ports: dict[int, dict[str, Any]] = {}
    if isinstance(source, list):
        for index, entry in enumerate(source):
            if not isinstance(entry, dict):
                continue
            slot = int(entry.get("slot_index", entry.get("slot", index)))
            label = str(entry.get("name", f"{direction}_{slot}"))
            ports[slot] = {
                "key": _safe_key(label, f"{direction}_{slot}"),
                "label": label,
                "dtype": _normal_dtype(entry.get("type")),
            }
    elif isinstance(source, dict) and direction == "in":
        for index, (name, value) in enumerate(source.items()):
            ports[index] = {
                "key": _safe_key(str(name), f"in_{index}"),
                "label": str(name),
                "dtype": _normal_dtype(None if isinstance(value, list) else type(value).__name__),
            }
    for slot, dtype in observed.items():
        ports.setdefault(
            slot,
            {
                "key": f"{direction}_{slot}",
                "label": f"{direction.title()} {slot}",
                "dtype": _normal_dtype(dtype),
            },
        )
    if direction == "out" and not ports:
        ports[0] = {"key": "out_0", "label": "Output 0", "dtype": "data.ANY"}
    return [ports[index] for index in sorted(ports)]


def _workflow_observed_ports(data: dict[str, Any]) -> dict[str, dict[str, dict[int, str]]]:
    observed: dict[str, dict[str, dict[int, str]]] = {}
    for raw_link in data.get("links", []):
        if isinstance(raw_link, dict):
            source_id, source_slot = raw_link.get("origin_id"), raw_link.get("origin_slot", 0)
            target_id, target_slot = raw_link.get("target_id"), raw_link.get("target_slot", 0)
            dtype = raw_link.get("type")
        else:
            _link_id, source_id, source_slot, target_id, target_slot, *rest = raw_link
            dtype = rest[0] if rest else None
        observed.setdefault(str(source_id), {"in": {}, "out": {}})["out"][int(source_slot or 0)] = (
            str(dtype or "data.ANY")
        )
        observed.setdefault(str(target_id), {"in": {}, "out": {}})["in"][int(target_slot or 0)] = (
            str(dtype or "data.ANY")
        )
    return observed


def _params_for(type_id: str, comfy_type: str, values: Any) -> dict[str, Any]:
    if isinstance(values, dict):
        return dict(values)
    definition = NODE_REGISTRY.get(type_id)
    if definition is None or not isinstance(values, list):
        return {}
    keys = COMFY_WIDGET_KEYS.get(comfy_type, [param.key for param in definition.params])
    mapped = {key: value for key, value in zip(keys, values)}
    params = {param.key: mapped[param.key] for param in definition.params if param.key in mapped}
    params["_comfy_widget_values"] = list(values)
    return params


def _port_key(type_id: str, direction: str, slot: int | str | None) -> str:
    definition = NODE_REGISTRY.get(type_id)
    ports = (
        []
        if definition is None
        else (definition.outputs if direction == "out" else definition.inputs)
    )
    if isinstance(slot, str) and any(port.key == slot for port in ports):
        return slot
    try:
        return ports[int(slot or 0)].key
    except (IndexError, ValueError, TypeError):
        return f"out_{slot or 0}" if direction == "out" else f"in_{slot or 0}"


def _unknown_specs_for_prompt(
    raw: dict[str, Any], output_slots: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = _ports_from_raw(raw, "in", {})
    outputs = _ports_from_raw(raw, "out", {slot: "data.ANY" for slot in output_slots})
    return inputs, outputs


def import_comfy_workflow_api(data: dict[str, Any]) -> Graph:
    """Import workflow or prompt JSON, retaining each unknown node's full port shape."""
    graph = Graph()
    ids: dict[str, str] = {}
    nodes = data.get("nodes")
    if isinstance(nodes, list):
        observed = _workflow_observed_ports(data)
        for raw in nodes:
            comfy_type = str(raw.get("type") or raw.get("class_type") or "Unknown")
            raw_id = str(raw.get("id", len(ids) + 1))
            if comfy_type in COMFY_TO_EGE:
                type_id = COMFY_TO_EGE[comfy_type]
            else:
                ports = observed.get(raw_id, {"in": {}, "out": {}})
                type_id = _unknown_type(
                    comfy_type,
                    _ports_from_raw(raw, "in", ports["in"]),
                    _ports_from_raw(raw, "out", ports["out"]),
                )
            pos = raw.get("pos") or [0, 0]
            inst = graph.add_node(
                type_id,
                _params_for(type_id, comfy_type, raw.get("widgets_values", raw.get("params", {}))),
                (float(pos[0]), float(pos[1])),
            )
            inst.params.update({"comfy_class_type": comfy_type, "_comfy_node_id": raw_id})
            if type_id.startswith("comfy.unknown."):
                inst.params["_comfy_ports"] = _unknown_ports(type_id)
            ids[raw_id] = inst.id
        for raw_link in data.get("links", []):
            if isinstance(raw_link, dict):
                source_id, source_slot = raw_link.get("origin_id"), raw_link.get("origin_slot")
                target_id, target_slot = raw_link.get("target_id"), raw_link.get("target_slot")
            else:
                _link_id, source_id, source_slot, target_id, target_slot, *_ = raw_link
            source, target = ids.get(str(source_id)), ids.get(str(target_id))
            if source is not None and target is not None:
                graph.add_link(
                    source,
                    _port_key(graph.nodes[source].type_id, "out", source_slot),
                    target,
                    _port_key(graph.nodes[target].type_id, "in", target_slot),
                )
        return graph

    prompt_nodes = {
        str(raw_id): raw
        for raw_id, raw in data.items()
        if isinstance(raw, dict) and "class_type" in raw
    }
    output_slots: dict[str, set[int]] = {raw_id: set() for raw_id in prompt_nodes}
    for raw in prompt_nodes.values():
        for value in raw.get("inputs", {}).values():
            if isinstance(value, list) and len(value) >= 2 and str(value[0]) in output_slots:
                output_slots[str(value[0])].add(int(value[1]))
    for raw_id, raw in prompt_nodes.items():
        comfy_type = str(raw["class_type"])
        if comfy_type in COMFY_TO_EGE:
            type_id = COMFY_TO_EGE[comfy_type]
        else:
            inputs, outputs = _unknown_specs_for_prompt(raw, output_slots[raw_id])
            type_id = _unknown_type(comfy_type, inputs, outputs)
        params = {
            key: value
            for key, value in raw.get("inputs", {}).items()
            if not isinstance(value, list)
        }
        inst = graph.add_node(type_id, params)
        inst.params.update({"comfy_class_type": comfy_type, "_comfy_node_id": raw_id})
        if type_id.startswith("comfy.unknown."):
            inst.params["_comfy_ports"] = _unknown_ports(type_id)
        ids[raw_id] = inst.id
    for raw_id, raw in prompt_nodes.items():
        target = ids[raw_id]
        for input_key, value in raw.get("inputs", {}).items():
            if isinstance(value, list) and len(value) >= 2 and str(value[0]) in ids:
                source = ids[str(value[0])]
                graph.add_link(
                    source,
                    _port_key(graph.nodes[source].type_id, "out", value[1]),
                    target,
                    _port_key(graph.nodes[target].type_id, "in", input_key),
                )
    return graph


def _widget_values(node: Any, comfy_type: str) -> list[Any]:
    definition = NODE_REGISTRY.get(node.type_id)
    keys = COMFY_WIDGET_KEYS.get(
        comfy_type, [param.key for param in definition.params] if definition else []
    )
    values = list(node.params.get("_comfy_widget_values", []))
    if len(values) < len(keys):
        values.extend([None] * (len(keys) - len(values)))
    defaults = {param.key: param.default for param in (definition.params if definition else [])}
    for index, key in enumerate(keys):
        if key in node.params:
            values[index] = node.params[key]
        elif values[index] is None and key in defaults:
            values[index] = defaults[key]
    return values


def export_comfy_workflow_api(graph: Graph, *, include_unknown: bool = False) -> dict[str, Any]:
    """Export only Comfy-aligned nodes by default, producing a loadable subset.

    ``include_unknown=True`` is for round-tripping within this package. Those
    nodes retain their original class and ports but cannot run in stock Comfy.
    """
    ordered = [
        node
        for node in graph.nodes.values()
        if node.type_id in EGE_TO_COMFY
        or include_unknown
        and node.type_id.startswith("comfy.unknown.")
    ]
    ids = {node.id: index + 1 for index, node in enumerate(ordered)}
    nodes: list[dict[str, Any]] = []
    for node in ordered:
        comfy_type = EGE_TO_COMFY.get(
            node.type_id, node.params.get("comfy_class_type", "EasyGraphUnknown")
        )
        raw: dict[str, Any] = {
            "id": ids[node.id],
            "type": comfy_type,
            "pos": list(node.pos),
            "widgets_values": _widget_values(node, comfy_type),
        }
        if node.type_id.startswith("comfy.unknown."):
            definition = NODE_REGISTRY[node.type_id]
            raw["inputs"] = [
                {"name": port.label, "type": port.dtype, "slot_index": index}
                for index, port in enumerate(definition.inputs)
            ]
            raw["outputs"] = [
                {"name": port.label, "type": port.dtype, "slot_index": index}
                for index, port in enumerate(definition.outputs)
            ]
        nodes.append(raw)
    links: list[list[Any]] = []
    for link in graph.links:
        if link.from_node not in ids or link.to_node not in ids:
            continue
        source, target = graph.nodes[link.from_node], graph.nodes[link.to_node]
        source_ports = NODE_REGISTRY[source.type_id].outputs
        target_ports = NODE_REGISTRY[target.type_id].inputs
        source_slot = next(
            (index for index, port in enumerate(source_ports) if port.key == link.from_port), 0
        )
        target_slot = next(
            (index for index, port in enumerate(target_ports) if port.key == link.to_port), 0
        )
        dtype = next((port.dtype for port in source_ports if port.key == link.from_port), "*")
        links.append(
            [len(links) + 1, ids[source.id], source_slot, ids[target.id], target_slot, dtype]
        )
    return {
        "version": 0.4,
        "last_node_id": len(nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
    }


def load_graph(path: str | Path) -> tuple[Graph, dict[str, Any]]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("version", GRAPH_VERSION) != GRAPH_VERSION:
        raise ValueError(f"Unsupported graph version: {data.get('version')}")
    _restore_unknown_nodes(list(data.get("nodes", [])))
    return Graph.from_dict(data), (data.get("editor") or {}).get("view", {})
