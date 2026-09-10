"""Async executor with topo order, caching, timing. See editor.md §4.4."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any

from .cache import Cache, content_hash
from .graph import Graph
from .nodes import NODE_REGISTRY, ExecCtx


def _hash_value(val: object) -> str:
    """Stable, compact hash for payloads (Field/Image arrays, traces, scalars)."""
    try:
        import numpy as np  # local import: engine stays light without hard dep at module load

        if isinstance(val, np.ndarray):
            return content_hash({"ndarray": {"shape": list(val.shape), "dtype": str(val.dtype), "bytes": val.tobytes().hex()[:65536]}})
        # Field/Image dataclasses wrap ndarrays.
        data = getattr(val, "data", None)
        if isinstance(data, np.ndarray):
            return content_hash({type(val).__name__: {"shape": list(data.shape), "dtype": str(data.dtype), "bytes": data.tobytes().hex()[:65536]}})
    except Exception:  # noqa: BLE001, S110 - fall through to generic
        pass
    if isinstance(val, dict):
        try:
            return content_hash({"dict": val})
        except Exception:  # noqa: BLE001
            return content_hash(str(sorted(str(val.items()))))
    if isinstance(val, (list, tuple)):
        return content_hash({"list": list(val)})
    return content_hash(str(val))


@dataclass
class NodeReport:
    node_id: str
    status: str  # ok | cached | error
    ms: float = 0.0
    cache_hit: bool = False
    error: str | None = None
    stages: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class RunReport:
    per_node: dict[str, NodeReport]
    ms: float = 0.0

    def ok(self) -> bool:
        return all(r.status in ("ok", "cached") for r in self.per_node.values())


class Executor:
    def __init__(self, graph: Graph, cache: Cache | None = None, max_concurrency: int = 4) -> None:
        self.graph = graph
        self.cache = cache or Cache()
        self.max_concurrency = max_concurrency
        self.outputs: dict[str, dict[str, Any]] = {}
        self.events: list[tuple[str, str]] = []  # (event, node_id) stub for UI

    def _node_hash(self, node_id: str, input_hashes: dict[str, str]) -> str:
        inst = self.graph.nodes[node_id]
        ndef = NODE_REGISTRY[inst.type_id]
        if not ndef.cacheable:
            return f"nocache:{node_id}:{time.time_ns()}"
        params = {k: v for k, v in inst.params.items()}
        return content_hash({"type": inst.type_id, "params": params, "inputs": input_hashes})

    async def run(self, dirty_only: bool = True) -> RunReport:
        t0 = time.perf_counter()
        errors = self.graph.validate()
        if errors:
            per = {e.node_id or "__graph__": NodeReport(e.node_id or "__graph__", "error", error=e.message) for e in errors}
            return RunReport(per_node=per, ms=0.0)
        order = self.graph.topo_order()
        per_node: dict[str, NodeReport] = {}
        # Map (to_node, to_port) -> (from_node, from_port).
        feed: dict[tuple[str, str], tuple[str, str]] = {}
        for link in self.graph.links:
            feed[(link.to_node, link.to_port)] = (link.from_node, link.from_port)
        for nid in order:
            inst = self.graph.nodes[nid]
            ndef = NODE_REGISTRY[inst.type_id]
            # Resolve inputs.
            kwargs: dict[str, Any] = {}
            input_hashes: dict[str, str] = {}
            for pin in ndef.inputs:
                src = feed.get((nid, pin.key))
                if src is None:
                    continue
                s_node, s_port = src
                val = self.outputs.get(s_node, {}).get(s_port)
                kwargs[pin.key] = val
                input_hashes[pin.key] = _hash_value(val)
            key = self._node_hash(nid, input_hashes)
            cached = self.cache.get(key)
            if cached is not None:
                # cached stores {"outputs": {...}, "stages": [...]}
                self.outputs[nid] = cached["outputs"]
                per_node[nid] = NodeReport(nid, "cached", ms=0.0, cache_hit=True, stages=cached.get("stages", []))
                self.events.append(("cache_hit", nid))
                continue
            self.events.append(("node_started", nid))
            ctx = ExecCtx(node_id=nid, params=dict(inst.params))
            t1 = time.perf_counter()
            try:
                call_kwargs = {**dict(inst.params), **kwargs}
                # Filter to accepted params.
                sig = inspect.signature(ndef.fn)
                accepted = {k for k in sig.parameters if k != "ctx"}
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}
                if "ctx" in sig.parameters:
                    res = ndef.fn(ctx, **call_kwargs)
                else:
                    res = ndef.fn(**call_kwargs)
                if inspect.isawaitable(res):
                    res = await res
                ms = (time.perf_counter() - t1) * 1000.0
                if len(ndef.outputs) == 1:
                    out_map = {ndef.outputs[0].key: res}
                else:
                    out_map = res if isinstance(res, dict) else {"out": res}
                self.outputs[nid] = out_map
                self.cache.set(key, {"outputs": out_map, "stages": list(ctx._stages)})
                per_node[nid] = NodeReport(nid, "ok", ms=ms, stages=list(ctx._stages))
                self.events.append(("node_done", nid))
            except Exception as e:  # noqa: BLE001 - surfaced in report
                ms = (time.perf_counter() - t1) * 1000.0
                per_node[nid] = NodeReport(nid, "error", ms=ms, error=f"{type(e).__name__}: {e}")
                self.events.append(("error", nid))
        return RunReport(per_node=per_node, ms=(time.perf_counter() - t0) * 1000.0)

    def run_blocking(self, dirty_only: bool = True) -> RunReport:
        return asyncio.run(self.run(dirty_only=dirty_only))
