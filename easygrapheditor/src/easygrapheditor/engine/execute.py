"""Async executor with topo order, caching, timing. See editor.md §4.4."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any

from .cache import Cache, content_hash
from .graph import SUBWORKFLOW_TYPE_ID, Graph
from .loops import is_feedback_link, loop_body_order
from .nodes import NODE_REGISTRY, ExecCtx
from .subworkflows import DEFAULT_MAX_RECURSIONS, build_inner_graph

ACCUMULATE_TYPE_ID = "control.accumulate"
COUNTER_TYPE_ID = "control.counter"


def _is_done(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    try:
        return float(value) != 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return bool(value)


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
    def __init__(
        self,
        graph: Graph,
        cache: Cache | None = None,
        max_concurrency: int = 4,
        max_recursions: int = DEFAULT_MAX_RECURSIONS,
    ) -> None:
        self.graph = graph
        self.cache = cache or Cache()
        self.max_concurrency = max_concurrency
        self.max_recursions = max_recursions
        self.outputs: dict[str, dict[str, Any]] = {}
        self.events: list[tuple[str, str]] = []  # (event, node_id) stub for UI

    # -- hashing -----------------------------------------------------
    def _node_hash(self, graph: Graph, node_id: str, input_hashes: dict[str, str]) -> str:
        inst = graph.nodes[node_id]
        ndef = NODE_REGISTRY[inst.type_id]
        if not ndef.cacheable:
            return f"nocache:{node_id}:{time.time_ns()}"
        params = {k: v for k, v in inst.params.items()}
        return content_hash({"type": inst.type_id, "params": params, "inputs": input_hashes})

    # -- input resolution --------------------------------------------
    @staticmethod
    def _feed(graph: Graph) -> dict[tuple[str, str], tuple[str, str]]:
        """Map (to_node, to_port) -> (from_node, from_port)."""
        return {(link.to_node, link.to_port): (link.from_node, link.from_port) for link in graph.links}

    @staticmethod
    def _resolve(
        nid: str,
        port: str,
        feed: dict[tuple[str, str], tuple[str, str]],
        store: dict[str, dict[str, Any]],
        seeded: dict[tuple[str, str], Any],
    ) -> tuple[bool, Any]:
        if (nid, port) in seeded:
            return True, seeded[(nid, port)]
        src = feed.get((nid, port))
        if src is None:
            return False, None
        return True, store.get(src[0], {}).get(src[1])

    def _inputs_for(
        self, graph: Graph, nid: str, feed: dict, store: dict, seeded: dict,
    ) -> tuple[dict, dict]:
        ndef = NODE_REGISTRY[graph.nodes[nid].type_id]
        kwargs: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        for pin in ndef.inputs:
            found, val = self._resolve(nid, pin.key, feed, store, seeded)
            if not found:
                continue
            kwargs[pin.key] = val
            hashes[pin.key] = _hash_value(val)
        return kwargs, hashes

    async def _call_fn(
        self, graph: Graph, nid: str, kwargs: dict[str, Any], iteration: int,
    ) -> tuple[dict, list, float]:
        inst, ndef = graph.nodes[nid], NODE_REGISTRY[graph.nodes[nid].type_id]
        ctx = ExecCtx(node_id=nid, params=dict(inst.params), iteration=iteration)
        t1 = time.perf_counter()
        call_kwargs = {**dict(inst.params), **kwargs}
        sig = inspect.signature(ndef.fn)
        accepted = {k for k in sig.parameters if k != "ctx"}
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}
        res = ndef.fn(ctx, **call_kwargs) if "ctx" in sig.parameters else ndef.fn(**call_kwargs)
        if inspect.isawaitable(res):
            res = await res
        ms = (time.perf_counter() - t1) * 1000.0
        out_map = {ndef.outputs[0].key: res} if len(ndef.outputs) == 1 else (res if isinstance(res, dict) else {"out": res})
        return out_map, list(ctx._stages), ms

    # -- single node ---------------------------------------------------
    async def _exec_single(
        self, graph: Graph, nid: str, store: dict, feed: dict, seeded: dict,
        depth: int, iteration: int, use_cache: bool, per_node: dict[str, NodeReport], prefix: str,
    ) -> None:
        inst = graph.nodes[nid]
        if inst.type_id == SUBWORKFLOW_TYPE_ID:
            await self._exec_subworkflow(graph, nid, store, feed, seeded, depth, per_node, prefix)
            return
        key = prefix + nid
        kwargs, hashes = self._inputs_for(graph, nid, feed, store, seeded)
        if use_cache:
            cached = self.cache.get(self._node_hash(graph, nid, hashes))
            if cached is not None:
                store[nid] = cached["outputs"]
                per_node[key] = NodeReport(nid, "cached", ms=0.0, cache_hit=True, stages=cached.get("stages", []))
                self.events.append(("cache_hit", nid))
                return
        self.events.append(("node_started", nid))
        try:
            out_map, stages, ms = await self._call_fn(graph, nid, kwargs, iteration)
            store[nid] = out_map
            if use_cache:
                self.cache.set(self._node_hash(graph, nid, hashes), {"outputs": out_map, "stages": stages})
            per_node[key] = NodeReport(nid, "ok", ms=ms, stages=stages)
            self.events.append(("node_done", nid))
        except Exception as e:  # noqa: BLE001 - surfaced in report
            per_node[key] = NodeReport(nid, "error", ms=0.0, error=f"{type(e).__name__}: {e}")
            self.events.append(("error", nid))

    # -- subworkflows ----------------------------------------------------
    async def _exec_subworkflow(
        self, graph: Graph, nid: str, store: dict, feed: dict, seeded: dict,
        depth: int, per_node: dict[str, NodeReport], prefix: str,
    ) -> None:
        key = prefix + nid
        if depth >= self.max_recursions:
            per_node[key] = NodeReport(nid, "error", error=f"max recursions ({self.max_recursions}) exceeded at '{nid}'")
            self.events.append(("error", nid))
            return
        inst = graph.nodes[nid]
        inner = build_inner_graph(inst)
        seeded_inner: dict[tuple[str, str], Any] = {}
        for mapping in inst.params.get("inputs", []):
            found, val = self._resolve(nid, mapping["key"], feed, store, seeded)
            if found:
                seeded_inner[(mapping["inner_node"], mapping["inner_port"])] = val
        inner_store: dict[str, dict[str, Any]] = {}
        await self._exec_graph(inner, inner_store, seeded_inner, depth + 1, True, per_node, prefix + nid + "/")
        if any(r.status == "error" for k, r in per_node.items() if k.startswith(prefix + nid + "/")):
            per_node[key] = NodeReport(nid, "error", error=f"inner error in subworkflow '{nid}'")
            self.events.append(("error", nid))
            return
        store[nid] = {m["key"]: inner_store.get(m["inner_node"], {}).get(m["inner_port"]) for m in inst.params.get("outputs", [])}
        per_node[key] = NodeReport(nid, "ok", ms=sum(r.ms for k, r in per_node.items() if k.startswith(prefix + nid + "/")))
        self.events.append(("node_done", nid))

    # -- loops -------------------------------------------------------------
    @staticmethod
    def _supernode_order(graph: Graph, loop_of: dict[str, int]) -> list[tuple[str, Any]]:
        sup_of = {nid: ("loop", loop_of[nid]) if nid in loop_of else ("node", nid) for nid in graph.nodes}
        sups: list[tuple[str, Any]] = list(dict.fromkeys(sup_of.values()))
        index = {sup: i for i, sup in enumerate(sups)}
        type_of = lambda nid: graph.nodes[nid].type_id if nid in graph.nodes else None
        edges: set[tuple[int, int]] = set()
        for link in graph.links:
            a, b = sup_of.get(link.from_node), sup_of.get(link.to_node)
            if a is not None and b is not None and a != b and not is_feedback_link(link, type_of):
                edges.add((index[a], index[b]))
        indeg = [0] * len(sups)
        adj: list[list[int]] = [[] for _ in sups]
        for a, b in edges:
            adj[a].append(b)
            indeg[b] += 1
        queue = [i for i, d in enumerate(indeg) if d == 0]
        order: list[tuple[str, Any]] = []
        while queue:
            i = queue.pop(0)
            order.append(sups[i])
            for nxt in adj[i]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(sups):
            raise ValueError("Graph has a cycle outside loops; cannot order.")
        return order

    async def _run_loop(
        self, graph: Graph, loop: Any, store: dict, feed: dict, seeded: dict,
        depth: int, per_node: dict[str, NodeReport], prefix: str,
    ) -> None:
        body = [nid for nid in loop.body if nid in graph.nodes]
        type_of = lambda nid: graph.nodes[nid].type_id if nid in graph.nodes else None
        order = loop_body_order(body, graph.links, type_of)
        carry: dict[str, Any] = {}
        for nid in body:  # accumulate nodes start from their `initial` param
            if graph.nodes[nid].type_id == ACCUMULATE_TYPE_ID:
                carry[nid] = graph.nodes[nid].params.get("initial", 0.0)
        cap = max(1, int(loop.max_iterations))
        done = False
        for it in range(cap):
            for nid in order:
                inst = graph.nodes[nid]
                key = prefix + nid
                if inst.type_id == COUNTER_TYPE_ID:
                    store[nid] = {"count": float(it)}
                    self._merge_loop_report(per_node, key, nid, 0.0, [])
                    continue
                if inst.type_id == ACCUMULATE_TYPE_ID:
                    store[nid] = {"current": carry[nid]}  # latch deferred to end of pass
                    self._merge_loop_report(per_node, key, nid, 0.0, [])
                    continue
                if inst.type_id == SUBWORKFLOW_TYPE_ID:
                    await self._exec_subworkflow(graph, nid, store, feed, seeded, depth, per_node, prefix)
                    continue
                kwargs, _hashes = self._inputs_for(graph, nid, feed, store, seeded)
                try:
                    out_map, stages, ms = await self._call_fn(graph, nid, kwargs, it)
                    store[nid] = out_map
                    self._merge_loop_report(per_node, key, nid, ms, stages)
                except Exception as e:  # noqa: BLE001 - surfaced in report
                    per_node[key] = NodeReport(nid, "error", ms=0.0, error=f"{type(e).__name__}: {e}")
                    self.events.append(("error", nid))
                    return
            if _is_done(store.get(loop.condition, {}).get("done", 0)):
                done = True
                break
            for nid in body:  # deferred latch: producers have all run this pass
                if graph.nodes[nid].type_id == ACCUMULATE_TYPE_ID:
                    found, nxt = self._resolve(nid, "next", feed, store, seeded)
                    if found and nxt is not None:
                        carry[nid] = nxt
        cond_key = prefix + loop.condition
        if not done:
            per_node[cond_key] = NodeReport(
                loop.condition, "error",
                error=f"loop '{loop.name}' did not finish within max_iterations={cap}",
            )
            self.events.append(("error", loop.condition))
        else:
            self.events.append(("node_done", loop.condition))

    @staticmethod
    def _merge_loop_report(per_node: dict[str, NodeReport], key: str, nid: str, ms: float, stages: list) -> None:
        if key in per_node and per_node[key].status == "ok":
            prev = per_node[key]
            per_node[key] = NodeReport(nid, "ok", ms=prev.ms + ms, stages=[*prev.stages, *stages])
        else:
            per_node[key] = NodeReport(nid, "ok", ms=ms, stages=list(stages))

    # -- graph driver ------------------------------------------------------
    async def _exec_graph(
        self, graph: Graph, store: dict, seeded: dict, depth: int,
        use_cache: bool, per_node: dict[str, NodeReport], prefix: str,
    ) -> None:
        loop_of = {nid: i for i, loop in enumerate(graph.loops) for nid in loop.body}
        feed = self._feed(graph)
        for kind, ref in self._supernode_order(graph, loop_of):
            if kind == "loop":
                await self._run_loop(graph, graph.loops[ref], store, feed, seeded, depth, per_node, prefix)
            else:
                await self._exec_single(graph, ref, store, feed, seeded, depth, 0, use_cache, per_node, prefix)

    async def run(self, dirty_only: bool = True) -> RunReport:
        t0 = time.perf_counter()
        errors = self.graph.validate()
        if errors:
            per = {e.node_id or "__graph__": NodeReport(e.node_id or "__graph__", "error", error=e.message) for e in errors}
            return RunReport(per_node=per, ms=0.0)
        per_node: dict[str, NodeReport] = {}
        await self._exec_graph(self.graph, self.outputs, {}, 0, True, per_node, "")
        return RunReport(per_node=per_node, ms=(time.perf_counter() - t0) * 1000.0)

    def run_blocking(self, dirty_only: bool = True) -> RunReport:
        return asyncio.run(self.run(dirty_only=dirty_only))
