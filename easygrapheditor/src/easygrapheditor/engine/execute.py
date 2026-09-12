"""Async dependency-wave executor with content caching and cancellation."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .cache import Cache, content_hash
from .graph import SUBWORKFLOW_TYPE_ID, Graph
from .loops import is_feedback_link, loop_body_order
from .nodes import NODE_REGISTRY, ExecCtx
from .subworkflows import DEFAULT_MAX_RECURSIONS, build_inner_graph

ACCUMULATE_TYPE_ID = "control.accumulate"
COUNTER_TYPE_ID = "control.counter"


class CancellationToken:
    """Small cooperative cancellation token usable from sync or async callers."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await asyncio.to_thread(self._event.wait)


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    check = getattr(token, "is_cancelled", None)
    if callable(check):
        return bool(check())
    check = getattr(token, "is_set", None)
    return bool(check()) if callable(check) else bool(getattr(token, "cancelled", False))


def _is_done(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    try:
        return float(value) != 0
    except (TypeError, ValueError):
        return bool(value)


def _hash_value(value: Any) -> str:
    """Hash full payloads, including every ndarray byte, not a preview prefix."""
    return content_hash(value)


@dataclass
class NodeReport:
    node_id: str
    status: str  # ok | cached | skipped | error | cancelled
    ms: float = 0.0
    cache_hit: bool = False
    error: str | None = None
    stages: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class RunReport:
    per_node: dict[str, NodeReport]
    ms: float = 0.0
    result_hash: str = ""

    @property
    def results_hash(self) -> str:
        """Plural alias retained for callers that describe the full result set."""
        return self.result_hash

    def ok(self) -> bool:
        return all(
            report.status in ("ok", "cached", "skipped") for report in self.per_node.values()
        )


class Executor:
    def __init__(
        self,
        graph: Graph,
        cache: Cache | None = None,
        max_concurrency: int = 4,
        max_recursions: int = DEFAULT_MAX_RECURSIONS,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.graph = graph
        self.cache = cache or Cache()
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_recursions = max_recursions
        self.outputs: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.on_event = on_event
        self._param_fingerprints: dict[str, str] = {}
        self._structure: dict[str, Any] | None = None
        self._forced_dirty: set[str] = set()
        self._cancel_token: Any = None

    def force_dirty(self, node_ids: set[str] | list[str] | None = None) -> None:
        """Force selected nodes and their downstream closure to recompute once.

        With no ids, the next run recomputes every current node. This is the
        engine-side operation used by a UI's "forget cache" control.
        """
        self._forced_dirty.update(self.graph.nodes if node_ids is None else node_ids)

    def forget_cache(self) -> None:
        """Clear cached values and force a fresh evaluation on the next run."""
        self.cache.clear()
        self.force_dirty()

    def reset(self, *, clear_cache: bool = False) -> None:
        """Forget execution state, optionally including the shared cache."""
        self.outputs.clear()
        self._param_fingerprints.clear()
        self._structure = None
        self._forced_dirty.clear()
        if clear_cache:
            self.cache.clear()

    def _emit(self, event: str, node_id: str | None = None, **detail: Any) -> None:
        item: dict[str, Any] = {"event": event, "node_id": node_id, **detail}
        self.events.append(item)
        if self.on_event is not None:
            self.on_event(item)

    def _effective_params(
        self, graph: Graph, node_id: str, *, hash_only: bool = False
    ) -> dict[str, Any]:
        inst = graph.nodes[node_id]
        ndef = NODE_REGISTRY.get(inst.type_id)
        if ndef is None:
            return dict(inst.params)
        params: dict[str, Any] = {}
        defined = {param.key: param for param in ndef.params}
        for param in ndef.params:
            if not hash_only or param.affects_hash:
                params[param.key] = inst.params.get(param.key, param.default)
        # Preserve custom params from imported/user nodes. Unknown params affect
        # the cache because the engine cannot safely know they are cosmetic.
        for key, value in inst.params.items():
            if key not in defined:
                params[key] = value
        return params

    def _node_hash(self, graph: Graph, node_id: str, input_hashes: dict[str, str]) -> str:
        inst = graph.nodes[node_id]
        ndef = NODE_REGISTRY[inst.type_id]
        if not ndef.cacheable:
            return f"nocache:{node_id}:{time.time_ns()}"
        return content_hash(
            {
                "node_id": node_id,
                "type": inst.type_id,
                "params": self._effective_params(graph, node_id, hash_only=True),
                "inputs": input_hashes,
            }
        )

    def _param_fingerprint(self, graph: Graph, node_id: str) -> str:
        inst = graph.nodes[node_id]
        return content_hash(
            {"type": inst.type_id, "params": self._effective_params(graph, node_id, hash_only=True)}
        )

    @staticmethod
    def _feed(graph: Graph) -> dict[tuple[str, str], tuple[str, str]]:
        return {
            (link.to_node, link.to_port): (link.from_node, link.from_port) for link in graph.links
        }

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
        source = feed.get((nid, port))
        if source is None:
            return False, None
        return True, store.get(source[0], {}).get(source[1])

    def _inputs_for(
        self, graph: Graph, nid: str, feed: dict, store: dict, seeded: dict
    ) -> tuple[dict, dict]:
        ndef = NODE_REGISTRY[graph.nodes[nid].type_id]
        kwargs: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        for port in ndef.inputs:
            found, value = self._resolve(nid, port.key, feed, store, seeded)
            if found:
                kwargs[port.key] = value
                hashes[port.key] = _hash_value(value)
        return kwargs, hashes

    async def _call_fn(
        self, graph: Graph, nid: str, kwargs: dict[str, Any], iteration: int
    ) -> tuple[dict, list, float]:
        inst = graph.nodes[nid]
        ndef = NODE_REGISTRY[inst.type_id]
        ctx = ExecCtx(
            node_id=nid,
            params=self._effective_params(graph, nid),
            iteration=iteration,
            cancel_token=self._cancel_token,
            _on_stage=lambda stage, frac: self._emit("stage", nid, stage=stage, frac=frac),
        )
        ctx.check_cancelled()
        call_kwargs = {**ctx.params, **kwargs}
        sig = inspect.signature(ndef.fn)
        accepted = {name for name in sig.parameters if name != "ctx"}
        call_kwargs = {name: value for name, value in call_kwargs.items() if name in accepted}
        started = time.perf_counter()
        result = ndef.fn(ctx, **call_kwargs) if "ctx" in sig.parameters else ndef.fn(**call_kwargs)
        if inspect.isawaitable(result):
            result = await result
        ctx.check_cancelled()
        if not ndef.outputs:
            output_map: dict[str, Any] = {}
        elif len(ndef.outputs) == 1:
            output_map = {ndef.outputs[0].key: result}
        elif isinstance(result, dict):
            output_map = result
        elif isinstance(result, (tuple, list)) and len(result) == len(ndef.outputs):
            output_map = {port.key: value for port, value in zip(ndef.outputs, result)}
        else:
            raise TypeError(f"node '{nid}' must return a dict or {len(ndef.outputs)} values")
        return output_map, list(ctx._stages), (time.perf_counter() - started) * 1000.0

    async def _exec_single(
        self,
        graph: Graph,
        nid: str,
        store: dict,
        feed: dict,
        seeded: dict,
        depth: int,
        iteration: int,
        use_cache: bool,
        per_node: dict[str, NodeReport],
        prefix: str,
        should_run: bool = True,
        bypass_cache: bool = False,
    ) -> None:
        key = prefix + nid
        if _cancelled(self._cancel_token):
            per_node[key] = NodeReport(nid, "cancelled", error="run cancelled")
            self._emit("cancelled", nid)
            return
        if graph.nodes[nid].type_id == SUBWORKFLOW_TYPE_ID:
            await self._exec_subworkflow(graph, nid, store, feed, seeded, depth, per_node, prefix)
            return
        kwargs, hashes = self._inputs_for(graph, nid, feed, store, seeded)
        node_hash = self._node_hash(graph, nid, hashes)
        cached = self.cache.get(node_hash) if use_cache and not bypass_cache else None
        if cached is not None:
            store[nid] = cached["outputs"]
            per_node[key] = NodeReport(
                nid, "cached", cache_hit=True, stages=cached.get("stages", [])
            )
            self._emit("cache_hit", nid)
            return
        if not should_run and nid in store:
            per_node[key] = NodeReport(nid, "skipped")
            self._emit("node_skipped", nid)
            return
        self._emit("node_started", nid)
        try:
            output_map, stages, ms = await self._call_fn(graph, nid, kwargs, iteration)
        except asyncio.CancelledError:
            per_node[key] = NodeReport(nid, "cancelled", error="run cancelled")
            self._emit("cancelled", nid)
            return
        except Exception as exc:  # noqa: BLE001 - user node errors belong in RunReport
            per_node[key] = NodeReport(nid, "error", error=f"{type(exc).__name__}: {exc}")
            self._emit("error", nid, error=per_node[key].error)
            return
        store[nid] = output_map
        if use_cache:
            self.cache.set(node_hash, {"outputs": output_map, "stages": stages})
        per_node[key] = NodeReport(nid, "ok", ms=ms, stages=stages)
        self._emit("node_done", nid, ms=ms)

    async def _exec_subworkflow(
        self,
        graph: Graph,
        nid: str,
        store: dict,
        feed: dict,
        seeded: dict,
        depth: int,
        per_node: dict[str, NodeReport],
        prefix: str,
    ) -> None:
        key = prefix + nid
        if depth >= self.max_recursions:
            per_node[key] = NodeReport(
                nid, "error", error=f"max recursions ({self.max_recursions}) exceeded at '{nid}'"
            )
            self._emit("error", nid, error=per_node[key].error)
            return
        self._emit("node_started", nid)
        inst = graph.nodes[nid]
        inner = build_inner_graph(inst)
        seeded_inner: dict[tuple[str, str], Any] = {}
        for mapping in inst.params.get("inputs", []):
            found, value = self._resolve(nid, mapping["key"], feed, store, seeded)
            if found:
                seeded_inner[(mapping["inner_node"], mapping["inner_port"])] = value
        inner_store: dict[str, dict[str, Any]] = {}
        await self._exec_graph(
            inner, inner_store, seeded_inner, depth + 1, True, per_node, prefix + nid + "/"
        )
        inner_reports = [
            report
            for report_key, report in per_node.items()
            if report_key.startswith(prefix + nid + "/")
        ]
        if any(report.status not in ("ok", "cached", "skipped") for report in inner_reports):
            per_node[key] = NodeReport(nid, "error", error=f"inner error in subworkflow '{nid}'")
            self._emit("error", nid, error=per_node[key].error)
            return
        store[nid] = {
            mapping["key"]: inner_store.get(mapping["inner_node"], {}).get(mapping["inner_port"])
            for mapping in inst.params.get("outputs", [])
        }
        elapsed = sum(report.ms for report in inner_reports)
        per_node[key] = NodeReport(nid, "ok", ms=elapsed)
        self._emit("node_done", nid, ms=elapsed)

    @staticmethod
    def _supernodes(
        graph: Graph,
    ) -> tuple[
        list[tuple[str, Any]],
        dict[str, tuple[str, Any]],
        dict[tuple[str, Any], set[tuple[str, Any]]],
    ]:
        loop_of = {nid: index for index, loop in enumerate(graph.loops) for nid in loop.body}
        by_node = {
            nid: ("loop", loop_of[nid]) if nid in loop_of else ("node", nid) for nid in graph.nodes
        }
        supers = list(dict.fromkeys(by_node.values()))
        parents = {sup: set() for sup in supers}
        type_of = lambda nid: graph.nodes[nid].type_id if nid in graph.nodes else None
        for link in graph.links:
            src, dst = by_node.get(link.from_node), by_node.get(link.to_node)
            if (
                src is not None
                and dst is not None
                and src != dst
                and not is_feedback_link(link, type_of)
            ):
                parents[dst].add(src)
        return supers, by_node, parents

    async def _run_loop(
        self,
        graph: Graph,
        loop: Any,
        store: dict,
        feed: dict,
        seeded: dict,
        depth: int,
        per_node: dict[str, NodeReport],
        prefix: str,
    ) -> None:
        body = [nid for nid in loop.body if nid in graph.nodes]
        type_of = lambda nid: graph.nodes[nid].type_id if nid in graph.nodes else None
        try:
            order = loop_body_order(body, graph.links, type_of)
        except ValueError as exc:
            per_node[prefix + loop.condition] = NodeReport(loop.condition, "error", error=str(exc))
            return
        carry = {
            nid: graph.nodes[nid].params.get("initial", 0.0)
            for nid in body
            if graph.nodes[nid].type_id == ACCUMULATE_TYPE_ID
        }
        for iteration in range(max(1, int(loop.max_iterations))):
            if _cancelled(self._cancel_token):
                for nid in body:
                    per_node[prefix + nid] = NodeReport(nid, "cancelled", error="run cancelled")
                return
            for nid in order:
                key = prefix + nid
                kind = graph.nodes[nid].type_id
                if kind == COUNTER_TYPE_ID:
                    store[nid] = {"count": float(iteration)}
                    self._merge_loop_report(per_node, key, nid, 0.0, [])
                    continue
                if kind == ACCUMULATE_TYPE_ID:
                    store[nid] = {"current": carry[nid]}
                    self._merge_loop_report(per_node, key, nid, 0.0, [])
                    continue
                await self._exec_single(
                    graph, nid, store, feed, seeded, depth, iteration, False, per_node, prefix
                )
                if per_node[key].status not in ("ok", "cached"):
                    return
            if _is_done(store.get(loop.condition, {}).get("done", 0)):
                return
            for nid in carry:
                found, value = self._resolve(nid, "next", feed, store, seeded)
                if found and value is not None:
                    carry[nid] = value
        per_node[prefix + loop.condition] = NodeReport(
            loop.condition,
            "error",
            error=f"loop '{loop.name}' did not finish within max_iterations={loop.max_iterations}",
        )
        self._emit("error", loop.condition, error=per_node[prefix + loop.condition].error)

    @staticmethod
    def _merge_loop_report(
        per_node: dict[str, NodeReport], key: str, nid: str, ms: float, stages: list
    ) -> None:
        previous = per_node.get(key)
        if previous is not None and previous.status == "ok":
            per_node[key] = NodeReport(
                nid, "ok", ms=previous.ms + ms, stages=[*previous.stages, *stages]
            )
        else:
            per_node[key] = NodeReport(nid, "ok", ms=ms, stages=list(stages))

    @staticmethod
    def _descendants(graph: Graph, roots: set[str]) -> set[str]:
        """Return roots plus their non-feedback downstream closure."""
        dirty = {nid for nid in roots if nid in graph.nodes}
        pending = list(dirty)
        adj: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
        type_of = lambda nid: graph.nodes[nid].type_id if nid in graph.nodes else None
        for link in graph.links:
            if (
                link.from_node in adj
                and link.to_node in adj
                and not is_feedback_link(link, type_of)
            ):
                adj[link.from_node].append(link.to_node)
        while pending:
            nid = pending.pop()
            for child in adj[nid]:
                if child not in dirty:
                    dirty.add(child)
                    pending.append(child)
        return dirty

    @staticmethod
    def _structure_snapshot(graph: Graph) -> dict[str, Any]:
        """Execution-relevant graph structure, excluding canvas-only state."""
        return {
            "nodes": {nid: inst.type_id for nid, inst in graph.nodes.items()},
            "links": {
                (link.from_node, link.from_port, link.to_node, link.to_port) for link in graph.links
            },
            "loops": {
                (loop.name, tuple(loop.body), loop.condition, loop.max_iterations)
                for loop in graph.loops
            },
        }

    def _structural_dirty(self, graph: Graph, current: dict[str, Any]) -> set[str]:
        previous = self._structure
        if previous is None:
            return set(graph.nodes)
        roots: set[str] = {
            nid
            for nid, type_id in current["nodes"].items()
            if previous["nodes"].get(nid) != type_id
        }
        for from_node, _from_port, to_node, _to_port in current["links"] ^ previous["links"]:
            # The receiver observes a new input contract. A removed source has
            # no current output to invalidate independently.
            if to_node in graph.nodes:
                roots.add(to_node)
            if from_node in graph.nodes and from_node not in previous["nodes"]:
                roots.add(from_node)
        if current["loops"] != previous["loops"]:
            old_members = {
                nid for _name, body, _condition, _cap in previous["loops"] for nid in body
            }
            new_members = {
                nid for _name, body, _condition, _cap in current["loops"] for nid in body
            }
            roots.update(nid for nid in old_members | new_members if nid in graph.nodes)
        return self._descendants(graph, roots)

    def _prune_deleted_state(self) -> None:
        live = set(self.graph.nodes)
        for mapping in (self.outputs, self._param_fingerprints):
            for nid in set(mapping) - live:
                del mapping[nid]
        self._forced_dirty.intersection_update(live)

    def _dirty_nodes(self, graph: Graph, dirty_only: bool, structural_dirty: set[str]) -> set[str]:
        if not dirty_only or set(graph.nodes) - set(self._param_fingerprints):
            return set(graph.nodes)
        changed = {
            nid
            for nid in graph.nodes
            if self._param_fingerprints.get(nid) != self._param_fingerprint(graph, nid)
        }
        return self._descendants(graph, changed | structural_dirty | self._forced_dirty)

    async def _exec_graph(
        self,
        graph: Graph,
        store: dict,
        seeded: dict,
        depth: int,
        use_cache: bool,
        per_node: dict[str, NodeReport],
        prefix: str,
        dirty_only: bool = False,
    ) -> None:
        feed = self._feed(graph)
        supers, _by_node, parents = self._supernodes(graph)
        structural_dirty = (
            self._structural_dirty(graph, self._structure_snapshot(graph))
            if graph is self.graph
            else set(graph.nodes)
        )
        dirty = (
            self._dirty_nodes(graph, dirty_only, structural_dirty)
            if graph is self.graph
            else set(graph.nodes)
        )
        forced_compute = structural_dirty | self._descendants(graph, self._forced_dirty)
        completed: set[tuple[str, Any]] = set()
        failed: set[tuple[str, Any]] = set()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        while len(completed) < len(supers):
            ready = [sup for sup in supers if sup not in completed and parents[sup] <= completed]
            if not ready:
                raise ValueError("Graph has a cycle outside loops; cannot order.")

            async def run_super(sup: tuple[str, Any]) -> tuple[tuple[str, Any], bool]:
                kind, ref = sup
                if parents[sup] & failed:
                    ids = graph.loops[ref].body if kind == "loop" else [ref]
                    for nid in ids:
                        per_node[prefix + nid] = NodeReport(
                            nid, "skipped", error="upstream node failed"
                        )
                        self._emit("node_skipped", nid, reason="upstream node failed")
                    return sup, False
                async with semaphore:
                    if kind == "loop":
                        await self._run_loop(
                            graph, graph.loops[ref], store, feed, seeded, depth, per_node, prefix
                        )
                        ids = graph.loops[ref].body
                    else:
                        await self._exec_single(
                            graph,
                            ref,
                            store,
                            feed,
                            seeded,
                            depth,
                            0,
                            use_cache,
                            per_node,
                            prefix,
                            should_run=ref in dirty,
                            bypass_cache=ref in forced_compute,
                        )
                        ids = [ref]
                return sup, all(
                    per_node.get(prefix + nid, NodeReport(nid, "error")).status
                    in ("ok", "cached", "skipped")
                    for nid in ids
                )

            tasks = [asyncio.create_task(run_super(sup)) for sup in ready]
            while any(not task.done() for task in tasks):
                if _cancelled(self._cancel_token):
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                await asyncio.wait(tasks, timeout=0.01)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for index, result in enumerate(results):
                if isinstance(result, BaseException):
                    sup = ready[index]
                    kind, ref = sup
                    ids = graph.loops[ref].body if kind == "loop" else [ref]
                    for nid in ids:
                        per_node[prefix + nid] = NodeReport(nid, "cancelled", error="run cancelled")
                    results[index] = (sup, False)
            for sup, ok in results:
                completed.add(sup)
                if not ok:
                    failed.add(sup)

    async def run(self, dirty_only: bool = True, cancel_token: Any = None) -> RunReport:
        started = time.perf_counter()
        self.events = []
        self._cancel_token = cancel_token
        self._prune_deleted_state()
        errors = self.graph.validate()
        if errors:
            per_node: dict[str, NodeReport] = {}
            for error in errors:
                base = error.node_id or "__graph__"
                key = base
                suffix = 2
                while key in per_node:
                    key = f"{base}#{suffix}"
                    suffix += 1
                per_node[key] = NodeReport(base, "error", error=error.message)
            self._emit("run_done", None, ok=False, ms=0.0)
            return RunReport(per_node, 0.0, content_hash({"outputs": self.outputs}))
        per_node: dict[str, NodeReport] = {}
        await self._exec_graph(
            self.graph, self.outputs, {}, 0, True, per_node, "", dirty_only=dirty_only
        )
        self._param_fingerprints = {
            nid: self._param_fingerprint(self.graph, nid) for nid in self.graph.nodes
        }
        self._structure = self._structure_snapshot(self.graph)
        self._forced_dirty.clear()
        elapsed = (time.perf_counter() - started) * 1000.0
        report = RunReport(per_node, elapsed, content_hash({"outputs": self.outputs}))
        self._emit("run_done", None, ok=report.ok(), ms=elapsed)
        return report

    def run_blocking(self, dirty_only: bool = True, cancel_token: Any = None) -> RunReport:
        return asyncio.run(self.run(dirty_only=dirty_only, cancel_token=cancel_token))
