"""Node definition + @node decorator. See editor.md §4.2."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .types import Field, Image, Number

ParamKind = Literal[
    "slider",  # legacy: continuous decimal slider
    "float_slider",  # continuous/decimal slider (min/max/step)
    "step_slider",  # stepped slider (snaps to step)
    "int",  # integer numerical input
    "number",  # decimal numerical input box
    "seed",  # numerical input with randomize affordance
    "text",  # single-line input box
    "textarea",  # multi-line input box
    "multiline",  # legacy alias of textarea
    "dropdown",  # option picker
    "select",  # alias of dropdown
    "toggle",  # boolean switch
    "checkbox",  # alias of toggle
    "file",  # file picker (stores path string)
]

# Aliases -> canonical kind. UI layers switch on the canonical form.
KIND_ALIASES: dict[str, str] = {
    "slider": "float_slider",
    "multiline": "textarea",
    "dropdown": "select",
    "toggle": "checkbox",
}


def canonical_kind(kind: str) -> str:
    """Normalize legacy/alias param kinds to the canonical widget name."""
    return KIND_ALIASES.get(kind, kind)


@dataclass
class ParamDef:
    key: str
    label: str
    kind: ParamKind = "number"
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    affects_hash: bool = True


@dataclass
class PortDef:
    key: str
    label: str
    dtype: str  # e.g. "data.NUMBER"
    direction: Literal["in", "out"]


@dataclass
class ExecCtx:
    node_id: str
    params: dict[str, Any]
    iteration: int = 0  # loop iteration index (0 outside loops); drives control.counter etc.
    _stages: list[tuple[str, float]] = field(default_factory=list)

    def report(self, stage: str, frac: float) -> None:
        self._stages.append((stage, frac))


@dataclass
class NodeDef:
    type_id: str
    title: str
    category: str
    color: str
    description: str
    params: list[ParamDef]
    inputs: list[PortDef]
    outputs: list[PortDef]
    fn: Callable
    cacheable: bool = True
    badges: list[str] = field(default_factory=list)


NODE_REGISTRY: dict[str, NodeDef] = {}


def _hint_to_dtype(ann: Any) -> str:
    if ann in (float, int, Number, "Number"):
        return "data.NUMBER"
    if ann is Field or getattr(ann, "__name__", "") == "Field":
        return "data.FIELD"
    if ann is Image or getattr(ann, "__name__", "") == "Image":
        return "data.IMAGE"
    return "data.ANY"


def Param(
    default: Any = None,
    kind: ParamKind = "number",
    label: str = "",
    min: float | None = None,
    max: float | None = None,
    step: float | None = None,
    options: list[str] | None = None,
) -> ParamDef:
    """Declare a rich param widget. Used as a default value marker."""
    return ParamDef(key="", label=label, kind=kind, default=default, min=min, max=max, step=step, options=options)


def node(
    _fn: Callable | None = None,
    *,
    type_id: str | None = None,
    title: str | None = None,
    category: str = "Misc",
    color: str = "grey",
    description: str = "",
    cacheable: bool = True,
    badges: list[str] | None = None,
    params: list[ParamDef] | None = None,
    inputs: list[PortDef] | None = None,
    outputs: list[PortDef] | None = None,
) -> Any:
    """Register a Python function as a graph node.

    Ports are inferred from type hints; params from defaults.
    Explicit params/inputs/outputs override inference (stub: minimal inference).
    """

    def wrap(fn: Callable) -> Callable:
        tid = type_id or f"{category.lower()}.{fn.__name__}"
        sig = inspect.signature(fn)
        inferred_params: list[ParamDef] = []
        inferred_inputs: list[PortDef] = []
        ret_dtype = "data.NUMBER"
        for name, p in sig.parameters.items():
            if name == "ctx":
                continue
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else Any
            default = p.default if p.default is not inspect.Parameter.empty else None
            if isinstance(default, ParamDef):
                default.key = default.key or name
                default.label = default.label or name
                inferred_params.append(default)
                # ParamDefs also imply an input port if hinted as Field/Image/Number?
                # Stub rule: ParamDef-marked args are params, not ports.
                continue
            dtype = _hint_to_dtype(ann)
            if p.default is not inspect.Parameter.empty:
                inferred_params.append(ParamDef(key=name, label=name, default=default))
            else:
                inferred_inputs.append(PortDef(key=name, label=name, dtype=dtype, direction="in"))
        if sig.return_annotation is not inspect.Signature.empty:
            ret_dtype = _hint_to_dtype(sig.return_annotation)
        inferred_outputs = [PortDef(key="out", label="Out", dtype=ret_dtype, direction="out")]
        ndef = NodeDef(
            type_id=tid,
            title=title or fn.__name__.replace("_", " ").title(),
            category=category,
            color=color,
            description=description or (fn.__doc__ or "").strip(),
            params=params if params is not None else inferred_params,
            inputs=inputs if inputs is not None else inferred_inputs,
            outputs=outputs if outputs is not None else inferred_outputs,
            fn=fn,
            cacheable=cacheable,
            badges=badges or [],
        )
        NODE_REGISTRY[tid] = ndef
        fn._node_def = ndef  # type: ignore[attr-defined]
        return fn

    if _fn is not None:
        return wrap(_fn)
    return wrap


def list_nodes() -> list[NodeDef]:
    return list(NODE_REGISTRY.values())


def get_node(type_id: str) -> NodeDef | None:
    return NODE_REGISTRY.get(type_id)
