"""Control nodes: loop primitives driven by the executor's loop runner.

* ``control.counter`` — current loop iteration (0 outside loops).
* ``control.accumulate`` — loop-carried state (previous iteration's ``next``,
  ``initial`` on iteration 0; passthrough/initial standalone).
* ``control.end_condition`` — EndConditionNode: computes ``done`` (1/0) from
  counter, numeric threshold, or truthiness. The executor repeats the loop
  body until ``done`` or ``max_iterations`` (default 1000).
"""

from __future__ import annotations

from typing import Any

from ..engine.nodes import ExecCtx, Param, PortDef, node

COUNTER_TYPE_ID = "control.counter"
ACCUMULATE_TYPE_ID = "control.accumulate"
END_CONDITION_TYPE_ID = "control.end_condition"


@node(
    type_id=COUNTER_TYPE_ID,
    title="Counter",
    category="Control",
    color="purple",
    description="Loop iteration index (0 on the first pass, 0 outside loops).",
    inputs=[],
    outputs=[PortDef("count", "Count", "data.NUMBER", "out")],
)
def counter(ctx: ExecCtx) -> float:
    return float(ctx.iteration)


@node(
    type_id=ACCUMULATE_TYPE_ID,
    title="Accumulate",
    category="Control",
    color="purple",
    description="Loop-carried state: emits the previous iteration's value, latches the new one.",
    inputs=[PortDef("next", "Next", "data.ANY", "in")],
    outputs=[PortDef("current", "Current", "data.ANY", "out")],
)
def accumulate(
    ctx: ExecCtx,
    initial: float = Param(0.0, kind="number", label="Initial"),  # type: ignore[no-untyped-def]
    next: Any = None,  # type: ignore[no-untyped-def]  # port key is `next` by design
) -> Any:
    return next if next is not None else initial


@node(
    type_id=END_CONDITION_TYPE_ID,
    title="End Condition",
    category="Control",
    color="purple",
    description="Computes when a loop finishes: done=1 ends the loop after this pass.",
    inputs=[PortDef("value", "Value", "data.ANY", "in")],
    outputs=[PortDef("done", "Done", "data.NUMBER", "out")],
)
def end_condition(
    ctx: ExecCtx,
    value: Any = None,  # type: ignore[no-untyped-def]
    mode: str = Param("counter", kind="select", label="Mode", options=["counter", "threshold", "truthy"]),  # type: ignore[no-untyped-def]
    target: float = Param(10.0, kind="number", label="Target"),  # type: ignore[no-untyped-def]
    tolerance: float = Param(0.0, kind="float_slider", label="Tolerance", min=0, max=10, step=0.01),  # type: ignore[no-untyped-def]
    iterations: int = Param(10, kind="step_slider", label="Iterations", min=1, max=1000, step=1),  # type: ignore[no-untyped-def]
) -> float:
    if mode == "threshold":
        try:
            return 1.0 if abs(float(value) - float(target)) <= abs(float(tolerance)) else 0.0
        except (TypeError, ValueError):
            return 0.0
    if mode == "truthy":
        return 1.0 if value else 0.0
    return 1.0 if (ctx.iteration + 1) >= max(1, int(iterations)) else 0.0
