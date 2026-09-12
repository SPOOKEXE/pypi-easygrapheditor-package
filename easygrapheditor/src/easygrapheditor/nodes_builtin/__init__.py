"""Built-in nodes: maths, async demo, and full numpy terrain pipeline.

Terrain chain mirrors plans/15-41-09:
  Noise + Ridged -> Domain Warp -> Combine -> Erode -> Slope/Colourise -> Readout
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np

from ..engine.nodes import ExecCtx, Param, PortDef, node
from ..engine.types import Field, Image


# ---------------------------------------------------------------- maths/async
@node(category="Input", color="blue", description="Constant number.")
def number(value: float = Param(1.0, kind="number", label="Value")) -> float:  # type: ignore[no-untyped-def]
    return float(value)


@node(category="Maths", color="orange", description="Add/subtract two numbers.")
def arithmetic(
    a: float,
    b: float,
    operation: str = Param("add", kind="select", label="Operation", options=["add", "subtract"]),  # type: ignore[no-untyped-def]
    absolute: bool = Param(False, kind="checkbox", label="Absolute"),  # type: ignore[no-untyped-def]
) -> float:
    res = a + b if operation == "add" else a - b
    return abs(res) if absolute else res


@node(
    category="Simulate",
    color="purple",
    description="Async staged task demo (planning/fetching/...).",
    inputs=[PortDef("after", "After", "data.ANY", "in", required=False)],
    outputs=[PortDef("done", "Done", "data.NUMBER", "out")],
)
async def staged_task(
    ctx: ExecCtx,
    after: Any = None,
    seconds: float = Param(2.0, kind="number", label="Seconds"),  # type: ignore[no-untyped-def]
    label: str = Param("left", kind="text", label="Label"),  # type: ignore[no-untyped-def]
) -> float:
    _ = after
    stages = ["planning", "fetching", "thinking", "writing", "checking"]
    for i, stage in enumerate(stages):
        ctx.check_cancelled()
        ctx.report(stage, (i + 1) / len(stages))
        await asyncio.sleep(float(seconds) / len(stages))
    return float(seconds)


# ---------------------------------------------------------------- noise core
def _bilinear_upsample(lattice: np.ndarray, res: int) -> np.ndarray:
    """Upsample (gh x gw) lattice to (res x res) via bilinear interpolation."""
    gh, gw = lattice.shape
    ys = np.linspace(0, gh - 1, res)
    xs = np.linspace(0, gw - 1, res)
    y0 = np.floor(ys).astype(int).clip(0, gh - 2)
    x0 = np.floor(xs).astype(int).clip(0, gw - 2)
    fy = (ys - y0).reshape(res, 1)
    fx = (xs - x0).reshape(1, res)
    q00 = lattice[y0[:, None], x0[None, :]]
    q01 = lattice[y0[:, None], x0[None, :] + 1]
    q10 = lattice[y0[:, None] + 1, x0[None, :]]
    q11 = lattice[y0[:, None] + 1, x0[None, :] + 1]
    return (
        q00 * (1 - fy) * (1 - fx) + q01 * (1 - fy) * fx + q10 * fy * (1 - fx) + q11 * fy * fx
    ).astype(np.float32)


def _value_noise(
    res: int, freq: float, seed: float, octaves: int, gain: float, ridged: bool
) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1000 + 7919) % (2**32 - 1))
    out = np.zeros((res, res), dtype=np.float64)
    amp, f, total = 1.0, max(1, round(freq)), 0.0
    for _ in range(max(1, octaves)):
        lattice = rng.random((f + 1, f + 1))
        layer = _bilinear_upsample(lattice, res).astype(np.float64)
        if ridged:
            layer = 1.0 - np.abs(2.0 * layer - 1.0)
        out += amp * layer
        total += amp
        amp *= gain
        f *= 2
    out /= max(total, 1e-9)
    return np.clip(out, 0, 1).astype(np.float32)


def _sample_bilinear(field: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = field.shape
    ys = np.clip(ys, 0, h - 1.001)
    xs = np.clip(xs, 0, w - 1.001)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = ys - y0, xs - x0
    # all 2D (h x w); keep weights 2D so output stays 2D
    q00 = field[y0, x0]
    q01 = field[y0, np.clip(x0 + 1, 0, w - 1)]
    q10 = field[np.clip(y0 + 1, 0, h - 1), x0]
    q11 = field[np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)]
    return (
        q00 * (1 - fy) * (1 - fx) + q01 * (1 - fy) * fx + q10 * fy * (1 - fx) + q11 * fy * fx
    ).astype(np.float32)


@node(category="Generate", color="blue", description="Value noise field 0..1.")
def noise(
    resolution: int = Param(64, kind="step_slider", label="Resolution", min=8, max=256, step=8),  # type: ignore[no-untyped-def]
    frequency: float = Param(4.0, kind="float_slider", label="Frequency", min=1, max=16, step=0.5),  # type: ignore[no-untyped-def]
    seed: float = Param(7.0, kind="seed", label="Seed"),  # type: ignore[no-untyped-def]
    octaves: int = Param(4, kind="step_slider", label="Octaves", min=1, max=8, step=1),  # type: ignore[no-untyped-def]
    gain: float = Param(0.5, kind="float_slider", label="Gain", min=0.1, max=1.0, step=0.05),  # type: ignore[no-untyped-def]
) -> Field:
    res = int(resolution)
    return Field(
        data=_value_noise(res, float(frequency), float(seed), int(octaves), float(gain), False)
    )


@node(category="Generate", color="blue", description="Ridged value noise field 0..1.")
def ridged(
    resolution: int = Param(64, kind="step_slider", label="Resolution", min=8, max=256, step=8),  # type: ignore[no-untyped-def]
    frequency: float = Param(4.0, kind="float_slider", label="Frequency", min=1, max=16, step=0.5),  # type: ignore[no-untyped-def]
    seed: float = Param(7.0, kind="seed", label="Seed"),  # type: ignore[no-untyped-def]
    octaves: int = Param(5, kind="step_slider", label="Octaves", min=1, max=8, step=1),  # type: ignore[no-untyped-def]
    gain: float = Param(0.5, kind="float_slider", label="Gain", min=0.1, max=1.0, step=0.05),  # type: ignore[no-untyped-def]
) -> Field:
    res = int(resolution)
    return Field(
        data=_value_noise(res, float(frequency), float(seed), int(octaves), float(gain), True)
    )


@node(
    type_id="filter.domain_warp",
    title="Domain Warp",
    category="Filter",
    color="green",
    description="Warp one field by another (amount in pixels).",
    inputs=[
        PortDef("field_in", "In", "data.FIELD", "in"),
        PortDef("warp_by", "By", "data.FIELD", "in"),
    ],
    outputs=[PortDef("out", "Out", "data.FIELD", "out")],
)
def domain_warp(
    field_in: Field,
    warp_by: Field,
    amount: float = Param(12.0, kind="float_slider", label="Amount", min=0, max=32, step=0.5),
) -> Field:  # type: ignore[no-untyped-def]
    h, w = field_in.data.shape
    # Second field drives an angle + magnitude warp; centred so mean-zero warps less.
    yy, xx = np.mgrid[0:h, 0:w]
    ang = (warp_by.data - 0.5) * 2 * math.pi
    mag = (warp_by.data) * float(amount)
    ys = yy + np.sin(ang) * mag
    xs = xx + np.cos(ang) * mag
    return Field(data=_sample_bilinear(field_in.data, ys, xs))


@node(
    type_id="filter.combine",
    title="Combine",
    category="Filter",
    color="green",
    description="Combine two fields (maximum/add/multiply/min).",
    inputs=[PortDef("a", "A", "data.FIELD", "in"), PortDef("b", "B", "data.FIELD", "in")],
    outputs=[PortDef("out", "Out", "data.FIELD", "out")],
)
def combine(
    a: Field,
    b: Field,
    amount: float = Param(0.5, kind="float_slider", label="Amount", min=0, max=1, step=0.05),  # type: ignore[no-untyped-def]
    mode: str = Param(
        "maximum", kind="select", label="Mode", options=["maximum", "add", "multiply", "min"]
    ),  # type: ignore[no-untyped-def]
) -> Field:
    t = float(amount)
    if mode == "add":
        out = a.data * (1 - t) + np.clip(a.data + b.data, 0, 1) * t
    elif mode == "multiply":
        out = a.data * (1 - t) + (a.data * b.data) * t
    elif mode == "min":
        out = np.minimum(a.data, b.data)
    else:
        out = np.maximum(a.data, b.data * t + a.data * (1 - t))
    return Field(data=np.clip(out, 0, 1).astype(np.float32))


@node(
    type_id="simulate.erode",
    title="Erode",
    category="Simulate",
    color="orange",
    description="Simplified thermal erosion (talus slope iterations).",
    inputs=[PortDef("field_in", "In", "data.FIELD", "in")],
    outputs=[PortDef("out", "Out", "data.FIELD", "out")],
)
def erode(
    field_in: Field,
    thermal_passes: float = Param(
        30.0, kind="step_slider", label="Thermal passes", min=0, max=200, step=1
    ),  # type: ignore[no-untyped-def]
    talus: float = Param(0.01, kind="float_slider", label="Talus", min=0, max=0.2, step=0.005),  # type: ignore[no-untyped-def]
    spin_passes: float = Param(
        20.0, kind="step_slider", label="Spin passes", min=0, max=100, step=1
    ),  # type: ignore[no-untyped-def]
) -> Field:
    h = field_in.data.astype(np.float32).copy()
    passes = int(min(max(float(thermal_passes), 0), 200))
    tal = float(talus)
    for _ in range(passes):
        # 4-neighbour differences; move half the excess over talus.
        up = np.roll(h, 1, axis=0)
        dn = np.roll(h, -1, axis=0)
        lf = np.roll(h, 1, axis=1)
        rt = np.roll(h, -1, axis=1)
        for nb in (up, dn, lf, rt):
            diff = h - nb
            move = np.clip((diff - tal) * 0.25, 0, None)
            h -= move * 0.5
    # spin passes: light smoothing
    for _ in range(int(min(max(float(spin_passes), 0), 100))):
        h = (h + np.roll(h, 1, 0) + np.roll(h, -1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 1)) / 5.0
    return Field(data=np.clip(h, 0, 1).astype(np.float32))


@node(
    type_id="filter.slope",
    title="Slope",
    category="Filter",
    color="green",
    description="Gradient magnitude of a field, normalised 0..1.",
    inputs=[PortDef("field_in", "In", "data.FIELD", "in")],
    outputs=[PortDef("out", "Out", "data.FIELD", "out")],
)
def slope(field_in: Field) -> Field:
    gy, gx = np.gradient(field_in.data.astype(np.float32))
    mag = np.sqrt(gx**2 + gy**2)
    m = mag.max()
    return Field(data=(mag / m if m > 0 else mag).astype(np.float32))


_ALPINE = np.array(
    [
        [30, 70, 160],  # deep water
        [60, 120, 200],  # shallow water
        [70, 140, 90],  # lowland
        [110, 160, 100],  # forest
        [150, 150, 130],  # rock
        [240, 240, 245],  # snow
    ],
    dtype=np.float32,
)


@node(
    type_id="colour.colourise",
    title="Colourise",
    category="Colour",
    color="pink",
    description="Height becomes a picture (alpine palette visualisation).",
    inputs=[PortDef("field_in", "In", "data.FIELD", "in")],
    outputs=[PortDef("out", "Out", "data.IMAGE", "out")],
)
def colourise(
    field_in: Field,
    palette: str = Param("alpine", kind="select", label="Palette", options=["alpine"]),  # type: ignore[no-untyped-def]
    sea_level: float = Param(0.32, kind="float_slider", label="Sea level", min=0, max=1, step=0.01),  # type: ignore[no-untyped-def]
    relief: bool = Param(False, kind="checkbox", label="Relief"),  # type: ignore[no-untyped-def]
) -> Image:
    _ = palette
    f = np.clip(field_in.data, 0, 1)
    sea = float(sea_level)
    # Map height to palette stops: [0, sea) water, [sea, 1] land.
    t = np.empty_like(f)
    t[f < sea] = (f[f < sea] / max(sea, 1e-6)) * 0.33
    t[f >= sea] = 0.33 + ((f[f >= sea] - sea) / max(1 - sea, 1e-6)) * 0.67
    pos = t * (len(_ALPINE) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(_ALPINE) - 2)
    frac = (pos - lo)[..., None]
    rgb = _ALPINE[lo] * (1 - frac) + _ALPINE[np.clip(lo + 1, 0, len(_ALPINE) - 1)] * frac
    if relief:
        gy, gx = np.gradient(f)
        shade = 1.0 - np.clip(np.sqrt(gx**2 + gy**2) * 4.0, 0, 0.45)
        rgb *= shade[..., None]
    return Image(data=np.clip(rgb, 0, 255).astype(np.uint8))


@node(
    type_id="output.readout",
    title="Readout",
    category="Output",
    color="grey",
    description="Scalar readout (mean of field / R of image / number).",
    inputs=[PortDef("value_in", "In", "data.ANY", "in")],
    outputs=[PortDef("value", "Value", "data.NUMBER", "out")],
)
def readout(value_in) -> float:  # type: ignore[no-untyped-def]
    if isinstance(value_in, Field):
        return float(value_in.data.mean())
    if isinstance(value_in, Image):
        return float(value_in.data[..., 0].mean())
    if isinstance(value_in, (int, float, np.floating)):
        return float(value_in)
    return 0.0


@node(
    type_id="filter.terrace",
    title="Terrace",
    category="Filter",
    color="green",
    description="Quantise a field into steps.",
    inputs=[PortDef("field_in", "In", "data.FIELD", "in")],
    outputs=[PortDef("out", "Out", "data.FIELD", "out")],
)
def terrace(
    field_in: Field,
    steps: float = Param(5.0, kind="step_slider", label="Steps", min=2, max=16, step=1),
) -> Field:  # type: ignore[no-untyped-def]
    s = max(2, int(float(steps)))
    return Field(data=(np.floor(field_in.data * s) / (s - 1)).clip(0, 1).astype(np.float32))


@node(
    type_id="filter.threshold",
    title="Threshold",
    category="Filter",
    color="green",
    description="Binary mask at cutoff.",
    inputs=[PortDef("field_in", "In", "data.FIELD", "in")],
    outputs=[PortDef("out", "Out", "data.FIELD", "out")],
)
def threshold(
    field_in: Field,
    cutoff: float = Param(0.5, kind="float_slider", label="Cutoff", min=0, max=1, step=0.01),
) -> Field:  # type: ignore[no-untyped-def]
    return Field(data=(field_in.data >= float(cutoff)).astype(np.float32))


@node(
    type_id="output.note",
    title="Note",
    category="Output",
    color="grey",
    description="Annotation, no compute.",
    inputs=[],
    outputs=[],
)
def note(text: str = Param("", kind="textarea", label="Text")) -> None:  # type: ignore[no-untyped-def]
    return None
