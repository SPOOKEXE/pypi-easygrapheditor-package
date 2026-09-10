"""Comfy-parity mock nodes (stubbed for M5)."""

from __future__ import annotations

from ..engine.nodes import node


@node(category="Comfy", color="grey", description="Load Image (mock).", badges=["Comfy"])
def load_image() -> float:
    raise NotImplementedError("Comfy demo nodes stubbed for M5.")


@node(category="Comfy", color="grey", description="Preview Image (mock).", badges=["Comfy"])
def preview_image(image: float) -> float:  # type: ignore[no-untyped-def]
    raise NotImplementedError("Comfy demo nodes stubbed for M5.")


@node(category="Comfy", color="grey", description="Resolution Selector (mock).")
def resolution_selector() -> float:
    raise NotImplementedError("Comfy demo nodes stubbed for M5.")
