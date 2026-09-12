"""Core data types for EasyGraphEditor.

See editor.md §4.1 and plans/ screenshots 15-42-41 (Types tab).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any as AnyHint
from typing import Literal

import numpy as np

PreviewKind = Literal["none", "spark", "thumbnail"]


@dataclass(frozen=True)
class DataType:
    id: str  # e.g. "data.NUMBER"
    label: str  # e.g. "Number"
    description: str
    preview: PreviewKind = "none"


# Payloads (v1, see editor.md §4.5).
Number = float
# Public annotation hint for wildcard ports. ``ANY`` remains the runtime
# wildcard value used by the type registry.
Any = AnyHint


@dataclass
class Field:
    """Square grid of heights 0..1."""

    data: np.ndarray  # float32 HxW


@dataclass
class Image:
    """Square grid of colours."""

    data: np.ndarray  # uint8 HxWx3


ANY = object()

_REGISTRY: dict[str, DataType] = {}


def register_type(dtype: DataType) -> DataType:
    _REGISTRY[dtype.id] = dtype
    return dtype


def get_type(type_id: str) -> DataType | None:
    return _REGISTRY.get(type_id)


def list_types() -> list[DataType]:
    return list(_REGISTRY.values())


def can_connect(out_id: str, in_id: str) -> bool:
    """A link is legal iff ids match or either end is the wildcard."""
    return out_id == in_id or out_id == "data.ANY" or in_id == "data.ANY"


# Built-in types.
register_type(DataType("data.NUMBER", "Number", "One scalar.", "none"))
register_type(DataType("data.FIELD", "Field", "A square grid of heights, 0..1.", "thumbnail"))
register_type(DataType("data.IMAGE", "Image", "A square grid of colours.", "thumbnail"))
register_type(DataType("data.ANY", "Any", "Wildcard: connects to anything.", "none"))

# Comfy-style extension ids (string-only for v1, no payload validation yet).
for _tid, _label, _desc in [
    ("IMAGE", "Image (Comfy)", "Comfy image tensor."),
    ("MASK", "Mask", "Comfy mask."),
    ("VIDEO", "Video", "Comfy video."),
    ("MODEL", "Model", "Diffusion model."),
    ("CLIP", "CLIP", "Comfy CLIP encoder."),
    ("VAE", "VAE", "Comfy VAE."),
    ("CONDITIONING", "Conditioning", "CLIP conditioning."),
    ("LATENT", "Latent", "Latent tensor."),
]:
    register_type(DataType(_tid, _label, _desc, "thumbnail"))
