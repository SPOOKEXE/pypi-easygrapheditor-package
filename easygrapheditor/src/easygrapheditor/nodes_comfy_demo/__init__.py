"""Small deterministic stand-ins for the ComfyUI nodes shown by the editor.

They intentionally model port shapes, not diffusion.  This lets imported
workflows validate, execute and round-trip without downloading model weights.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..engine.nodes import Param, PortDef, node
from ..engine.types import Image

COMFY = ["Comfy"]


def _image_for(value: str, width: int = 64, height: int = 64) -> Image:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    colour = np.array([digest[0], digest[1], digest[2]], dtype=np.uint8)
    return Image(np.broadcast_to(colour, (max(1, height), max(1, width), 3)).copy())


@node(
    type_id="comfy.load_image",
    title="Load Image",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Deterministic image generated from the selected path.",
    outputs=[PortDef("image", "IMAGE", "IMAGE", "out")],
)
def load_image(image: str = Param("", kind="file", label="Image")) -> Image:  # type: ignore[no-untyped-def]
    return _image_for(str(image))


@node(
    type_id="comfy.preview_image",
    title="Preview Image",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Passes an image through for previewing.",
    inputs=[PortDef("images", "Images", "IMAGE", "in")],
    outputs=[PortDef("image", "IMAGE", "IMAGE", "out")],
)
def preview_image(images: Any) -> Any:
    return images


@node(
    type_id="comfy.save_image_advanced",
    title="Save Image (Advanced)",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Mock save that returns the image without writing a file.",
    inputs=[PortDef("images", "Images", "IMAGE", "in")],
    outputs=[PortDef("image", "IMAGE", "IMAGE", "out")],
    cacheable=False,
)
def save_image_advanced(
    images: Any, filename_prefix: str = Param("EasyGraph", kind="text", label="Filename prefix")
) -> Any:  # type: ignore[no-untyped-def]
    _ = filename_prefix
    return images


@node(
    type_id="comfy.load_checkpoint",
    title="Load Checkpoint",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Produces deterministic mock MODEL, CLIP and VAE handles.",
    outputs=[
        PortDef("model", "MODEL", "MODEL", "out"),
        PortDef("clip", "CLIP", "CLIP", "out"),
        PortDef("vae", "VAE", "VAE", "out"),
    ],
)
def load_checkpoint(
    ckpt_name: str = Param(
        "mock.safetensors", kind="select", label="Checkpoint", options=["mock.safetensors"]
    ),
) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {"model": f"model:{ckpt_name}", "clip": f"clip:{ckpt_name}", "vae": f"vae:{ckpt_name}"}


@node(
    type_id="comfy.load_lora",
    title="Load LoRA",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Tags MODEL and CLIP handles with a mock LoRA.",
    inputs=[PortDef("model", "MODEL", "MODEL", "in"), PortDef("clip", "CLIP", "CLIP", "in")],
    outputs=[PortDef("model", "MODEL", "MODEL", "out"), PortDef("clip", "CLIP", "CLIP", "out")],
)
def load_lora(
    model: Any,
    clip: Any,
    lora_name: str = Param(
        "mock.safetensors", kind="select", label="LoRA", options=["mock.safetensors"]
    ),
) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {"model": f"{model}|lora:{lora_name}", "clip": f"{clip}|lora:{lora_name}"}


@node(
    type_id="comfy.clip_text_encode",
    title="CLIP Text Encode",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Creates a deterministic CONDITIONING payload from text.",
    inputs=[PortDef("clip", "CLIP", "CLIP", "in")],
    outputs=[PortDef("conditioning", "CONDITIONING", "CONDITIONING", "out")],
)
def clip_text_encode(
    clip: Any, text: str = Param("", kind="textarea", label="Text")
) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {"clip": str(clip), "text": str(text)}


@node(
    type_id="comfy.empty_latent_image",
    title="Empty Latent Image",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Creates a tiny deterministic latent descriptor.",
    outputs=[PortDef("latent", "LATENT", "LATENT", "out")],
)
def empty_latent_image(
    width: int = Param(512, kind="int", label="Width"),
    height: int = Param(512, kind="int", label="Height"),
    batch_size: int = Param(1, kind="int", label="Batch size"),
) -> dict[str, int]:  # type: ignore[no-untyped-def]
    return {"width": int(width), "height": int(height), "batch_size": int(batch_size)}


@node(
    type_id="comfy.vae_encode",
    title="VAE Encode",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Encodes IMAGE metadata into a mock LATENT.",
    inputs=[PortDef("pixels", "IMAGE", "IMAGE", "in"), PortDef("vae", "VAE", "VAE", "in")],
    outputs=[PortDef("latent", "LATENT", "LATENT", "out")],
)
def vae_encode(pixels: Any, vae: Any) -> dict[str, Any]:
    return {"pixels": pixels, "vae": str(vae)}


@node(
    type_id="comfy.vae_decode",
    title="VAE Decode",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Decodes a mock latent into a deterministic thumbnail.",
    inputs=[PortDef("samples", "LATENT", "LATENT", "in"), PortDef("vae", "VAE", "VAE", "in")],
    outputs=[PortDef("image", "IMAGE", "IMAGE", "out")],
)
def vae_decode(samples: Any, vae: Any) -> Image:
    return _image_for(f"{samples}|{vae}")


@node(
    type_id="comfy.ksampler",
    title="KSampler",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Adds sampler settings to a mock latent.",
    inputs=[
        PortDef("model", "MODEL", "MODEL", "in"),
        PortDef("positive", "Positive", "CONDITIONING", "in"),
        PortDef("negative", "Negative", "CONDITIONING", "in"),
        PortDef("latent_image", "Latent", "LATENT", "in"),
    ],
    outputs=[PortDef("latent", "LATENT", "LATENT", "out")],
)
def ksampler(
    model: Any,
    positive: Any,
    negative: Any,
    latent_image: Any,
    seed: int = Param(0, kind="seed", label="Seed"),
    steps: int = Param(20, kind="int", label="Steps"),
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return {
        "latent": latent_image,
        "model": str(model),
        "positive": positive,
        "negative": negative,
        "seed": int(seed),
        "steps": int(steps),
    }


@node(
    type_id="comfy.video_combine",
    title="Video Combine",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Wraps images in a deterministic VIDEO descriptor.",
    inputs=[PortDef("images", "Images", "IMAGE", "in")],
    outputs=[PortDef("video", "VIDEO", "VIDEO", "out")],
)
def video_combine(
    images: Any, frame_rate: float = Param(8.0, kind="number", label="Frame rate")
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return {"frames": images, "fps": float(frame_rate)}


@node(
    type_id="comfy.image_to_video",
    title="Image to Video",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Turns an image into a one-frame VIDEO descriptor.",
    inputs=[PortDef("image", "IMAGE", "IMAGE", "in")],
    outputs=[PortDef("video", "VIDEO", "VIDEO", "out")],
)
def image_to_video(image: Any) -> dict[str, Any]:
    return {"frames": image, "fps": 1.0}


@node(
    type_id="comfy.save_video",
    title="Save Video",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Mock save that returns VIDEO without external I/O.",
    inputs=[PortDef("video", "VIDEO", "VIDEO", "in")],
    outputs=[PortDef("video", "VIDEO", "VIDEO", "out")],
    cacheable=False,
)
def save_video(
    video: Any, filename_prefix: str = Param("EasyGraph", kind="text", label="Filename prefix")
) -> Any:  # type: ignore[no-untyped-def]
    _ = filename_prefix
    return video


@node(
    type_id="comfy.resolution_selector",
    title="Resolution Selector",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Outputs a selected width and height.",
    outputs=[
        PortDef("width", "Width", "data.NUMBER", "out"),
        PortDef("height", "Height", "data.NUMBER", "out"),
    ],
)
def resolution_selector(
    width: int = Param(512, kind="int", label="Width"),
    height: int = Param(512, kind="int", label="Height"),
) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    return float(width), float(height)


@node(
    type_id="comfy.get_image_size",
    title="Get Image Size",
    category="Comfy",
    color="grey",
    badges=COMFY,
    description="Reads IMAGE dimensions.",
    inputs=[PortDef("image", "IMAGE", "IMAGE", "in")],
    outputs=[
        PortDef("width", "Width", "data.NUMBER", "out"),
        PortDef("height", "Height", "data.NUMBER", "out"),
    ],
)
def get_image_size(image: Any) -> tuple[float, float]:
    data = getattr(image, "data", image)
    shape = getattr(data, "shape", (64, 64))
    return float(shape[1]), float(shape[0])
