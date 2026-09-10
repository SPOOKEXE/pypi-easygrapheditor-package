"""Param widgets: backend-neutral specs + per-backend component factories.

Canonical widget names (see ``canonical_kind`` in engine.nodes):
  slider    continuous/decimal slider  (float_slider)
  stepslider stepped slider            (step_slider)
  numeric    decimal input box          (number)
  numeric_int integer input box        (int)
  seed      numerical input + randomize affordance
  input     single-line input box      (text)
  textarea  multi-line input box
  options   dropdown                   (select)
  boolean   switch/checkbox            (checkbox)
  file      file picker (stores path string)
"""

from __future__ import annotations

from typing import Any

from ..engine.nodes import ParamDef, canonical_kind


def widget_spec(param: ParamDef) -> dict[str, Any]:
    """Describe how to render ``param`` without importing any UI toolkit."""
    kind = canonical_kind(param.kind)
    spec: dict[str, Any] = {"key": param.key, "label": param.label or param.key, "kind": kind}
    if kind in ("float_slider", "slider"):
        lo = 0.0 if param.min is None else float(param.min)
        hi = 1.0 if param.max is None else float(param.max)
        step = float(param.step) if param.step else (hi - lo) / 100.0 or 0.01
        spec.update({"widget": "slider", "min": lo, "max": hi, "step": step, "decimal": True})
    elif kind == "step_slider":
        lo = 0 if param.min is None else param.min
        hi = 100 if param.max is None else param.max
        spec.update({"widget": "stepslider", "min": lo, "max": hi, "step": param.step or 1})
    elif kind == "int":
        spec.update({"widget": "numeric_int"})
    elif kind in ("number", "seed"):
        spec.update({"widget": "seed" if kind == "seed" else "numeric"})
    elif kind == "text":
        spec.update({"widget": "input"})
    elif kind in ("textarea", "multiline"):
        spec.update({"widget": "textarea", "lines": 4})
    elif kind in ("select", "dropdown"):
        spec.update({"widget": "options", "options": list(param.options or [])})
    elif kind in ("checkbox", "toggle"):
        spec.update({"widget": "boolean"})
    elif kind == "file":
        spec.update({"widget": "file"})
    else:
        spec.update({"widget": "input"})
    return spec


def cast_param_value(param: ParamDef, raw: Any) -> Any:
    """Coerce a UI-entered value back to the param's declared type."""
    default = param.default
    kind = canonical_kind(param.kind)
    try:
        if kind in ("checkbox", "toggle"):
            if isinstance(raw, str):
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return bool(raw)
        if isinstance(default, bool):
            return bool(raw)
        if isinstance(default, int) and not isinstance(default, bool):
            return int(float(raw))
        if isinstance(default, float):
            return float(raw)
        if kind in ("int", "step_slider") and isinstance(raw, float) and float(raw).is_integer():
            return int(raw)
        return raw
    except (TypeError, ValueError):
        return default


def adjust_param_value(param: ParamDef, current: Any, direction: int, factor: float = 1.0) -> Any:
    """Keyboard nudge for pygame manipulation: sliders/numbers step, booleans flip, options cycle."""
    kind = canonical_kind(param.kind)
    if kind in ("checkbox", "toggle"):
        return not bool(current)
    if kind in ("select", "dropdown") and param.options:
        opts = list(param.options)
        try:
            i = opts.index(current)
        except ValueError:
            i = 0 if direction > 0 else len(opts) - 1
            return opts[i]
        return opts[(i + direction) % len(opts)]
    if kind in ("float_slider", "slider", "step_slider", "int", "number", "seed"):
        spec = widget_spec(param)
        step = float(spec.get("step") or 1) * factor
        try:
            val = float(current) + direction * step
        except (TypeError, ValueError):
            return current
        if spec.get("min") is not None:
            val = max(float(spec["min"]), val)
        if spec.get("max") is not None:
            val = min(float(spec["max"]), val)
        if kind == "int" or (kind == "step_slider" and isinstance(param.default, int)):
            return round(val)
        return val
    return current


def gradio_param_component(gr: Any, param: ParamDef, value: Any) -> Any:
    """Build a Gradio input component for ``param`` (host imports gradio)."""
    spec = widget_spec(param)
    label = spec["label"]
    widget = spec["widget"]
    if widget == "slider":
        return gr.Slider(minimum=spec["min"], maximum=spec["max"], step=spec["step"], value=value, label=label)
    if widget == "stepslider":
        return gr.Slider(minimum=spec["min"], maximum=spec["max"], step=spec["step"], value=value, label=label)
    if widget == "numeric_int":
        return gr.Number(value=value, label=label, precision=0)
    if widget in ("numeric", "seed"):
        return gr.Number(value=value, label=label)
    if widget == "textarea":
        return gr.Textbox(value=value, label=label, lines=spec["lines"])
    if widget == "options":
        choices = spec["options"] or ([value] if value is not None else [])
        return gr.Dropdown(choices=choices, value=value if value in choices else (choices[0] if choices else None), label=label)
    if widget == "boolean":
        return gr.Checkbox(value=bool(value), label=label)
    if widget == "file":
        return gr.File(value=value or None, label=label)
    return gr.Textbox(value="" if value is None else str(value), label=label)


def streamlit_param_widget(container: Any, param: ParamDef, value: Any, key: str) -> Any:
    """Render a Streamlit widget for ``param``; returns the (possibly new) value."""
    spec = widget_spec(param)
    label = spec["label"]
    widget = spec["widget"]
    if widget == "slider":
        return container.slider(label, min_value=float(spec["min"]), max_value=float(spec["max"]),
                                value=float(value or 0.0), step=float(spec["step"]), key=key)
    if widget == "stepslider":
        if isinstance(param.default, int):
            return container.slider(label, min_value=int(spec["min"]), max_value=int(spec["max"]),
                                    value=int(value or 0), step=int(spec["step"] or 1), key=key)
        return container.slider(label, min_value=float(spec["min"]), max_value=float(spec["max"]),
                                value=float(value or 0.0), step=float(spec["step"] or 1), key=key)
    if widget == "numeric_int":
        return container.number_input(label, value=int(value or 0), step=1, key=key)
    if widget in ("numeric", "seed"):
        return container.number_input(label, value=float(value or 0.0), key=key)
    if widget == "textarea":
        return container.text_area(label, value="" if value is None else str(value), key=key)
    if widget == "options":
        choices = spec["options"] or ([value] if value is not None else [])
        return container.selectbox(label, choices, index=choices.index(value) if value in choices else 0, key=key)
    if widget == "boolean":
        return container.checkbox(label, value=bool(value), key=key)
    if widget == "file":
        uploaded = container.file_uploader(label, key=key)
        return uploaded.name if uploaded is not None else value
    return container.text_input(label, value="" if value is None else str(value), key=key)


__all__ = [
    "adjust_param_value",
    "cast_param_value",
    "gradio_param_component",
    "streamlit_param_widget",
    "widget_spec",
]
