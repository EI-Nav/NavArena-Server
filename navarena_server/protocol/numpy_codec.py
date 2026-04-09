"""Numpy codec for msgpack serialization."""

from __future__ import annotations

import contextvars
from typing import Any

import numpy as np
from PIL import Image as PILImage

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

from navarena_server.protocol.image_codec import (
    ImageFormat,
    _is_image_array,
    decode_image,
    encode_image,
    is_encoded_image,
)

_ALLOWED_DTYPE_KINDS = frozenset("biuf")
_NDARRAY_KEY = "__ndarray__"

_image_format_var: contextvars.ContextVar[ImageFormat] = contextvars.ContextVar(
    "_image_format", default="png"
)


def set_image_format(fmt: ImageFormat) -> None:
    _image_format_var.set(fmt)


def get_image_format() -> ImageFormat:
    return _image_format_var.get()


def encode_ndarray(obj: Any) -> Any:
    """msgpack default hook: convert numpy arrays to serializable dicts."""
    if isinstance(obj, PILImage.Image):
        obj = np.asarray(obj, dtype=np.uint8)
    if _has_torch and isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        fmt = _image_format_var.get()
        if fmt != "raw" and _is_image_array(obj):
            return encode_image(obj, fmt)
        return {
            _NDARRAY_KEY: True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": list(obj.shape),
        }
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def decode_ndarray(obj: Any) -> Any:
    """msgpack object_hook: reconstruct numpy arrays from dicts."""
    if not isinstance(obj, dict):
        return obj
    if is_encoded_image(obj):
        return decode_image(obj)
    if _NDARRAY_KEY not in obj:
        return obj
    dtype = np.dtype(obj["dtype"])
    if dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ValueError(
            f"Disallowed numpy dtype: {dtype}. Only bool/int/uint/float are permitted."
        )
    return np.frombuffer(obj["data"], dtype=dtype).reshape(obj["shape"]).copy()
