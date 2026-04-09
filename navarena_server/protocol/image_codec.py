"""Image codec for HWC uint8 arrays — JPEG, PNG, or raw."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Literal

import numpy as np

ImageFormat = Literal["raw", "jpeg", "png"]

_IMAGE_KEY = "__image__"


def _is_image_array(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[2] in (1, 3, 4) and arr.dtype == np.uint8


def is_encoded_image(obj: Any) -> bool:
    return isinstance(obj, dict) and _IMAGE_KEY in obj


def encode_image(arr: np.ndarray, fmt: ImageFormat = "png") -> dict:
    shape = list(arr.shape)
    if fmt == "raw":
        return {
            _IMAGE_KEY: True,
            "data": arr.tobytes(),
            "format": "raw",
            "shape": shape,
            "dtype": arr.dtype.str,
        }

    from PIL import Image

    if fmt == "jpeg" and arr.shape[2] != 3:
        raise ValueError(
            f"JPEG requires RGB (3-channel) images, got {arr.shape[2]} channels"
        )

    img = Image.fromarray(arr)
    buf = BytesIO()
    pil_format = "JPEG" if fmt == "jpeg" else "PNG"
    img.save(buf, format=pil_format)
    return {_IMAGE_KEY: True, "data": buf.getvalue(), "format": fmt, "shape": shape}


def decode_image(obj: dict) -> np.ndarray:
    for key in ("format", "shape", "data"):
        if key not in obj:
            raise ValueError(f"Encoded image missing required field: {key!r}")

    fmt = obj["format"]
    shape = obj["shape"]
    data = obj["data"]

    if not isinstance(shape, (list, tuple)) or not all(isinstance(s, int) for s in shape):
        raise ValueError(f"Invalid image shape: {shape!r}")

    if fmt == "raw":
        dtype = np.dtype(obj.get("dtype", "uint8"))
        expected_size = int(np.prod(shape)) * dtype.itemsize
        if len(data) != expected_size:
            raise ValueError(
                f"Raw image buffer size mismatch: got {len(data)} bytes, "
                f"expected {expected_size} for shape {shape} dtype {dtype}"
            )
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    if fmt not in ("jpeg", "png"):
        raise ValueError(f"Unknown image format: {fmt!r}")

    from PIL import Image

    img = Image.open(BytesIO(data))
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr
