"""Frozen BigEarthNet Sentinel-1/Sentinel-2 rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def scale_uint8(array: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(np.asarray(array, dtype=np.float32), low, high)
    normalized = (clipped - low) / (high - low)
    return np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8)


def _resize(rgb: np.ndarray, size: int | None, resample: int) -> np.ndarray:
    if size is None:
        return rgb
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize((size, size), resample=resample))


def render_s2_arrays(
    b02: np.ndarray,
    b03: np.ndarray,
    b04: np.ndarray,
    size: int | None = None,
) -> np.ndarray:
    rgb = np.stack(
        [scale_uint8(b04, 0, 3000), scale_uint8(b03, 0, 3000), scale_uint8(b02, 0, 3000)],
        axis=-1,
    )
    return _resize(rgb, size, Image.Resampling.BICUBIC)


def render_s1_arrays(vv: np.ndarray, vh: np.ndarray, size: int | None = None) -> np.ndarray:
    vv_u8 = scale_uint8(vv, -25, 0)
    vh_u8 = scale_uint8(vh, -32, -5)
    average = np.rint((vv_u8.astype(np.float32) + vh_u8.astype(np.float32)) / 2).astype(np.uint8)
    rgb = np.stack([vv_u8, vh_u8, average], axis=-1)
    return _resize(rgb, size, Image.Resampling.BILINEAR)


def load_single_band(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    array = np.squeeze(array).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"unsupported band shape {array.shape}: {path}")
    return array


def render_s2_files(b02: Path, b03: Path, b04: Path, output: Path, size: int = 512) -> None:
    rgb = render_s2_arrays(load_single_band(b02), load_single_band(b03), load_single_band(b04), size)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output)


def render_s1_files(vv: Path, vh: Path, output: Path, size: int = 512) -> None:
    rgb = render_s1_arrays(load_single_band(vv), load_single_band(vh), size)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output)
