from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from sensor_faithfulness.data.prompt_protocol import ordered_image_refs
from sensor_faithfulness.utils.io import sha256_file


def assert_public_relative_path(value: str) -> None:
    normalized = value.replace("\\", "/").lower()
    if Path(value).is_absolute() or "/test/" in normalized or normalized.startswith("test/"):
        raise ValueError(f"prohibited image reference: {value}")


def verify_and_load_images(
    row: dict[str, Any],
    data_root: Path,
) -> tuple[list[Image.Image], list[Path]]:
    images: list[Image.Image] = []
    paths: list[Path] = []
    for ref in ordered_image_refs(row["image_refs"]):
        relative = str(ref["relative_path"])
        assert_public_relative_path(relative)
        path = data_root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        if ref.get("sha256") and sha256_file(path) != ref["sha256"]:
            raise RuntimeError(f"image hash mismatch: {relative}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
        paths.append(path)
    if len(images) != 2:
        raise RuntimeError("the frozen protocol requires exactly two images")
    return images, paths


def chat_messages(row: dict[str, Any], image_count: int = 2) -> list[dict[str, Any]]:
    content = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": row["user_prompt"]})
    return [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": content},
    ]
