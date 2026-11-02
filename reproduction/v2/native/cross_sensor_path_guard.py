"""Path safety guards for cross-sensor remote-sensing data.

The guard is deliberately tiny and importable by every data loader.  Any path
whose normalized components include a prohibited official-evaluation split must
raise before the caller can open the file.
"""
from __future__ import annotations

from pathlib import PurePosixPath, Path


PROHIBITED_SPLIT_COMPONENTS = {"test"}


def normalized_components(path: str | Path) -> tuple[str, ...]:
    text = str(path).replace("\\", "/")
    return tuple(part for part in PurePosixPath(text).parts if part not in {"", "."})


def assert_no_prohibited_split(path: str | Path) -> None:
    parts = {part.lower() for part in normalized_components(path)}
    overlap = parts.intersection(PROHIBITED_SPLIT_COMPONENTS)
    if overlap:
        raise ValueError("Prohibited official-evaluation split component in data path")


def assert_sanitized_data_path(path: str | Path) -> None:
    assert_no_prohibited_split(path)
