#!/usr/bin/env python3
"""Build the named-branch M4 view from the frozen retrospective fixture.

No rows are sampled or relabeled. The builder selects only the already
predeclared requested-first dual rows, then removes generic slot semantics for
EarthMind's fixed SAR/RGB branch. It does not read images or model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CONDITIONS = (
    "A0_CONGRUENT",
    "A0_SECOND_CONGRUENT",
    "A2_CONFLICT",
    "A3_CONFLICT_REPLICATE_1",
    "A3_CONFLICT_REPLICATE_2",
)
REPLAY_SEED = "M4_UNIFIED_128_EARTHMIND_REPLAY_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def _extract_question(prompt: str, fixture_id: str) -> str:
    match = re.search(r"(?:^|\n)Question: (.*?)(?:\nCandidate answers:|\Z)", prompt, re.S)
    if not match:
        raise ValueError(f"question text cannot be recovered from frozen prompt: {fixture_id}")
    return match.group(1).strip()


def _branch(ref: dict[str, Any], sensor_name: str) -> dict[str, Any]:
    if not ref.get("relative_path") or len(str(ref.get("sha256", ""))) != 64:
        raise ValueError("M4 image reference lacks relative path or SHA-256")
    return {
        "relative_path": str(ref["relative_path"]),
        "sha256": str(ref["sha256"]).lower(),
        "source_id": str(ref["source_id"]),
        "sensor": sensor_name,
    }


def _to_named_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("row_family") != "dual" or str(row.get("requested_image_position")) != "1":
        raise ValueError(f"EarthMind projection accepts only frozen primary dual rows: {row.get('fixture_id')}")
    if row.get("condition_id") not in CONDITIONS:
        raise ValueError(f"unexpected M4 condition: {row.get('condition_id')}")
    refs = row.get("image_refs") or []
    if len(refs) != 2:
        raise ValueError(f"M4 dual row must have two source refs: {row.get('fixture_id')}")
    refs_by_sensor = {str(ref["sensor"]).lower(): ref for ref in refs}
    if set(refs_by_sensor) != {"optical", "sar"}:
        raise ValueError(f"M4 row must have one optical and one SAR image: {row.get('fixture_id')}")
    requested = str(row["requested_sensor"]).lower()
    requested_ref = next(ref for ref in refs if ref.get("role") == "target_requested")
    distractor_ref = next(ref for ref in refs if ref.get("role") == "distractor_non_requested")
    if str(requested_ref["sensor"]).lower() != requested:
        raise ValueError(f"requested target source/sensor mismatch: {row.get('fixture_id')}")
    expected_target = str(row["target_source"]["source_id"])
    if str(requested_ref["source_id"]) != expected_target:
        raise ValueError(f"requested image source changed: {row.get('fixture_id')}")
    expected_distractor = str(row["distractor_source"]["source_id"])
    if str(distractor_ref["source_id"]) != expected_distractor:
        raise ValueError(f"distractor source changed: {row.get('fixture_id')}")

    # Canonical metadata order is only for deterministic serialization. The
    # native model call is keyed by rgb_image/sar_image, never by list position.
    branches = {
        "rgb_image": _branch(refs_by_sensor["optical"], "optical"),
        "sar_image": _branch(refs_by_sensor["sar"], "SAR"),
    }
    question = _extract_question(str(row["user_prompt"]), str(row["fixture_id"]))
    requested_name = "optical/RGB" if requested == "optical" else "SAR"
    native_prompt = (
        f"Answer the binary presence question using the requested {requested_name} evidence. "
        "The other named sensor is non-requested context and must not override it. "
        f"Question: {question} Reply with exactly one of: yes, no, abstain."
    )
    out = dict(row)
    out.update({
        "row_family": "dual_named",
        "input_mode": "earthmind_named_dual_sensor_branches",
        "prompt_template_id": "earthmind_native_branch_control_v1",
        "image_paths": branches,
        "native_sensor_branches": branches,
        "image_1": None,
        "image_2": None,
        "image_order": "native_named_rgb_sar_branches",
        "requested_image_position": "NOT_APPLICABLE_NATIVE_NAMED_BRANCH",
        "requested_position": "named_sar_rgb",
        "position_role": "not_applicable",
        "prompt": native_prompt,
        "user_prompt": native_prompt,
        "question_text": question,
        "intended_use": "unified_128_eight_model_comparison",
        "matrix_version": "unified_128_eight_model_20261018_v1",
        "is_replay": False,
    })
    # Keep image_refs solely for audited file verification/adapter loading;
    # remove generic slot labels and use canonical sensor order.
    canonical_refs = []
    for sensor in ("optical", "sar"):
        ref = dict(refs_by_sensor[sensor])
        ref.pop("slot", None)
        canonical_refs.append(ref)
    out["image_refs"] = canonical_refs
    return out


def _replay_manifest(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["requested_sensor"]), str(row["condition_id"])), []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda row: (
            hashlib.sha256(f"{REPLAY_SEED}\0{row['fixture_id']}".encode()).hexdigest(),
            str(row["fixture_id"]),
        ))
        selected.extend(ranked[:2])
    remainder = [row for row in rows if row["fixture_id"] not in {x["fixture_id"] for x in selected}]
    remainder.sort(key=lambda row: (
        hashlib.sha256(f"{REPLAY_SEED}\0fill\0{row['fixture_id']}".encode()).hexdigest(),
        str(row["fixture_id"]),
    ))
    selected.extend(remainder[:26 - len(selected)])
    if len(selected) != 26 or len({row["fixture_id"] for row in selected}) != 26:
        raise ValueError("EarthMind M4 replay manifest must contain 26 unique fixture IDs")
    return [{"fixture_id": str(row["fixture_id"]), "call_role": "replay"} for row in selected]


def build_m4_earthmind_fixture(*, source_fixture: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = read_jsonl(source_fixture)
    rows = [_to_named_row(row) for row in source_rows
            if row.get("row_family") == "dual" and str(row.get("requested_image_position")) == "1"]
    if len(rows) != 4200:
        raise ValueError(f"expected 4,200 M4 primary-position EarthMind rows; got {len(rows)}")
    ids = [str(row["fixture_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate EarthMind fixture IDs")
    target_ids = {str(row["target_source_cluster_id"]) for row in rows}
    if len(target_ids) != 420:
        raise ValueError(f"expected 420 target source clusters, got {len(target_ids)}")
    counts = Counter((str(row["requested_sensor"]), str(row["condition_id"])) for row in rows)
    if any(counts[(sensor, condition)] != 420 for sensor in ("optical", "sar") for condition in CONDITIONS):
        raise ValueError("M4 EarthMind cell counts do not match 420 × 2 directions × 5 conditions")
    replays = _replay_manifest(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "M4_EARTHMIND_NAMED_128_FIXTURE.jsonl"
    replay_path = output_dir / "M4_EARTHMIND_NAMED_REPLAY_MANIFEST.jsonl"
    write_jsonl(fixture_path, rows)
    write_jsonl(replay_path, replays)
    manifest = {
        "matrix_version": "unified_128_eight_model_20261018_v1",
        "dataset": "M4-SAR",
        "model_interface": "EarthMind native named SAR/RGB dual branch; no generic position factor",
        "source_fixture_sha256": sha256_file(source_fixture),
        "source_parquet_or_images_opened_by_builder": False,
        "model_outputs_used": False,
        "official_test_accessed": False,
        "sealed_final_accessed": False,
        "selection": "retain only frozen generic fixture rows with row_family=dual and requested_image_position=1; no source/distractor/label selection",
        "target_sources": len(target_ids),
        "unique_rows": len(rows),
        "replay_calls": len(replays),
        "total_calls": len(rows) + len(replays),
        "counts_by_direction_condition": {f"{sensor}|{condition}": counts[(sensor, condition)]
                                           for sensor in ("optical", "sar") for condition in CONDITIONS},
        "fixture": {"path": fixture_path.name, "sha256": sha256_file(fixture_path), "rows": len(rows)},
        "replay_manifest": {"path": replay_path.name, "sha256": sha256_file(replay_path), "rows": len(replays)},
        "replay_seed": REPLAY_SEED,
        "parent_scene_independence": "unverified; historical exposed source allocation",
    }
    manifest_path = output_dir / "M4_EARTHMIND_NAMED_FIXTURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_m4_earthmind_fixture(source_fixture=args.source_fixture,
                                               output_dir=args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
