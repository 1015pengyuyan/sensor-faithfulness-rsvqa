from __future__ import annotations

import json
from collections import Counter
from typing import Any


def _source_id(row: dict[str, Any]) -> str:
    for key in ("source_id", "target_id", "example_id", "target_source_id"):
        if row.get(key) is not None:
            return str(row[key])
    target = row.get("target_source")
    if isinstance(target, dict):
        for key in ("source_id", "patch_id", "s1_name"):
            if target.get(key) is not None:
                return str(target[key])
    raise KeyError(f"cannot infer source id for {row.get('fixture_id')}")


def _target_and_distractor_ids(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    targets: set[str] = set()
    distractors: set[str] = set()
    for ref in row.get("image_refs", []):
        if not isinstance(ref, dict) or ref.get("source_id") is None:
            continue
        role = str(ref.get("role", "")).lower()
        if role.startswith("target") or role == "requested":
            targets.add(str(ref["source_id"]))
        else:
            distractors.add(str(ref["source_id"]))
    return targets, distractors


def _label_valid(row: dict[str, Any]) -> bool:
    target = str(row.get("target_label", "")).lower()
    distractor = str(row.get("distractor_label", "")).lower()
    if target not in {"yes", "no"} or distractor not in {"yes", "no"}:
        return False
    condition = str(row.get("condition_id", "")).upper()
    relation = str(row.get("condition_relation", row.get("condition_family", ""))).lower()
    congruent = "A0" in condition or relation == "congruent"
    conflict = "A2" in condition or "A3" in condition or "CONFLICT" in condition or relation == "opposite"
    if congruent:
        return target == distractor
    if conflict:
        return target != distractor
    return False


def validate_fixture(rows: list[dict[str, Any]]) -> dict[str, int]:
    ids = [str(row.get("fixture_id", "")) for row in rows]
    duplicate_ids = sum(count - 1 for count in Counter(ids).values() if count > 1)
    overlap = 0
    label_violations = 0
    missing_requested = 0
    test_refs = 0
    for row in rows:
        targets, distractors = _target_and_distractor_ids(row)
        overlap += int(bool(targets & distractors))
        label_violations += int(not _label_valid(row))
        requested = str(row.get("requested_sensor", "")).lower()
        requested_refs = [
            ref for ref in row.get("image_refs", [])
            if str(ref.get("sensor", "")).lower() == requested
            and str(ref.get("role", "")).lower().startswith("target")
        ]
        missing_requested += int(len(requested_refs) != 1)
        blob = json.dumps(row, ensure_ascii=False).replace("\\", "/").lower()
        test_refs += int("/test/" in blob or blob.startswith("test/") or '"test/' in blob)
    return {
        "rows": len(rows),
        "source_clusters": len({_source_id(row) for row in rows}),
        "duplicate_fixture_ids": duplicate_ids,
        "target_distractor_overlap": overlap,
        "label_violations": label_violations,
        "missing_requested_evidence": missing_requested,
        "prohibited_test_references": test_refs,
    }
