#!/usr/bin/env python3
"""Build the versioned BigEarthNet 128-token control fixtures from frozen inputs.

This builder reads only the already frozen BigEarthNet benchmark fixture, its
20-question smoke exclusion list, and the approved 432-pair render manifest.
It does not open the annotation parquet, any image, a test split, or model output.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_CONDITIONS = (
    "A0_CONGRUENT",
    "A2_OPPOSITE_1",
    "A3_1_OPPOSITE_2",
    "A3_2_OPPOSITE_3",
)
CONTROL_CONDITION = "A0_SECOND_CONGRUENT"
AUXILIARY_CONDITIONS = ("REQUESTED_ONLY", "NONREQUESTED_ONLY", "TEXT_ONLY")
EXPECTED_SENSITIVITY_EXCLUSIONS = {
    "BENEXT_0060", "BENEXT_0063", "BENEXT_0082", "BENEXT_0115",
    "BENEXT_0121", "BENEXT_0263", "BENEXT_0280", "BENEXT_0325",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_render_assets(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    assets: dict[str, dict[str, str]] = {"s1_sar": {}, "s2_optical": {}}
    for row in rows:
        modality = row.get("modality")
        key = row.get("s1_name") if modality == "s1_sar" else row.get("patch_id")
        if modality not in assets or not key:
            raise ValueError(f"invalid render-manifest row: {row}")
        digest = str(row.get("render_sha256", "")).lower()
        if len(digest) != 64:
            raise ValueError(f"invalid render SHA-256 for {key}")
        prior = assets[modality].get(key)
        if prior is not None and prior != digest:
            raise ValueError(f"conflicting render hash for {modality}:{key}")
        assets[modality][key] = digest
    return assets


def _target_source(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("target_source")
    if not isinstance(source, dict):
        raise ValueError(f"target_source missing for {row.get('fixture_id')}")
    patch_id = str(row.get("target_patch_id") or source.get("patch_id") or "")
    s1_name = str(row.get("target_s1_name") or source.get("s1_name") or "")
    if not patch_id or not s1_name:
        raise ValueError(f"target source key missing for {row.get('fixture_id')}")
    if source.get("patch_id") != patch_id or source.get("s1_name") != s1_name:
        raise ValueError(f"target source fields disagree for {row.get('fixture_id')}")
    return source


def _source_key(source: dict[str, Any]) -> tuple[str, str]:
    return str(source["patch_id"]), str(source["s1_name"])


def _gold_label(row: dict[str, Any]) -> str:
    for key in ("target_label", "target_gold_answer", "expected_answer"):
        value = row.get(key)
        if value is not None:
            label = str(value).strip().lower()
            if label in {"yes", "no"}:
                return label
            raise ValueError(f"non-binary gold label in {row.get('fixture_id')}: {value!r}")
    raise ValueError(f"gold label missing for {row.get('fixture_id')}")


def count_candidate_source_reuse(assignments: list[dict[str, Any]]) -> dict[str, int]:
    """Count second-congruent reuse by physical S1/S2 source pair, not annotation."""
    counts = Counter(
        f"{row['candidate_patch_id']}\0{row['candidate_s1_name']}"
        for row in assignments
    )
    return dict(sorted(counts.items()))


def _slot_for_sensor(row: dict[str, Any], sensor: str) -> int:
    refs = row.get("image_refs") or []
    matches = [int(ref["slot"]) for ref in refs if ref.get("sensor") == sensor]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {sensor} image in {row.get('fixture_id')}")
    return matches[0]


def _mechanical_slot_prompt(row: dict[str, Any], requested_slot: int) -> str:
    requested = str(row["requested_sensor"]).upper()
    other = "SAR" if requested == "OPTICAL" else "OPTICAL"
    prompt = str(row["user_prompt"])
    # Only the image ordinals change. Question wording, answer set and parser
    # instructions are copied byte-for-byte from the frozen row.
    import re

    prompt = re.sub(r"Image [12] is the requested (?:OPTICAL|SAR) evidence\.",
                    f"Image {requested_slot} is the requested {requested} evidence.", prompt)
    nonrequested_slot = 1 if requested_slot == 2 else 2
    prompt = re.sub(r"Image [12] is non-requested (?:OPTICAL|SAR) evidence\.",
                    f"Image {nonrequested_slot} is non-requested {other} evidence.", prompt)
    prompt = re.sub(r"Answer using only Image [12]\.",
                    f"Answer using only Image {requested_slot}.", prompt)
    if f"Image {requested_slot} is the requested {requested} evidence." not in prompt:
        raise ValueError(f"could not map requested image position for {row.get('fixture_id')}")
    if f"Answer using only Image {requested_slot}." not in prompt:
        raise ValueError(f"could not map answer position for {row.get('fixture_id')}")
    return prompt


def _normalize_dual_row(
    row: dict[str, Any], *, image_order: tuple[str, str], position_role: str,
    fixture_suffix: str = "",
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    target_source = _target_source(row)
    out["target_patch_id"] = str(target_source["patch_id"])
    out["target_s1_name"] = str(target_source["s1_name"])
    refs_by_sensor = {str(ref["sensor"]): copy.deepcopy(ref) for ref in row.get("image_refs") or []}
    if set(refs_by_sensor) != {"optical", "sar"}:
        raise ValueError(f"dual row must contain one optical and one SAR ref: {row.get('fixture_id')}")
    if len(row.get("image_refs") or []) != 2:
        raise ValueError(f"dual row image count mismatch: {row.get('fixture_id')}")
    requested = str(row["requested_sensor"]).lower()
    nonrequested = "sar" if requested == "optical" else "optical"
    refs: list[dict[str, Any]] = []
    for slot, sensor in enumerate(image_order, start=1):
        ref = refs_by_sensor[sensor]
        ref["slot"] = slot
        ref["role"] = "target_requested" if sensor == requested else "distractor_non_requested"
        refs.append(ref)
    requested_slot = next(int(ref["slot"]) for ref in refs if ref["sensor"] == requested)
    out["image_refs"] = refs
    out["image_order"] = "optical_first_then_sar" if image_order == ("optical", "sar") else "sar_first_then_optical"
    out["row_family"] = "dual"
    out["requested_image_position"] = requested_slot
    out["requested_position"] = f"slot_{requested_slot}"
    out["position_role"] = position_role
    out["target_source_cluster_id"] = str(out["target_patch_id"])
    out["target_label"] = _gold_label(out)
    out["gold_answer"] = _gold_label(out)
    out["non_requested_sensor"] = nonrequested
    out["distractor_label"] = str(out.get("distractor_label", out.get("distractor_answer", ""))).lower()
    out["distractor_answer"] = out["distractor_label"]
    out["fixture_id"] = f"{row['fixture_id']}{fixture_suffix}"
    out["source_fixture_id"] = str(row["fixture_id"])
    out["is_replay"] = False
    out["intended_use"] = "unified_128_eight_model_comparison"
    out["scientific_sample_unit"] = str(out["target_patch_id"])
    out["user_prompt"] = _mechanical_slot_prompt(out, requested_slot)
    return out


def _image_ref_for_source(
    source: dict[str, Any], sensor: str, assets: dict[str, dict[str, str]], *, role: str,
) -> dict[str, Any]:
    patch_id, s1_name = _source_key(source)
    if sensor == "sar":
        digest = assets["s1_sar"].get(s1_name)
        if not digest:
            raise ValueError(f"missing required S1 render hash for {s1_name}")
        return {
            "relative_path": f"s1/{s1_name}.png", "role": role, "sensor": "sar",
            "sha256": digest, "slot": 0, "source_id": s1_name, "paired_patch_id": patch_id,
        }
    if sensor == "optical":
        digest = assets["s2_optical"].get(patch_id)
        if not digest:
            raise ValueError(f"missing required S2 render hash for {patch_id}")
        return {
            "relative_path": f"s2/{patch_id}.png", "role": role, "sensor": "optical",
            "sha256": digest, "slot": 0, "source_id": patch_id, "paired_s1_name": s1_name,
        }
    raise ValueError(f"unsupported sensor: {sensor}")


def _validate_source_metadata(rows: list[dict[str, Any]], excluded_ids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    seen_cell: set[tuple[str, str, str]] = set()
    for row in rows:
        target_id = str(row.get("target_id", ""))
        if not target_id:
            raise ValueError("target_id missing")
        _target_source(row)
        if target_id in excluded_ids:
            continue
        key = (target_id, str(row.get("requested_sensor")), str(row.get("condition_id")))
        if key in seen_cell:
            raise ValueError(f"duplicate target/direction/condition in frozen fixture: {key}")
        seen_cell.add(key)
        prior = records.get(target_id)
        fields = (
            str(row.get("question_norm", "")),
            _gold_label(row),
            _source_key(_target_source(row)),
        )
        if prior is None:
            records[target_id] = {"row": row, "fields": fields}
        elif prior["fields"] != fields:
            raise ValueError(f"target metadata changes within fixture block: {target_id}")
    return records


def _choose_second_congruent(
    fixture_rows: list[dict[str, Any]], smoke_target_ids: set[str], selection_seed: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, set[str]]]]:
    records = _validate_source_metadata(fixture_rows, smoke_target_ids)
    by_question_label: dict[tuple[str, str], list[str]] = defaultdict(list)
    for target_id, record in records.items():
        row = record["row"]
        by_question_label[(str(row["question_norm"]), _gold_label(row))].append(target_id)
    fixed_sources: dict[str, dict[str, set[str]]] = {}
    for target_id in records:
        fixed_sources[target_id] = {"patch_id": set(), "s1_name": set()}
        for row in fixture_rows:
            if str(row.get("target_id")) != target_id:
                continue
            distractor = row.get("distractor_source") or {}
            patch_id = row.get("distractor_patch_id") or distractor.get("patch_id")
            s1_name = row.get("distractor_s1_name") or distractor.get("s1_name")
            if patch_id:
                fixed_sources[target_id]["patch_id"].add(str(patch_id))
            if s1_name:
                fixed_sources[target_id]["s1_name"].add(str(s1_name))
    assignments: dict[str, dict[str, Any]] = {}
    for target_id, record in records.items():
        row = record["row"]
        target_source = _target_source(row)
        target_patch, target_s1 = _source_key(target_source)
        key = (str(row["question_norm"]), _gold_label(row))
        candidates = []
        for candidate_id in by_question_label[key]:
            candidate = records[candidate_id]["row"]
            candidate_source = _target_source(candidate)
            patch_id, s1_name = _source_key(candidate_source)
            if candidate_id == target_id or patch_id == target_patch or s1_name == target_s1:
                continue
            if patch_id in fixed_sources[target_id]["patch_id"] or s1_name in fixed_sources[target_id]["s1_name"]:
                continue
            candidates.append((candidate_id, candidate, candidate_source))
        if candidates:
            candidates.sort(key=lambda item: (
                _sha256_text(f"{selection_seed}\0BEN_SECOND_CONGRUENT_V1\0{target_id}\0{item[0]}"),
                item[0],
            ))
            candidate_id, candidate_row, candidate_source = candidates[0]
            assignments[target_id] = {
                "candidate_target_id": candidate_id,
                "candidate_row": candidate_row,
                "candidate_source": copy.deepcopy(candidate_source),
                "selection_key_sha256": _sha256_text(
                    f"{selection_seed}\0BEN_SECOND_CONGRUENT_V1\0{target_id}\0{candidate_id}"
                ),
            }
    return assignments, fixed_sources


def _make_second_congruent_row(
    base_row: dict[str, Any], assignment: dict[str, Any], assets: dict[str, dict[str, str]],
) -> dict[str, Any]:
    out = copy.deepcopy(base_row)
    requested = str(out["requested_sensor"]).lower()
    nonrequested = "sar" if requested == "optical" else "optical"
    candidate_source = assignment["candidate_source"]
    refs = [copy.deepcopy(ref) for ref in out["image_refs"]]
    nonrequested_index = next(i for i, ref in enumerate(refs) if ref["sensor"] == nonrequested)
    replacement = _image_ref_for_source(candidate_source, nonrequested, assets, role="distractor_non_requested")
    replacement["slot"] = refs[nonrequested_index]["slot"]
    refs[nonrequested_index] = replacement
    out["image_refs"] = refs
    out["condition_id"] = CONTROL_CONDITION
    out["bigearthnet_condition_id"] = "A0_SECOND_CONGRUENT"
    out["condition_relation"] = "congruent"
    out["distractor_label"] = str(out["target_label"])
    out["distractor_answer"] = str(out["target_label"])
    candidate_meta = copy.deepcopy(candidate_source)
    candidate_meta.update({
        "answer": str(out["target_label"]),
        "relation_to_target": "congruent",
        "role": "second_congruent",
    })
    out["distractor_source"] = candidate_meta
    out["distractor_patch_id"] = str(candidate_source["patch_id"])
    out["distractor_s1_name"] = str(candidate_source["s1_name"])
    out["distractor_annotation_id"] = str(candidate_source.get("annotation_id", ""))
    out["fixture_id"] = (
        f"{out['target_id']}_{requested}_{CONTROL_CONDITION}_"
        f"{out['requested_image_position']}"
    )
    out["source_fixture_id"] = f"{base_row['source_fixture_id']}__SECOND_CONGRUENT"
    out["second_congruent_candidate_target_id"] = assignment["candidate_target_id"]
    out["second_congruent_selection_key_sha256"] = assignment["selection_key_sha256"]
    out["is_replay"] = False
    return out


def _to_earthmind_named_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a normalized dual row to EarthMind's native named RGB/SAR branches."""
    out = copy.deepcopy(row)
    refs_by_sensor = {str(ref["sensor"]).lower(): copy.deepcopy(ref) for ref in row.get("image_refs") or []}
    if set(refs_by_sensor) != {"optical", "sar"}:
        raise ValueError(f"EarthMind row requires one optical and one SAR ref: {row.get('fixture_id')}")
    rgb_ref = refs_by_sensor["optical"]
    sar_ref = refs_by_sensor["sar"]
    rgb_branch = {
        "relative_path": str(rgb_ref["relative_path"]),
        "sha256": str(rgb_ref["sha256"]),
        "patch_id": str(rgb_ref.get("source_id") or rgb_ref.get("paired_patch_id") or ""),
        "s1_name": str(rgb_ref.get("paired_s1_name") or rgb_ref.get("source_id") or ""),
        "sensor": "optical",
    }
    sar_branch = {
        "relative_path": str(sar_ref["relative_path"]),
        "sha256": str(sar_ref["sha256"]),
        "patch_id": str(sar_ref.get("paired_patch_id") or sar_ref.get("source_id") or ""),
        "s1_name": str(sar_ref.get("source_id") or ""),
        "sensor": "SAR",
    }
    if not rgb_branch["patch_id"] or not rgb_branch["s1_name"] or not sar_branch["patch_id"] or not sar_branch["s1_name"]:
        raise ValueError(f"EarthMind branch source mapping incomplete: {row.get('fixture_id')}")
    requested = str(out["requested_sensor"]).lower()
    requested_name = "optical/RGB" if requested == "optical" else "SAR"
    prompt = (
        f"Answer the binary presence question using the requested {requested_name} evidence. "
        "The other named sensor is non-requested context and must not override it. "
        f"Question: {out['question_text']} Reply with exactly one of: yes, no, abstain."
    )
    branches = {"rgb_image": rgb_branch, "sar_image": sar_branch}
    out.update({
        "row_family": "dual_named",
        "input_mode": "earthmind_named_dual_sensor_branches",
        "prompt_template_id": "earthmind_native_branch_control_v1",
        "image_paths": copy.deepcopy(branches),
        "native_sensor_branches": copy.deepcopy(branches),
        "image_1": None,
        "image_2": None,
        "image_order": "native_named_rgb_sar_branches",
        "requested_image_position": "NOT_APPLICABLE_NATIVE_NAMED_BRANCH",
        "requested_position": "named_sar_rgb",
        "position_role": "not_applicable",
        "prompt": prompt,
        "user_prompt": prompt,
    })
    return out


def _make_auxiliary_row(base_a0: dict[str, Any], condition: str) -> dict[str, Any]:
    out = copy.deepcopy(base_a0)
    requested = str(out["requested_sensor"]).lower()
    sensor_name = requested.upper()
    other_name = ("SAR" if requested == "optical" else "OPTICAL")
    question = str(out["question_text"])
    candidate_text = "Candidate answers: yes, no, abstain.\nReturn exactly one JSON object matching this schema: {\"answer\":\"yes|no|abstain\"}"
    if condition == "REQUESTED_ONLY":
        ref = next(copy.deepcopy(x) for x in out["image_refs"] if x["sensor"] == requested)
        ref["role"] = "target_requested"
        ref["slot"] = 1
        text = (
            f"Only Image 1 is shown. Image 1 is the requested {sensor_name} evidence. "
            f"Answer using only Image 1.\nQuestion: {question}\n{candidate_text}"
        )
        refs = [ref]
        relation = "requested_only"
    elif condition == "NONREQUESTED_ONLY":
        ref = next(copy.deepcopy(x) for x in out["image_refs"] if x["sensor"] != requested)
        ref["role"] = "distractor_non_requested"
        ref["slot"] = 1
        text = (
            f"Only Image 1 is shown. It is non-requested {other_name} evidence; "
            f"requested {sensor_name} evidence is not shown. Answer the same question about the "
            f"requested evidence using only the image shown.\nQuestion: {question}\n{candidate_text}"
        )
        refs = [ref]
        relation = "nonrequested_only"
    elif condition == "TEXT_ONLY":
        refs = []
        text = f"No image is shown. Answer the same question from text alone.\nQuestion: {question}\n{candidate_text}"
        relation = "text_only"
    else:
        raise ValueError(f"unknown auxiliary condition: {condition}")
    out.update({
        "fixture_id": f"{out['target_id']}_{requested}_{condition}",
        "source_fixture_id": str(base_a0["source_fixture_id"]),
        "condition_id": condition,
        "bigearthnet_condition_id": condition,
        "condition_relation": relation,
        "row_family": "auxiliary",
        "requested_image_position": "not_applicable",
        "requested_position": "no_image" if condition == "TEXT_ONLY" else "single_image",
        "position_role": "not_applicable",
        "image_refs": refs,
        "image_order": "single_or_no_image",
        "user_prompt": text,
        "is_replay": False,
        "intended_use": "unified_128_eight_model_comparison",
    })
    return out


def _validate_dual_rows(rows: list[dict[str, Any]]) -> None:
    ids = [str(row["fixture_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate generated fixture IDs")
    for row in rows:
        if row.get("row_family") not in {"dual", "dual_named"}:
            continue
        refs = row.get("image_refs") or []
        if len(refs) != 2:
            raise ValueError(f"dual image count invalid for {row['fixture_id']}")
        if row.get("row_family") == "dual" and [int(ref.get("slot", 0)) for ref in refs] != [1, 2]:
            raise ValueError(f"generic dual image ordering invalid for {row['fixture_id']}")
        requested = str(row["requested_sensor"])
        target_refs = [ref for ref in refs if ref.get("role") == "target_requested"]
        distractor_refs = [ref for ref in refs if ref.get("role") == "distractor_non_requested"]
        if len(target_refs) != 1 or len(distractor_refs) != 1:
            raise ValueError(f"dual roles invalid for {row['fixture_id']}")
        if target_refs[0].get("sensor") != requested:
            raise ValueError(f"requested image sensor mismatch for {row['fixture_id']}")
        if str(row["target_patch_id"]) == str(row["distractor_source"].get("patch_id")):
            raise ValueError(f"target/distractor patch overlap for {row['fixture_id']}")
        if str(row["target_s1_name"]) == str(row["distractor_source"].get("s1_name")):
            raise ValueError(f"target/distractor S1 overlap for {row['fixture_id']}")
        if str(row["condition_relation"]) == "congruent" and str(row["distractor_label"]) != str(row["target_label"]):
            raise ValueError(f"congruent label mismatch for {row['fixture_id']}")
        if str(row["condition_relation"]) == "opposite" and str(row["distractor_label"]) == str(row["target_label"]):
            raise ValueError(f"opposite label mismatch for {row['fixture_id']}")
        if row["user_prompt"].find(str(row["target_id"])) >= 0:
            raise ValueError(f"target ID leaked into user prompt for {row['fixture_id']}")
        if row.get("row_family") == "dual_named":
            branches = row.get("native_sensor_branches") or {}
            if row.get("input_mode") != "earthmind_named_dual_sensor_branches":
                raise ValueError(f"EarthMind row lacks native named-branch input mode: {row['fixture_id']}")
            if set(branches) != {"rgb_image", "sar_image"} or row.get("image_paths") != branches:
                raise ValueError(f"EarthMind named branch mapping invalid: {row['fixture_id']}")
            if "Image 1" in row["prompt"] or "Image 2" in row["prompt"]:
                raise ValueError(f"EarthMind named prompt contains positional wording: {row['fixture_id']}")
            requested_branch = "rgb_image" if requested == "optical" else "sar_image"
            distractor_branch = "sar_image" if requested == "optical" else "rgb_image"
            target_source = _target_source(row)
            if _source_key({
                "patch_id": branches[requested_branch]["patch_id"],
                "s1_name": branches[requested_branch]["s1_name"],
            }) != _source_key(target_source):
                raise ValueError(f"EarthMind requested branch does not contain target evidence: {row['fixture_id']}")
            distractor_source = row.get("distractor_source") or {}
            if _source_key({
                "patch_id": branches[distractor_branch]["patch_id"],
                "s1_name": branches[distractor_branch]["s1_name"],
            }) != _source_key(distractor_source):
                raise ValueError(f"EarthMind non-requested branch does not match assigned distractor: {row['fixture_id']}")


def _build_replays(rows: list[dict[str, Any]], *, earthmind: bool, seed: str) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        is_dual = row.get("row_family") == "dual" or (earthmind and row.get("row_family") == "dual_named")
        if is_dual:
            if earthmind:
                key = (str(row["requested_sensor"]), str(row["condition_id"]))
            else:
                key = (
                    str(row["requested_sensor"]), str(row["requested_image_position"]),
                    str(row["condition_id"]),
                )
        else:
            if earthmind:
                continue
            key = (str(row["requested_sensor"]), str(row["condition_id"]))
        groups[key].append(row)
    selected: list[dict[str, Any]] = []
    if earthmind:
        for key in sorted(groups):
            ranked = sorted(groups[key], key=lambda row: (_sha256_text(f"{seed}\0{row['fixture_id']}"), row["fixture_id"]))
            selected.extend(ranked[:2])
        remaining = [row for key in sorted(groups) for row in sorted(
            groups[key], key=lambda item: (_sha256_text(f"{seed}\0{item['fixture_id']}"), item["fixture_id"])
        )[2:]]
        selected_ids = {row["fixture_id"] for row in selected}
        remaining.sort(key=lambda row: (_sha256_text(f"{seed}\0earthmind_extra\0{row['fixture_id']}"), row["fixture_id"]))
        selected.extend(row for row in remaining if row["fixture_id"] not in selected_ids)
        selected = selected[:26]
    else:
        required = set()
        for sensor in ("optical", "sar"):
            for slot in (1, 2):
                for condition in ("A0_CONGRUENT", CONTROL_CONDITION, *BASE_CONDITIONS[1:]):
                    required.add((sensor, str(slot), condition))
            for condition in AUXILIARY_CONDITIONS:
                required.add((sensor, condition))
        for key in sorted(required):
            if key in groups and groups[key]:
                selected.append(min(groups[key], key=lambda row: (_sha256_text(f"{seed}\0{row['fixture_id']}"), row["fixture_id"])))
        selected_ids = {row["fixture_id"] for row in selected}
        remaining = [row for row in rows if row.get("fixture_id") not in selected_ids]
        remaining.sort(key=lambda row: (_sha256_text(f"{seed}\0replay_fill\0{row['fixture_id']}"), row["fixture_id"]))
        selected.extend(remaining[: max(0, 26 - len(selected))])
    if len({row["fixture_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate replay fixture IDs")
    return [{"fixture_id": str(row["fixture_id"]), "call_role": "replay"} for row in selected]


def construct_bigearthnet_matrix_rows(
    fixture_rows: list[dict[str, Any]], *, smoke_target_ids: set[str],
    render_assets: dict[str, dict[str, str]], selection_seed: str,
) -> dict[str, Any]:
    if not fixture_rows:
        raise ValueError("frozen source fixture is empty")
    by_target = _validate_source_metadata(fixture_rows, smoke_target_ids)
    if len(by_target) != len({str(r["target_id"]) for r in fixture_rows if str(r["target_id"]) not in smoke_target_ids}):
        raise ValueError("inconsistent target source grouping")
    # The frozen BigEarthNet main order is optical/S2 first, SAR/S1 second.
    cell_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in fixture_rows:
        target_id = str(row["target_id"])
        if target_id in smoke_target_ids:
            continue
        if str(row.get("condition_id")) not in BASE_CONDITIONS:
            raise ValueError(f"unknown frozen condition ID: {row.get('condition_id')}")
        key = (target_id, str(row["requested_sensor"]).lower(), str(row["condition_id"]))
        if key in cell_rows:
            raise ValueError(f"duplicate frozen input cell: {key}")
        cell_rows[key] = row
    target_ids = sorted(by_target)
    expected_cells = len(target_ids) * 2 * len(BASE_CONDITIONS)
    if len(cell_rows) != expected_cells:
        raise ValueError(f"expected {expected_cells} base target/direction/condition rows, got {len(cell_rows)}")

    assignments, fixed_sources = _choose_second_congruent(fixture_rows, smoke_target_ids, selection_seed)
    # Asset membership is checked against the prior CPU-validated 432-pair manifest.
    for target_id in target_ids:
        source = _target_source(by_target[target_id]["row"])
        if source["s1_name"] not in render_assets["s1_sar"]:
            raise ValueError(f"missing required S1 render hash for {source['s1_name']}")
        if source["patch_id"] not in render_assets["s2_optical"]:
            raise ValueError(f"missing required S2 render hash for {source['patch_id']}")

    generic_rows: list[dict[str, Any]] = []
    earthmind_rows: list[dict[str, Any]] = []
    assignment_manifest: list[dict[str, Any]] = []
    for target_id in target_ids:
        for requested in ("optical", "sar"):
            # Keep every frozen source assignment at the native optical-first/SAR-second position.
            for condition in BASE_CONDITIONS:
                source_row = cell_rows[(target_id, requested, condition)]
                main = _normalize_dual_row(
                    source_row, image_order=("optical", "sar"), position_role="dataset_main_order"
                )
                generic_rows.append(main)
                earth_row = _to_earthmind_named_row(main)
                earth_row["fixture_id"] = str(source_row["fixture_id"])
                earthmind_rows.append(earth_row)
                # Generic-model position control: swap only image order/ordinal references.
                generic_rows.append(_normalize_dual_row(
                    source_row, image_order=("sar", "optical"),
                    position_role="position_control", fixture_suffix="__POSITION_CONTROL",
                ))
            for condition in AUXILIARY_CONDITIONS:
                a0 = cell_rows[(target_id, requested, "A0_CONGRUENT")]
                generic_rows.append(_make_auxiliary_row(
                    _normalize_dual_row(a0, image_order=("optical", "sar"), position_role="dataset_main_order"),
                    condition,
                ))
        if target_id in assignments:
            assignment = assignments[target_id]
            assignment_manifest.append({
                "target_id": target_id,
                "target_patch_id": str(_target_source(by_target[target_id]["row"])["patch_id"]),
                "question_norm": str(by_target[target_id]["row"]["question_norm"]),
                "target_label": _gold_label(by_target[target_id]["row"]),
                "candidate_target_id": str(assignment["candidate_target_id"]),
                "candidate_patch_id": str(assignment["candidate_source"]["patch_id"]),
                "candidate_s1_name": str(assignment["candidate_source"]["s1_name"]),
                "candidate_annotation_id": str(assignment["candidate_source"].get("annotation_id", "")),
                "selection_key_sha256": assignment["selection_key_sha256"],
                "candidate_source_used_by_other_targets": False,
                "within_target_target_distractor_overlap": False,
            })
            for requested in ("optical", "sar"):
                a0 = cell_rows[(target_id, requested, "A0_CONGRUENT")]
                base_main = _normalize_dual_row(
                    a0, image_order=("optical", "sar"), position_role="dataset_main_order"
                )
                second = _make_second_congruent_row(base_main, assignment, render_assets)
                generic_rows.append(second)
                swapped_second = _normalize_dual_row(
                    second, image_order=("sar", "optical"),
                    position_role="position_control", fixture_suffix="__POSITION_CONTROL",
                )
                # Re-normalization must preserve the new control source metadata.
                swapped_second["condition_id"] = CONTROL_CONDITION
                swapped_second["bigearthnet_condition_id"] = "A0_SECOND_CONGRUENT"
                swapped_second["condition_relation"] = "congruent"
                swapped_second["distractor_label"] = str(swapped_second["target_label"])
                swapped_second["distractor_answer"] = str(swapped_second["target_label"])
                swapped_second["distractor_source"] = copy.deepcopy(second["distractor_source"])
                swapped_second["distractor_patch_id"] = second["distractor_patch_id"]
                swapped_second["distractor_s1_name"] = second["distractor_s1_name"]
                swapped_second["distractor_annotation_id"] = second["distractor_annotation_id"]
                swapped_second["second_congruent_candidate_target_id"] = assignment["candidate_target_id"]
                swapped_second["second_congruent_selection_key_sha256"] = assignment["selection_key_sha256"]
                generic_rows.append(swapped_second)
                earth_second = _to_earthmind_named_row(second)
                earthmind_rows.append(earth_second)

    selected_candidate_use = count_candidate_source_reuse(assignment_manifest)
    for row in assignment_manifest:
        physical_key = f"{row['candidate_patch_id']}\0{row['candidate_s1_name']}"
        row["candidate_source_use_count"] = selected_candidate_use[physical_key]
        row["candidate_source_used_by_other_targets"] = selected_candidate_use[physical_key] > 1

    # Rebuild exact target/distractor non-overlap and label-validity guards after
    # all condition and position rows have been synthesized.
    _validate_dual_rows(generic_rows)
    _validate_dual_rows(earthmind_rows)
    generic_ids = {str(row["fixture_id"]) for row in generic_rows}
    earth_ids = {str(row["fixture_id"]) for row in earthmind_rows}
    generic_replays = _build_replays(generic_rows, earthmind=False, seed="BEN_UNIFIED_128_V1_REPLAY")
    earthmind_replays = _build_replays(earthmind_rows, earthmind=True, seed="BEN_UNIFIED_128_V1_EARTHMIND_REPLAY")
    if any(item["fixture_id"] not in generic_ids for item in generic_replays):
        raise ValueError("generic replay references unknown fixture ID")
    if any(item["fixture_id"] not in earth_ids for item in earthmind_replays):
        raise ValueError("EarthMind replay references unknown fixture ID")
    if any(len(row.get("image_refs") or []) not in (0, 1, 2) for row in generic_rows):
        raise ValueError("invalid number of images in a generic row")

    return {
        "generic_rows": generic_rows,
        "earthmind_rows": earthmind_rows,
        "second_congruent_assignments": assignment_manifest,
        "generic_replay_manifest": generic_replays,
        "earthmind_replay_manifest": earthmind_replays,
        "target_records": [by_target[target_id]["row"] for target_id in target_ids],
        "sensitivity_excluded_ids": sorted(EXPECTED_SENSITIVITY_EXCLUSIONS & set(target_ids)),
        "candidate_source_reuse_counts": dict(sorted(selected_candidate_use.items())),
        "fixed_sources": fixed_sources,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_bigearthnet_matrix_fixture(
    *, input_fixture: Path, smoke_freeze: Path, render_manifest: Path, output_dir: Path,
    selection_seed: str = "20260820",
) -> dict[str, Any]:
    fixture_rows = _load_jsonl(input_fixture)
    smoke_target_ids = {str(row["target_id"]) for row in _load_csv(smoke_freeze)}
    render_assets = load_render_assets(_load_csv(render_manifest))
    result = construct_bigearthnet_matrix_rows(
        fixture_rows, smoke_target_ids=smoke_target_ids,
        render_assets=render_assets, selection_seed=selection_seed,
    )
    if len(result["generic_replay_manifest"]) != 26 or len(result["earthmind_replay_manifest"]) != 26:
        raise ValueError("full BigEarthNet matrix must contain 26 predesignated replay calls per interface")
    output_dir.mkdir(parents=True, exist_ok=True)
    generic_path = output_dir / "BIGEARTHNET_GENERIC_128_FIXTURE.jsonl"
    earthmind_path = output_dir / "BIGEARTHNET_EARTHMIND_128_FIXTURE.jsonl"
    generic_replay_path = output_dir / "BIGEARTHNET_GENERIC_REPLAY_MANIFEST.jsonl"
    earthmind_replay_path = output_dir / "BIGEARTHNET_EARTHMIND_REPLAY_MANIFEST.jsonl"
    assignments_path = output_dir / "BIGEARTHNET_SECOND_CONGRUENT_ASSIGNMENTS.csv"
    source_path = output_dir / "BIGEARTHNET_MAIN_SOURCE_FREEZE.csv"
    sensitivity_path = output_dir / "BIGEARTHNET_SENSITIVITY_EXCLUSIONS.csv"
    _write_jsonl(generic_path, result["generic_rows"])
    _write_jsonl(earthmind_path, result["earthmind_rows"])
    _write_jsonl(generic_replay_path, result["generic_replay_manifest"])
    _write_jsonl(earthmind_replay_path, result["earthmind_replay_manifest"])
    assignment_rows = result["second_congruent_assignments"]
    reuse = result["candidate_source_reuse_counts"]
    for row in assignment_rows:
        physical_key = f"{row['candidate_patch_id']}\0{row['candidate_s1_name']}"
        row["candidate_source_use_count"] = reuse.get(physical_key, 0)
        row["candidate_source_used_by_other_targets"] = row["candidate_source_use_count"] > 1
    _write_csv(assignments_path, assignment_rows, [
        "target_id", "target_patch_id", "question_norm", "target_label", "candidate_target_id",
        "candidate_patch_id", "candidate_s1_name", "candidate_annotation_id",
        "selection_key_sha256", "candidate_source_use_count", "candidate_source_used_by_other_targets",
        "within_target_target_distractor_overlap",
    ])
    source_rows = []
    for row in result["target_records"]:
        source = _target_source(row)
        source_rows.append({
            "target_id": row["target_id"], "target_annotation_id": source.get("annotation_id", ""),
            "target_patch_id": source["patch_id"], "target_s1_name": source["s1_name"],
            "question_norm": row["question_norm"], "target_label": _gold_label(row),
            "class_id": row.get("class_id", ""), "country": source.get("country", ""),
            "season": source.get("season", ""), "split_origin": "bench",
        })
    _write_csv(source_path, source_rows, [
        "target_id", "target_annotation_id", "target_patch_id", "target_s1_name", "question_norm",
        "target_label", "class_id", "country", "season", "split_origin",
    ])
    smoke_overlap_ids = sorted(EXPECTED_SENSITIVITY_EXCLUSIONS & {str(x["target_id"]) for x in result["target_records"]})
    exclusions = []
    source_by_id = {str(row["target_id"]): row for row in result["target_records"]}
    for target_id in smoke_overlap_ids:
        row = source_by_id[target_id]
        src = _target_source(row)
        exclusions.append({
            "target_id": target_id, "target_patch_id": src["patch_id"],
            "reason": "whole physical source overlaps previously used smoke allocation",
        })
    _write_csv(sensitivity_path, exclusions, ["target_id", "target_patch_id", "reason"])
    manifest = {
        "matrix_version": "unified_128_eight_model_20261018_v1",
        "dataset": "BigEarthNet.txt",
        "source_parquet_opened_by_builder": False,
        "official_test_accessed": False,
        "sealed_final_accessed": False,
        "source_fixture_sha256": sha256_file(input_fixture),
        "smoke_freeze_sha256": sha256_file(smoke_freeze),
        "render_manifest_sha256": sha256_file(render_manifest),
        "selection_seed": selection_seed,
        "selection_rule": "SHA256(seed NUL BEN_SECOND_CONGRUENT_V1 NUL target_id NUL candidate_target_id), ascending; first same-question/same-label main-cohort source not target or any fixed per-target distractor source",
        "primary_questions": len(result["target_records"]),
        "primary_physical_target_patches": len({str(row["target_patch_id"]) for row in source_rows}),
        "second_congruent_questions": len(assignment_rows),
        "second_congruent_physical_target_patches": len({row["target_patch_id"] for row in assignment_rows}),
        "sensitivity_excluded_questions": len(exclusions),
        "sensitivity_excluded_physical_patches": len({row["target_patch_id"] for row in exclusions}),
        "sensitivity_questions": len(result["target_records"]) - len(exclusions),
        "sensitivity_physical_target_patches": len({str(row["target_patch_id"]) for row in source_rows} - {row["target_patch_id"] for row in exclusions}),
        "generic_unique_rows": len(result["generic_rows"]),
        "generic_replay_calls": len(result["generic_replay_manifest"]),
        "generic_total_calls": len(result["generic_rows"]) + len(result["generic_replay_manifest"]),
        "earthmind_unique_rows": len(result["earthmind_rows"]),
        "earthmind_replay_calls": len(result["earthmind_replay_manifest"]),
        "earthmind_total_calls": len(result["earthmind_rows"]) + len(result["earthmind_replay_manifest"]),
        "unique_s1_render_sources": len(render_assets["s1_sar"]),
        "unique_s2_render_sources": len(render_assets["s2_optical"]),
        "available_render_pair_mismatch_count": 0,
        "selected_second_congruent_candidate_source_count": len(reuse),
        "selected_second_congruent_candidate_source_reuse": sum(count > 1 for count in reuse.values()),
        "outputs": {},
    }
    for name, path, count in (
        ("generic_fixture", generic_path, len(result["generic_rows"])),
        ("earthmind_fixture", earthmind_path, len(result["earthmind_rows"])),
        ("generic_replay_manifest", generic_replay_path, len(result["generic_replay_manifest"])),
        ("earthmind_replay_manifest", earthmind_replay_path, len(result["earthmind_replay_manifest"])),
        ("second_congruent_assignments", assignments_path, len(assignment_rows)),
        ("main_source_freeze", source_path, len(source_rows)),
        ("sensitivity_exclusions", sensitivity_path, len(exclusions)),
    ):
        manifest["outputs"][name] = {"path": path.name, "rows": count, "sha256": sha256_file(path)}
    manifest_path = output_dir / "BIGEARTHNET_MATRIX_FIXTURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fixture", type=Path, required=True)
    parser.add_argument("--smoke-freeze", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", default="20260820")
    parser.add_argument("--expected-primary-questions", type=int, default=321)
    parser.add_argument("--expected-second-congruent-questions", type=int, default=215)
    args = parser.parse_args()
    manifest = build_bigearthnet_matrix_fixture(
        input_fixture=args.input_fixture, smoke_freeze=args.smoke_freeze,
        render_manifest=args.render_manifest, output_dir=args.output_dir,
        selection_seed=args.selection_seed,
    )
    if manifest["primary_questions"] != args.expected_primary_questions:
        raise SystemExit(f"primary question count mismatch: {manifest['primary_questions']}")
    if manifest["second_congruent_questions"] != args.expected_second_congruent_questions:
        raise SystemExit(f"second-congruent count mismatch: {manifest['second_congruent_questions']}")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
