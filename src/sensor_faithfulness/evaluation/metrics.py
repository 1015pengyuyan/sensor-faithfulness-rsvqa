"""CIFR, RSR, ASYM, robust flips, and paired source-cluster bootstrap.

The three conflict replicates are averaged within each source/direction before
pooling. Bootstrap resampling is performed over target sources, retaining all
requested directions and replicates for a sampled source. This avoids treating
replicates as independent observations.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np


VALID_ANSWERS = {"yes", "no", "abstain"}


def source_id(row: dict[str, Any]) -> str:
    for key in ("source_id", "target_source_id", "target_id", "example_id"):
        if row.get(key) is not None:
            return str(row[key])
    target = row.get("target_source")
    if isinstance(target, dict):
        for key in ("source_id", "patch_id", "s1_name"):
            if target.get(key) is not None:
                return str(target[key])
    raise KeyError(f"cannot infer target source for {row.get('fixture_id')}")


def _answer(row: dict[str, Any]) -> str | None:
    value = row.get("parsed_answer")
    if isinstance(value, str) and value.lower() in VALID_ANSWERS:
        return value.lower()
    return None


def _index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (source_id(row), str(row["requested_sensor"]).lower(), str(row["condition_id"]))
        if key in indexed:
            raise ValueError(f"duplicate source/requested/condition row: {key}")
        indexed[key] = row
    return indexed


def _source_direction_values(
    rows: list[dict[str, Any]],
    a0_condition: str,
    conflict_conditions: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    indexed = _index(rows)
    sources = sorted({key[0] for key in indexed})
    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in sources:
        for sensor in ("optical", "sar"):
            a0 = indexed.get((source, sensor, a0_condition))
            if a0 is None:
                continue
            target = str(a0.get("target_label", "")).lower()
            a0_answer = _answer(a0)
            record: dict[str, Any] = {
                "a0_correct": a0_answer == target,
                "a0_abstain": a0_answer == "abstain",
                "cifr_values": [],
                "rsr_values": [],
                "conflict_correct": [],
                "conflict_abstain": [],
                "replicate_cifr": {},
            }
            for condition in conflict_conditions:
                conflict = indexed.get((source, sensor, condition))
                if conflict is None:
                    continue
                answer = _answer(conflict)
                distractor = str(conflict.get("distractor_label", "")).lower()
                target_label = str(conflict.get("target_label", target)).lower()
                flip = float(answer == distractor and distractor in {"yes", "no"} and distractor != target_label)
                record["conflict_correct"].append(float(answer == target_label))
                record["conflict_abstain"].append(float(answer == "abstain"))
                if record["a0_correct"]:
                    record["cifr_values"].append(flip)
                    record["rsr_values"].append(float(answer == target_label))
                    record["replicate_cifr"][condition] = flip
            output[source][sensor] = record
    return dict(output)


def _flatten(
    source_values: dict[str, dict[str, dict[str, Any]]],
    field: str,
    sensor: str | None = None,
    sampled_sources: Sequence[str] | None = None,
) -> list[float]:
    selected = sampled_sources if sampled_sources is not None else sorted(source_values)
    values: list[float] = []
    for source in selected:
        records = source_values[source]
        sensors = (sensor,) if sensor else ("optical", "sar")
        for requested in sensors:
            record = records.get(requested)
            if not record:
                continue
            field_values = record.get(field, [])
            if field_values:
                values.append(float(mean(field_values)))
    return values


def _paired_asym_values(
    source_values: dict[str, dict[str, dict[str, Any]]],
    sampled_sources: Sequence[str] | None = None,
) -> list[float]:
    selected = sampled_sources if sampled_sources is not None else sorted(source_values)
    values: list[float] = []
    for source in selected:
        optical = source_values[source].get("optical", {}).get("cifr_values", [])
        sar = source_values[source].get("sar", {}).get("cifr_values", [])
        if optical and sar:
            values.append(float(mean(sar) - mean(optical)))
    return values


def _safe_mean(values: Sequence[float]) -> float | None:
    return float(mean(values)) if values else None


def compute_metrics(
    rows: list[dict[str, Any]],
    a0_condition: str,
    conflict_conditions: Sequence[str],
    sampled_sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    values = _source_direction_values(rows, a0_condition, conflict_conditions)
    selected = sampled_sources if sampled_sources is not None else sorted(values)
    cifr = _flatten(values, "cifr_values", sampled_sources=selected)
    rsr = _flatten(values, "rsr_values", sampled_sources=selected)
    optical = _flatten(values, "cifr_values", "optical", selected)
    sar = _flatten(values, "cifr_values", "sar", selected)
    asym = _paired_asym_values(values, selected)
    a0_correct = []
    a0_abstain = []
    conflict_correct = []
    conflict_abstain = []
    replicate: dict[str, list[float]] = {condition: [] for condition in conflict_conditions}
    for source in selected:
        for record in values[source].values():
            a0_correct.append(float(record["a0_correct"]))
            a0_abstain.append(float(record["a0_abstain"]))
            conflict_correct.extend(record["conflict_correct"])
            conflict_abstain.extend(record["conflict_abstain"])
            if record["a0_correct"]:
                for condition, value in record["replicate_cifr"].items():
                    replicate[condition].append(value)
    return {
        "a0_accuracy": _safe_mean(a0_correct),
        "a0_abstention": _safe_mean(a0_abstain),
        "conflict_accuracy": _safe_mean(conflict_correct),
        "conflict_abstention": _safe_mean(conflict_abstain),
        "cifr": _safe_mean(cifr),
        "rsr": _safe_mean(rsr),
        "optical_cifr": _safe_mean(optical),
        "sar_cifr": _safe_mean(sar),
        "asym": _safe_mean(asym),
        "eligible_n": len(cifr),
        "optical_eligible_n": len(optical),
        "sar_eligible_n": len(sar),
        "asym_eligible_n": len(asym),
        "replicate_cifr": {condition: _safe_mean(rep_values) for condition, rep_values in replicate.items()},
    }


def robust_flip_rate(
    rows: list[dict[str, Any]],
    a0_condition: str,
    conflict_conditions: Sequence[str],
    sampled_sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    values = _source_direction_values(rows, a0_condition, conflict_conditions)
    selected = sampled_sources if sampled_sources is not None else sorted(values)
    robust: list[float] = []
    for source in selected:
        for record in values[source].values():
            flips = list(record["replicate_cifr"].values())
            if record["a0_correct"] and len(flips) == len(conflict_conditions):
                robust.append(float(sum(flips) >= 2))
    return {"robust_flip_rate": _safe_mean(robust), "eligible_n": len(robust)}


def _interval(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, 0.025, method="linear")),
        float(np.quantile(array, 0.975, method="linear")),
    )


def bootstrap_metrics(
    rows: list[dict[str, Any]],
    a0_condition: str,
    conflict_conditions: Sequence[str],
    n_resamples: int = 10000,
    seed: int = 20260828,
) -> dict[str, Any]:
    source_values = _source_direction_values(rows, a0_condition, conflict_conditions)
    sources = sorted(source_values)
    if not sources:
        raise ValueError("no source clusters")
    point = compute_metrics(rows, a0_condition, conflict_conditions)
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[float]] = {key: [] for key in ("cifr", "rsr", "optical_cifr", "sar_cifr", "asym")}
    for _ in range(n_resamples):
        draw = [sources[index] for index in rng.integers(0, len(sources), len(sources))]
        metrics = compute_metrics(rows, a0_condition, conflict_conditions, draw)
        for key in sampled:
            value = metrics[key]
            if value is not None:
                sampled[key].append(float(value))
    output: dict[str, Any] = {
        "cluster_unit": "target_source",
        "n_source_clusters": len(sources),
        "n_resamples": n_resamples,
        "seed": seed,
    }
    for key, draws in sampled.items():
        low, high = _interval(draws)
        output[key] = point[key]
        output[f"{key}_ci_low"] = low
        output[f"{key}_ci_high"] = high
    output.update({key: point[key] for key in ("eligible_n", "optical_eligible_n", "sar_eligible_n", "asym_eligible_n")})
    return output
