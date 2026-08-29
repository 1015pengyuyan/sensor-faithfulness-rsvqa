#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sensor_faithfulness.evaluation.metrics import bootstrap_metrics, compute_metrics, robust_flip_rate
from sensor_faithfulness.evaluation.parser import parse_answer


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    fixture = read_csv(args.fixture)
    predictions = {row["fixture_id"]: row for row in read_csv(args.predictions)}
    rows = []
    for row in fixture:
        prediction = predictions.get(row["fixture_id"])
        if prediction is None:
            raise ValueError(f"missing prediction: {row['fixture_id']}")
        enriched = dict(row)
        enriched["parsed_answer"] = prediction.get("parsed_answer") or parse_answer(prediction.get("raw_output", "")).answer
        rows.append(enriched)
    conditions = {row["condition_id"] for row in rows}
    if "A2_CONFLICT" in conditions:
        a0, reps = "A0_CONGRUENT", ("A2_CONFLICT", "A3_CONFLICT_REPLICATE_1", "A3_CONFLICT_REPLICATE_2")
    elif "A2_OPPOSITE_1" in conditions:
        a0, reps = "A0_CONGRUENT", ("A2_OPPOSITE_1", "A3_1_OPPOSITE_2", "A3_2_OPPOSITE_3")
    else:
        a0, reps = "A0", ("A2", "A3-1", "A3-2")
    output = compute_metrics(rows, a0, reps)
    output.update(robust_flip_rate(rows, a0, reps))
    if args.bootstrap:
        output["bootstrap"] = bootstrap_metrics(rows, a0, reps, args.bootstrap, args.seed)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
