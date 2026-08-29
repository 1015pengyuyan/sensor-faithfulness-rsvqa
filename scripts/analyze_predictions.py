#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pstdev

from sensor_faithfulness.evaluation.metrics import bootstrap_metrics, compute_metrics, robust_flip_rate
from sensor_faithfulness.evaluation.parser import parse_answer
from sensor_faithfulness.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--a0-condition", required=True)
    parser.add_argument("--conflict-condition", action="append", required=True)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    predictions = {row["fixture_id"]: row for row in read_jsonl(args.predictions)}
    rows = []
    for fixture_row in read_jsonl(args.fixture):
        predicted = predictions.get(fixture_row["fixture_id"])
        if predicted is None:
            raise ValueError(f"missing prediction: {fixture_row['fixture_id']}")
        row = dict(fixture_row)
        row["parsed_answer"] = predicted.get("parsed_answer") or parse_answer(str(predicted.get("raw_output", ""))).answer
        rows.append(row)
    metrics = compute_metrics(rows, args.a0_condition, args.conflict_condition)
    replicate_values = [value for value in metrics["replicate_cifr"].values() if value is not None]
    output = {
        "metrics": metrics,
        "robust_flip": robust_flip_rate(rows, args.a0_condition, args.conflict_condition),
        "replicate_stability": {
            "replicate_cifr": metrics["replicate_cifr"],
            "population_sd": pstdev(replicate_values) if len(replicate_values) > 1 else 0.0,
        },
        "bootstrap": bootstrap_metrics(
            rows, args.a0_condition, args.conflict_condition,
            n_resamples=args.n_resamples, seed=args.seed,
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
