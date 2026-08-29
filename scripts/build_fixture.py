#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sensor_faithfulness.data.fixture_construction import construct_target_rows
from sensor_faithfulness.data.fixture_validation import validate_fixture
from sensor_faithfulness.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, help="JSONL rows with source_id, question_id, and question_text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--image-order", choices=["requested_first", "optical_first_then_sar"], required=True)
    args = parser.parse_args()
    sources = read_jsonl(args.source_catalog)
    by_id = {str(source["source_id"]): source for source in sources}
    rows = []
    for item in read_jsonl(args.targets):
        target = by_id[str(item["source_id"])]
        rows.extend(construct_target_rows(
            target, sources, str(item["question_id"]), str(item["question_text"]),
            args.seed, args.image_order,
        ))
    report = validate_fixture(rows)
    if any(report[key] for key in (
        "duplicate_fixture_ids", "target_distractor_overlap", "label_violations",
        "missing_requested_evidence", "prohibited_test_references",
    )):
        raise SystemExit(f"fixture validation failed: {report}")
    write_jsonl(args.output, rows)
    print(report)


if __name__ == "__main__":
    main()
