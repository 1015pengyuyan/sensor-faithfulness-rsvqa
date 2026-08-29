#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_faithfulness.data.fixture_validation import validate_fixture
from sensor_faithfulness.utils.io import read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    observed = sha256_file(args.fixture)
    if args.expected_sha256 and observed != args.expected_sha256:
        raise SystemExit(f"SHA256 mismatch: {observed} != {args.expected_sha256}")
    report = validate_fixture(read_jsonl(args.fixture))
    report["sha256"] = observed
    report["valid"] = all(report[key] == 0 for key in (
        "duplicate_fixture_ids", "target_distractor_overlap", "label_violations",
        "missing_requested_evidence", "prohibited_test_references",
    ))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
