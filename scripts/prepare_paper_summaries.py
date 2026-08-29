#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "paper_summary" / "cross_dataset_summary.csv"


def write_subset(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fields = list(reader.fieldnames or [])

write_subset(ROOT / "results" / "paper_summary" / "m4sar_summary.csv", fields, [row for row in rows if row["dataset"] == "M4-SAR"])
write_subset(ROOT / "results" / "paper_summary" / "bigearthnet_summary.csv", fields, [row for row in rows if row["dataset"].startswith("BigEarthNet.txt")])
