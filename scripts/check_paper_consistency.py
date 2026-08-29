#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from sensor_faithfulness.paper.reproduce import table3_rows
from sensor_faithfulness.utils.io import sha256_file


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "fixtures/m4sar/COUNTERFACTUAL_CONFIRMATORY_FIXTURE.jsonl": "94b71c918476a266a901ee7d3430f16d3c104369c6ce29c1cce1a4e50555baef",
    "fixtures/bigearthnet/BIGEARTHNET_EXTERNAL_FULL_FIXTURE.jsonl": "262ebdcc3b09da82ed4508cae7edce9235857b864acd81b2130078be51fda6cb",
    "configs/bigearthnet_rendering.json": "deb9c5e958fb2c0578603000854417444f57c35b0341ca93bcdcbac7284f0b08",
    "results/paper_summary/cross_dataset_summary.csv": "04752168a58831a5200dbdb1c4d7078b74950e8f359e293e3722e0906ac437a5",
    "results/paper_summary/robust_flip_results.csv": "359cece0fb9fd0ae240af8293c2bd448ef4c9aa7223161b2507fc10a2eab4401",
}


hash_results = {}
for relative, expected in EXPECTED_HASHES.items():
    observed = sha256_file(ROOT / relative)
    hash_results[relative] = {"expected": expected, "observed": observed, "match": observed == expected}

rows = table3_rows(ROOT / "results" / "paper_summary" / "cross_dataset_summary.csv")
unique_pairs = {(row["dataset"], row["model"]) for row in rows}
report = {
    "hashes": hash_results,
    "summary_rows": len(rows),
    "unique_dataset_model_pairs": len(unique_pairs),
    "table3_ready": len(rows) == 6 and len(unique_pairs) == 6,
    "all_hashes_match": all(item["match"] for item in hash_results.values()),
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["all_hashes_match"] or not report["table3_ready"]:
    raise SystemExit(1)
