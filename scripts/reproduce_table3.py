#!/usr/bin/env python3
from pathlib import Path

from sensor_faithfulness.paper.reproduce import write_table3


ROOT = Path(__file__).resolve().parents[1]
write_table3(
    ROOT / "results" / "paper_summary" / "cross_dataset_summary.csv",
    ROOT / "results" / "paper_summary" / "table3_reproduced.tex",
)
