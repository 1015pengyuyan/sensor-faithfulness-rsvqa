#!/usr/bin/env python3
from pathlib import Path

from sensor_faithfulness.paper.figures import reproduce_figure2


ROOT = Path(__file__).resolve().parents[1]
reproduce_figure2(ROOT / "results" / "paper_summary" / "cross_dataset_summary.csv", ROOT / "figures")
