from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_COLUMNS = {
    "dataset", "model", "a0_accuracy", "overall_cifr", "overall_ci_low",
    "overall_ci_high", "optical_cifr", "sar_cifr", "asym", "asym_ci_low",
    "asym_ci_high", "rsr",
}


def load_frozen_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"frozen summary missing columns: {sorted(missing)}")
    if len(frame) != 6:
        raise ValueError(f"expected six frozen dataset/model rows, found {len(frame)}")
    return frame


def table3_rows(path: Path) -> list[dict[str, Any]]:
    """Read paper values from their literal CSV tokens without recomputation."""
    numeric = EXPECTED_COLUMNS - {"dataset", "model"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise ValueError(f"expected six frozen dataset/model rows, found {len(rows)}")
    return [
        {key: (float(value) if key in numeric and value not in {"", "NA"} else value) for key, value in row.items()}
        for row in rows
    ]

def _f(value: float) -> str:
    return f"{float(value):.3f}"


def _ci(estimate: float, low: float, high: float) -> str:
    return f"\\shortstack{{{_f(estimate)}\\\\{{\\scriptsize [{_f(low)}, {_f(high)}]}}}}"


def write_table3(summary_path: Path, output_path: Path) -> None:
    rows = table3_rows(summary_path)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-dataset counterfactual flip results. A0 accuracy is computed over requested-direction pairs. CIFR and RSR are computed among A0-correct source--direction pairs. Brackets give 95\\% source-cluster bootstrap intervals for overall CIFR and ASYM; ASYM is SAR-requested minus optical-requested CIFR.}",
        "\\label{tab:cross_dataset_cifr}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Dataset & Model & A0 Acc. & CIFR [95\\% CI] & Opt. CIFR & SAR CIFR & ASYM [95\\% CI] & RSR \\\\",
        "\\midrule",
    ]
    for row in rows:
        dataset = "BigEarthNet.txt (321)" if str(row["dataset"]).startswith("BigEarthNet.txt") else str(row["dataset"])
        lines.append(
            f"{dataset} & {row['model']} & {_f(row['a0_accuracy'])} & "
            f"{_ci(row['overall_cifr'], row['overall_ci_low'], row['overall_ci_high'])} & "
            f"{_f(row['optical_cifr'])} & {_f(row['sar_cifr'])} & "
            f"{_ci(row['asym'], row['asym_ci_low'], row['asym_ci_high'])} & {_f(row['rsr'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
