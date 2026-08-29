from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .reproduce import table3_rows


COLORS = {"overall": "#333333", "optical": "#0072B2", "sar": "#D55E00", "grid": "#D9D9D9"}
MODELS = [
    ("Qwen2.5-VL-3B", "Qwen"),
    ("EarthMind-4B", "EarthMind"),
    ("LLaVA-OneVision-7B", "LLaVA-OV"),
]


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "mathtext.fontset": "stix",
    })


def _select(rows, dataset_prefix: str, model: str):
    matches = [row for row in rows if str(row["dataset"]).startswith(dataset_prefix) and row["model"] == model]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {dataset_prefix}/{model}, found {len(matches)}")
    return matches[0]


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)


def reproduce_figure2(summary_path: Path, output_dir: Path) -> None:
    configure_style()
    rows = table3_rows(summary_path)
    datasets = [("M4-SAR", "(a) M4-SAR"), ("BigEarthNet.txt", "(b) BigEarthNet.txt")]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), sharex=True, sharey=True)
    ybase = [2, 1, 0]
    offsets = {"optical": -0.18, "overall": 0.0, "sar": 0.18}
    markers = {"optical": "o", "overall": "D", "sar": "s"}
    for axis, (dataset_prefix, panel_label) in zip(axes, datasets):
        for y, (model, _) in zip(ybase, MODELS):
            row = _select(rows, dataset_prefix, model)
            overall, low, high = (float(row[key]) for key in ("overall_cifr", "overall_ci_low", "overall_ci_high"))
            axis.errorbar(
                overall, y + offsets["overall"], xerr=[[overall - low], [high - overall]],
                fmt=markers["overall"], color=COLORS["overall"], ecolor=COLORS["overall"],
                elinewidth=1.0, capsize=2.5, markersize=5.0, markeredgewidth=0.6, zorder=3,
            )
            axis.plot(float(row["optical_cifr"]), y + offsets["optical"], marker="o", color=COLORS["optical"], markersize=5.4, linestyle="none", zorder=3)
            axis.plot(float(row["sar_cifr"]), y + offsets["sar"], marker="s", color=COLORS["sar"], markersize=5.4, linestyle="none", zorder=3)
        axis.set_xlim(0.0, 0.55)
        axis.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        axis.set_xlabel("Conflict-induced flip rate (CIFR)")
        axis.grid(axis="x", color=COLORS["grid"], linewidth=0.6, zorder=0)
        axis.text(0.02, 0.97, panel_label, transform=axis.transAxes, ha="left", va="top", fontweight="bold")
    axes[0].set_yticks(ybase)
    axes[0].set_yticklabels([label for _, label in MODELS])
    handles = [
        Line2D([0], [0], marker="D", color=COLORS["overall"], linestyle="none", label="Overall (95% CI)", markersize=5),
        Line2D([0], [0], marker="o", color=COLORS["optical"], linestyle="none", label="Optical requested", markersize=5),
        Line2D([0], [0], marker="s", color=COLORS["sar"], linestyle="none", label="SAR requested", markersize=5),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.14, right=0.99, top=0.98, bottom=0.24, wspace=0.12)
    _save(fig, output_dir, "figure2_cross_dataset_cifr")


def reproduce_figure3(summary_path: Path, output_dir: Path) -> None:
    configure_style()
    rows = table3_rows(summary_path)
    datasets = [("M4-SAR", "M4-SAR"), ("BigEarthNet.txt", "BigEarthNet.txt")]
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.65), sharey=True)
    x, width = [0, 1], 0.32
    for axis, (model, label) in zip(axes, [(m, f"({chr(97+i)}) {short}") for i, (m, short) in enumerate(MODELS)]):
        selected = [_select(rows, prefix, model) for prefix, _ in datasets]
        optical = [float(row["optical_cifr"]) for row in selected]
        sar = [float(row["sar_cifr"]) for row in selected]
        axis.bar([value - width / 2 for value in x], optical, width, color=COLORS["optical"], label="Optical requested")
        axis.bar([value + width / 2 for value in x], sar, width, color=COLORS["sar"], label="SAR requested")
        axis.set_xticks(x)
        axis.set_xticklabels([name for _, name in datasets], rotation=18, ha="right")
        axis.set_ylim(0.0, 0.55)
        axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.text(0.02, 0.97, label, transform=axis.transAxes, ha="left", va="top", fontweight="bold")
    axes[0].set_ylabel("Conflict-induced flip rate (CIFR)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.29, wspace=0.13)
    _save(fig, output_dir, "figure3_directional_interference")
