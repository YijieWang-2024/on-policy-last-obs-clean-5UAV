"""Grouped bar chart for offloading and completion metrics."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DEFAULT_DATA_DIR = ANALYSIS_DIR / "results_30det_latest_20260817"
COLORS = ["#0072B2", "#E69F00", "#CC79A7"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def psi_value(model: str) -> float:
    match = re.search(r"psi0p(\d+)", model)
    if not match:
        raise ValueError(f"Cannot parse psi from model name: {model}")
    return int(match.group(1)) / 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=HERE / "plot.png")
    args = parser.parse_args()

    pooled_rows = read_csv(args.data_dir / "pooled_metrics.csv")
    metric_specs = (
        (
            "offload_ratio_pooled",
            "Offload ratio",
        ),
        (
            "task_offloading_success_rate_pooled",
            "Offloading success rate",
        ),
        (
            "overall_completion_pooled",
            "Overall completion rate",
        ),
    )

    rows = []
    for pooled in pooled_rows:
        model = pooled["model"]
        rows.append(
            {
                "model": model,
                "psi": psi_value(model),
                "values": [float(pooled[value]) * 100 for value, _ in metric_specs],
            }
        )
    rows.sort(key=lambda row: row["psi"])

    x = list(range(len(rows)))
    width = 0.23
    offsets = [-width, 0.0, width]
    labels = [f"ψ={row['psi']:.1f}" for row in rows]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    for metric_index, (_column, label) in enumerate(metric_specs):
        values = [row["values"][metric_index] for row in rows]
        bars = ax.bar(
            [value + offsets[metric_index] for value in x],
            values,
            width,
            color=COLORS[metric_index],
            edgecolor="black",
            linewidth=0.35,
            label=label,
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x, labels)
    ax.set_xlabel("Association threshold, ψ_min")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Offloading and completion under association-threshold sensitivity")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", length=4)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        handlelength=1.3,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.24)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
