"""Reviewer-metric bars plus system-gain line without overall completion bars."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parent / "fixed_flight_psi_sweep_25det_20260817_combined_v2"

# Keep the original color order for the retained metrics and use a
# colorblind-friendly teal for CPU useful utilization.
BAR_COLORS = ["#0072B2", "#E69F00", "#009E73"]
GAIN_COLOR = "#000000"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def psi_value(model: str) -> float:
    match = re.search(r"Rpsi([0-9]+(?:\.[0-9]+)?)", model)
    if match:
        return float(match.group(1))
    match = re.search(r"psi0p(\d+)", model)
    if match:
        return int(match.group(1)) / 10.0
    raise ValueError(f"Cannot parse psi from model name: {model}")


def integer_format(value: float, _position: int) -> str:
    return f"{value:.0f}"


def percent_label(value: float) -> str:
    """Keep the bar heights unchanged but avoid displaying 100.0%."""
    if round(value, 1) >= 100.0:
        return "99.9%"
    return f"{value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=HERE / "plot.png")
    args = parser.parse_args()

    pooled_rows = read_csv(args.data_dir / "pooled_metrics.csv")
    metric_specs = (
        ("offload_ratio_pooled", "Offload ratio"),
        ("task_offloading_success_rate_pooled", "Offloading success rate"),
        ("cpu_useful_utilization", "CPU utilization"),
    )

    rows = []
    for pooled in pooled_rows:
        rows.append(
            {
                "model": pooled["model"],
                "psi": psi_value(pooled["model"]),
                "values": [float(pooled[column]) * 100 for column, _ in metric_specs],
                "gain": float(pooled["system_performance_true_all_GUs"]),
            }
        )
    rows.sort(key=lambda row: row["psi"])

    x = list(range(len(rows)))
    # The three bars in one psi group touch; integer x spacing leaves a gap
    # between neighboring psi groups.
    width = 0.22
    offsets = [-width, 0.0, width]
    labels = [f"ψ={row['psi']:.1f}" for row in rows]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bar_handles = []
    for metric_index, (_column, metric_label) in enumerate(metric_specs):
        values = [row["values"][metric_index] for row in rows]
        bars = ax.bar(
            [value + offsets[metric_index] for value in x],
            values,
            width,
            color=BAR_COLORS[metric_index],
            edgecolor="none",
            linewidth=0.0,
            label=metric_label,
        )
        bar_handles.append(bars[0])
        for bar, value in zip(bars, values):
            label_dx = {-1: 0, 0: -4, 1: 4}[metric_index - 1]
            ax.annotate(
                percent_label(value),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(label_dx, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax.set_xticks(x, labels)
    ax.set_xlabel("Association threshold, ψ_min")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_zorder(1)
    ax.patch.set_visible(False)
    ax2 = ax.twinx()
    ax2.set_zorder(3)
    gains = [row["gain"] for row in rows]
    (gain_line,) = ax2.plot(
        x,
        gains,
        color=GAIN_COLOR,
        linestyle="--",
        linewidth=1.25,
        marker="o",
        markersize=6.0,
        markerfacecolor="white",
        markeredgecolor=GAIN_COLOR,
        markeredgewidth=0.65,
        alpha=0.95,
        label="System gain (right axis)",
        zorder=5,
    )
    ax2.set_ylabel("System gain", color=GAIN_COLOR)
    ax2.yaxis.set_major_formatter(FuncFormatter(integer_format))
    ax2.tick_params(axis="y", colors=GAIN_COLOR, direction="out", length=4)
    ax2.set_ylim(100000.0, 685000.0)
    ax2.set_yticks([100000, 200000, 300000, 400000, 500000, 600000])

    ax.set_title(
        "Offloading, resource utilization, and system gain vs. ψ_min"
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GAIN_COLOR)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax.tick_params(direction="out", length=4)

    ax.legend(
        handles=bar_handles,
        labels=[label for _, label in metric_specs],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.86,
        edgecolor="none",
        handlelength=1.3,
        labelspacing=0.35,
    )
    ax2.legend(
        handles=[gain_line],
        labels=["System gain (right axis)"],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.86,
        edgecolor="none",
        handlelength=1.3,
    )

    fig.subplots_adjust(bottom=0.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
