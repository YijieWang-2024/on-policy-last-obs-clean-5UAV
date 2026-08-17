"""Bar chart of system gain for the five psi experiments."""

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
ANALYSIS_DIR = HERE.parent
DEFAULT_DATA_DIR = ANALYSIS_DIR / "results_30det_latest_20260817"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def psi_value(model: str) -> float:
    match = re.search(r"psi0p(\d+)", model)
    if not match:
        raise ValueError(f"Cannot parse psi from model name: {model}")
    return int(match.group(1)) / 10.0


def integer_format(value: float, _position: int) -> str:
    return f"{value:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=HERE / "plot.png")
    args = parser.parse_args()

    pooled_rows = read_csv(args.data_dir / "pooled_metrics.csv")
    rows = []
    for pooled in pooled_rows:
        model = pooled["model"]
        rows.append(
            {
                "model": model,
                "psi": psi_value(model),
                "gain": float(pooled["system_performance_true_all_GUs"]),
            }
        )
    rows.sort(key=lambda row: row["psi"])

    x = list(range(len(rows)))
    values = [row["gain"] for row in rows]
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
        }
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bars = ax.bar(
        x,
        values,
        color=OKABE_ITO[: len(rows)],
        edgecolor="black",
        linewidth=0.4,
    )
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value / 1000:.1f}k",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x, labels)
    ax.set_xlabel("Association threshold, ψ_min")
    ax.set_ylabel("System gain (mean)")
    ax.set_title("System gain under association-threshold sensitivity")
    ax.yaxis.set_major_formatter(FuncFormatter(integer_format))
    ax.set_ylim(0, max(values) * 1.14)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", length=4)
    fig.subplots_adjust(bottom=0.17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
