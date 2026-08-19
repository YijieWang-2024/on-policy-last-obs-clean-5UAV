"""Reproducible color-vision-friendly plot for the 25-seed UAV-wise eval."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "eval25_per_uav_20260817_current"
INPUT = RESULTS / "per_uav_aggregate.csv"
OUTPUT = RESULTS / "per_uav_metrics_25det.png"

MODEL_ORDER = [
    "heterogeneous_local_latest",
    "homogeneous_remote_final",
    "hybrid_Fhomo_Rhetero",
]
MODEL_LABELS = {
    "heterogeneous_local_latest": "Native heterogeneous",
    "homogeneous_remote_final": "Native homogeneous",
    "hybrid_Fhomo_Rhetero": "Hybrid: homo flight + hetero resource",
}
# Okabe-Ito colors: blue, orange, and purple remain distinguishable for
# common red-green color-vision deficiencies; hatches provide a second cue.
COLORS = {
    "heterogeneous_local_latest": "#0072B2",
    "homogeneous_remote_final": "#E69F00",
    "hybrid_Fhomo_Rhetero": "#CC79A7",
}
HATCHES = {
    "heterogeneous_local_latest": "///",
    "homogeneous_remote_final": "\\\\",
    "hybrid_Fhomo_Rhetero": "...",
}


def load_rows(input_path: Path):
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot UAV-wise deterministic evaluation metrics.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    cli = parser.parse_args()
    input_path = cli.results_dir / "per_uav_aggregate.csv"
    output = cli.output or (cli.results_dir / "per_uav_metrics_25det.png")
    rows = load_rows(input_path)
    models = [model for model in MODEL_ORDER if any(row["model"] == model for row in rows)]
    if len(models) != 3:
        raise RuntimeError(f"Expected all three models, found {models}")
    n_uavs = sorted({int(row["uav_id"]) for row in rows})
    metrics = [
        ("service_md_count", "Served MD / episode", "MD count", 1.0),
        ("successful_service_md_count", "On-time served MD / episode", "MD count", 1.0),
        ("service_success_rate", "Service success rate", "%", 100.0),
        ("service_step_fraction", "Service-step fraction", "%", 100.0),
        ("episode_reward", "Episode reward", "reward", 1.0),
        ("mean_reward", "Average step reward", "reward / step", 1.0),
        ("system_performance_individual", "Individual system performance", "performance", 1.0),
    ]
    lookup = {(row["model"], int(row["uav_id"])): row for row in rows}
    x = np.arange(len(n_uavs), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 13.2), constrained_layout=False)
    axes = axes.ravel()
    for panel, (field, title, ylabel, scale) in enumerate(metrics):
        ax = axes[panel]
        for model_index, model in enumerate(models):
            means = np.array([
                float(lookup[(model, uav)][f"{field}_mean"]) * scale for uav in n_uavs
            ])
            stds = np.array([
                float(lookup[(model, uav)][f"{field}_std"]) * scale for uav in n_uavs
            ])
            offset = (model_index - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=2.5,
                color=COLORS[model],
                hatch=HATCHES[model],
                edgecolor="#303030",
                linewidth=0.65,
                error_kw={"elinewidth": 0.8, "ecolor": "#303030"},
                label=MODEL_LABELS[model],
            )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"UAV {uav}" for uav in n_uavs])
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if scale == 100.0:
            ax.set_ylim(bottom=0.0, top=105.0)

    axes[-2].axis("off")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[: len(models)],
        labels[: len(models)],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=3,
        frameon=True,
        fontsize=10,
    )
    fig.suptitle(
        "Fixed600-200 | 25 deterministic episodes (seeds 7001–7025)\n"
        "UAV-wise service, reward, and performance comparison",
        fontsize=15,
        y=0.995,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.86, wspace=0.28, hspace=0.32)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
