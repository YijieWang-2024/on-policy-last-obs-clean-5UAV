"""Plot the deterministic-vs-stochastic latest-checkpoint evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "latest_policy_modes_20260815" / "aggregate_metrics.csv"
OUTPUT = HERE / "latest_policy_modes_20260815" / "latest_policy_modes_summary.png"

PALETTE = {
    "local_psi0p5_deadline_off": "#0072B2",  # blue
    "local_psi0p7_deadline_off": "#D55E00",  # vermilion
}
MODE_HATCH = {"deterministic": "", "stochastic": "//"}


def load_rows() -> list[dict]:
    with DATA.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    order = [
        ("local_psi0p5_deadline_off", "deterministic"),
        ("local_psi0p5_deadline_off", "stochastic"),
        ("local_psi0p7_deadline_off", "deterministic"),
        ("local_psi0p7_deadline_off", "stochastic"),
    ]
    lookup = {(r["model"], r["action_mode"]): r for r in rows}
    labels = ["ψ=0.5\nDet", "ψ=0.5\nStoch", "ψ=0.7\nDet", "ψ=0.7\nStoch"]
    x = np.arange(len(order))

    panels = [
        ("system_performance_true_all_GUs", "System performance", "{:,.0f}"),
        ("env_association_score_pass_rate", "Threshold pass rate", "{:.1%}"),
        ("effective_md_association_rate", "Effective MD association rate", "{:.1%}"),
        ("md_admission_ratio", "MD admission ratio", "{:.1%}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    for ax, (metric, title, _) in zip(axes.flat, panels):
        means = []
        errors = []
        colors = []
        hatches = []
        for model, mode in order:
            row = lookup[(model, mode)]
            means.append(float(row[f"{metric}_mean"]))
            errors.append(float(row[f"{metric}_std"]))
            colors.append(PALETTE[model])
            hatches.append(MODE_HATCH[mode])
        bars = ax.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", linestyle=":", alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if "performance" in metric:
            ax.set_ylabel("mean ± episode std")
        else:
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("ratio")

    fig.suptitle("Latest checkpoint: deterministic vs stochastic evaluation (20 common seeds)")
    fig.savefig(OUTPUT, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
