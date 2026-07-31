from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORDER = [
    "DC-PPO (50M+50M)",
    "MAPPO (50M+50M)",
    "Cartesian MAPPO",
    "E1: spatial",
    "E2: spatial + curriculum",
    "B0: standard + curriculum",
    "B1: B0 + neighbor positions",
]
COLORS = {
    "DC-PPO (50M+50M)": "#0072B2",
    "MAPPO (50M+50M)": "#D55E00",
    "Cartesian MAPPO": "#999999",
    "E1: spatial": "#CC79A7",
    "E2: spatial + curriculum": "#000000",
    "B0: standard + curriculum": "#E69F00",
    "B1: B0 + neighbor positions": "#009E73",
}
LINESTYLES = {
    "DC-PPO (50M+50M)": "--",
    "MAPPO (50M+50M)": "--",
    "Cartesian MAPPO": ":",
    "E1: spatial": "-.",
    "E2: spatial + curriculum": "-",
    "B0: standard + curriculum": "-",
    "B1: B0 + neighbor positions": "-",
}
PANELS = [
    ("true60", "True performance over all 60 MDs", "Performance", False),
    ("equivalent60", "Equivalent full-60 performance", "Performance", False),
    ("system_performance", "Current system performance", "Performance", False),
    ("admission", "Overall MD admission ratio", "Ratio", True),
    ("upper_admission", "Upper-right admission ratio", "Ratio", True),
    ("completion", "Task completion ratio", "Ratio", True),
]


def trailing_step_mean(steps: np.ndarray, values: np.ndarray, span: float = 2_500_000) -> np.ndarray:
    spacing = float(np.median(np.diff(steps))) if len(steps) > 1 else span
    window = max(3, int(round(span / spacing)))
    result = np.empty_like(values, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result


curves: dict[tuple[str, str], list[tuple[float, float]]] = {}
with (ROOT / "training_curves.csv").open(encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        curves.setdefault((row["experiment"], row["metric"]), []).append(
            (float(row["step"]), float(row["value"]))
        )

manifest = {}
with (ROOT / "run_manifest.csv").open(encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        manifest[row["experiment"]] = row

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.titlepad": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.6,
    "figure.titlesize": 12,
})

fig, axes = plt.subplots(3, 2, figsize=(8.2, 8.0), sharex=True)
legend_handles = []
legend_labels = []
for panel_index, (metric, title, ylabel, is_ratio) in enumerate(PANELS):
    ax = axes.flat[panel_index]
    for experiment in ORDER:
        points = sorted(curves[(experiment, metric)])
        steps = np.asarray([point[0] for point in points], dtype=float)
        values = np.asarray([point[1] for point in points], dtype=float)
        smooth = trailing_step_mean(steps, values)
        line, = ax.plot(
            steps / 1e6,
            smooth,
            color=COLORS[experiment],
            linestyle=LINESTYLES[experiment],
            linewidth=2.2 if experiment == "E2: spatial + curriculum" else 1.55,
            alpha=1.0 if experiment in ORDER[-3:] else 0.85,
        )
        ax.scatter(steps[-1] / 1e6, smooth[-1], color=COLORS[experiment], s=15, zorder=4)
        if panel_index == 0:
            legend_handles.append(line)
            status = manifest[experiment]["status"]
            suffix = "" if status == "complete" else f", {status}"
            legend_labels.append(f"{experiment} ({steps[-1]/1e6:.1f}M{suffix})")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    if is_ratio:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.text(-0.11, 1.02, f"({chr(97 + panel_index)})", transform=ax.transAxes, fontweight="bold")

for ax in axes[-1, :]:
    ax.set_xlabel("Environment steps (millions)")

fig.suptitle("Strict 1+5 regional-MD experiments under the same fixed UAV start", y=0.985)
fig.text(
    0.5,
    0.952,
    "Curves are trailing 2.5M-step means; dots mark each run's available endpoint; single training seed",
    ha="center",
    fontsize=8,
    color="0.35",
)
fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=2,
    frameon=False,
    columnspacing=1.3,
)
fig.subplots_adjust(top=0.91, bottom=0.18, left=0.09, right=0.99, hspace=0.34, wspace=0.24)
fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
