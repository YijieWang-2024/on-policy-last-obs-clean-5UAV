from pathlib import Path
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[3] / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
RUNS = {
    "E2: spatial + curriculum": RESULTS / "regional_dynamic_strict1p5_cartesian_spatialflight_curriculum_e2_seed2_100m_r64_20260727" / "run1" / "logs",
    "B0: standard actor": RESULTS / "regional_dynamic_strict1p5_cartesian_curriculum_distance_b0_seed2_100m_r64_20260728" / "run1" / "logs",
    "B1: B0 + neighbor positions": RESULTS / "regional_dynamic_strict1p5_cartesian_curriculum_distance_neighbor_b1_seed2_100m_r64_20260728" / "run1" / "logs",
}
COLORS = {
    "E2: spatial + curriculum": "#0072B2",
    "B0: standard actor": "#E69F00",
    "B1: B0 + neighbor positions": "#009E73",
}
METRICS = [
    ("agent0/cumulative_reward", "Cumulative training reward", "Reward"),
    ("agent0/system_performance_true_all_GUs", "True performance over all 60 candidate MDs", "Performance"),
    ("agent0/system_performance", "Performance of admitted/active MDs", "Performance"),
]


def load_scalars(log_dir, tag):
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    values = accumulator.Scalars(tag)
    return np.asarray([v.step for v in values], dtype=float), np.asarray([v.value for v in values], dtype=float)


def trailing_step_mean(steps, values, span=2_500_000):
    if len(values) < 2:
        return values.copy()
    spacing = float(np.median(np.diff(steps)))
    window = max(3, int(round(span / spacing)))
    csum = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result[index] = (csum[index + 1] - csum[start]) / (index + 1 - start)
    return result


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titlepad": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
})

fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.3), sharex=True)
legend_handles = []
legend_labels = []
for metric_index, (tag, title, ylabel) in enumerate(METRICS):
    ax = axes[metric_index]
    for name, log_dir in RUNS.items():
        steps, values = load_scalars(log_dir, tag)
        smooth = trailing_step_mean(steps, values)
        x = steps / 1e6
        ax.plot(x, values, color=COLORS[name], alpha=0.11, linewidth=0.55)
        line, = ax.plot(x, smooth, color=COLORS[name], linewidth=2.0)
        ax.scatter(x[-1], smooth[-1], color=COLORS[name], s=18, zorder=4)
        if metric_index == 0:
            legend_handles.append(line)
            legend_labels.append(f"{name} ({x[-1]:.1f}M)")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}k"))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.text(-0.075, 1.02, f"({chr(97 + metric_index)})", transform=ax.transAxes, fontweight="bold")

axes[-1].set_xlabel("Environment steps (millions)")
fig.suptitle("Recent strict-1+5 experiments: core training performance", y=0.985)
fig.text(0.5, 0.945, "Faint: raw records; solid: trailing 2.5M-step mean; dots: current endpoints",
         ha="center", va="center", fontsize=8, color="0.35")
fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
           ncol=3, frameon=False)
fig.subplots_adjust(top=0.89, bottom=0.12, hspace=0.38, left=0.12, right=0.98)
fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
