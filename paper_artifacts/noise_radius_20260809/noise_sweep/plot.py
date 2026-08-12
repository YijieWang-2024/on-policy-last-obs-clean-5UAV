from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "training_curves.csv"
ROLLING_POINTS = 20
TAIL_POINTS = 25


def magnitude(value, _position=None):
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def status_text(status):
    return {
        "complete": "complete",
        "near_complete_inactive": "inactive near-complete",
        "intentionally_stopped": "stopped",
    }.get(status, status.replace("_", " "))


def main():
    frame = pd.read_csv(DATA)
    frame = frame.loc[
        (frame["comparison"] == "noise")
        & frame["protocol_valid"].astype(bool)
        & frame["use_in_figure"].astype(bool)
    ].copy()
    series_meta = (
        frame.groupby("series_id", as_index=False)
        .agg(
            label=("label", "first"),
            noise_scale=("noise_scale", "first"),
            status=("status", "first"),
            last_step=("step", "max"),
        )
        .sort_values("noise_scale")
    )
    common_step = int(series_meta["last_step"].min())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titlepad": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
        }
    )
    fig = plt.figure(figsize=(8.2, 7.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.25, 1.0], hspace=0.34)
    ax_curve = fig.add_subplot(grid[0])
    ax_common = fig.add_subplot(grid[1])

    nonzero = series_meta.loc[series_meta["noise_scale"] > 0]
    cmap = plt.get_cmap("viridis")
    colors = {
        row.series_id: cmap(0.10 + 0.82 * index / max(len(nonzero) - 1, 1))
        for index, row in enumerate(nonzero.itertuples())
    }
    colors["baseline"] = "#000000"
    styles = {
        "complete": "-",
        "near_complete_inactive": "--",
        "intentionally_stopped": ":",
    }
    common_values = []
    for row in series_meta.itertuples():
        data = frame.loc[frame["series_id"] == row.series_id].sort_values("step")
        smooth = data["value"].rolling(ROLLING_POINTS, min_periods=1).mean()
        legend_label = (
            f"{row.label} | {status_text(row.status)} @ {row.last_step / 1e6:.1f}M"
        )
        ax_curve.plot(
            data["step"] / 1e6,
            smooth,
            color=colors[row.series_id],
            linestyle=styles.get(row.status, "-."),
            linewidth=1.65,
            label=legend_label,
        )
        ax_curve.scatter(
            data["step"].iloc[-1] / 1e6,
            smooth.iloc[-1],
            color=colors[row.series_id],
            s=18,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        shared = data.loc[data["step"] <= common_step, "value"].tail(TAIL_POINTS)
        common_values.append(float(shared.mean()))

    ax_curve.set_title("Fixed-layout per-UAV advantage-noise sweep")
    ax_curve.set_xlabel("Environment steps (million)")
    ax_curve.set_ylabel("True60 system performance")
    ax_curve.set_xlim(0, series_meta["last_step"].max() / 1e6 * 1.015)
    ax_curve.yaxis.set_major_formatter(FuncFormatter(magnitude))
    ax_curve.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.75)
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.tick_params(direction="out", length=4)

    positions = np.arange(len(series_meta))
    ax_common.plot(positions, common_values, color="#666666", linewidth=0.8, zorder=1)
    ax_common.scatter(
        positions,
        common_values,
        c=[colors[series_id] for series_id in series_meta["series_id"]],
        s=38,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )
    ax_common.set_title(
        f"Matched early budget: tail{TAIL_POINTS} at {common_step / 1e6:.2f}M steps"
    )
    ax_common.set_xlabel("Noise scale")
    ax_common.set_ylabel("Matched-budget True60")
    ax_common.set_xticks(
        positions,
        [f"{value:g}" for value in series_meta["noise_scale"]],
    )
    ax_common.yaxis.set_major_formatter(FuncFormatter(magnitude))
    ax_common.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.75)
    ax_common.spines[["top", "right"]].set_visible(False)
    ax_common.tick_params(direction="out", length=4)

    handles, labels = ax_curve.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.45,
        labelspacing=0.45,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.22)
    fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    main()
