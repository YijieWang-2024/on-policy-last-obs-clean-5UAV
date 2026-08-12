from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "training_curves.csv"
NOISE_DATA = HERE.parent / "data" / "radius_noise_magnitude_common_budget.csv"
ROLLING_POINTS = 20
TAIL_POINTS = 100
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def magnitude(value, _position=None):
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def status_text(status):
    return {
        "complete": "complete",
        "partial_inactive": "inactive partial",
        "near_complete_inactive": "inactive near-complete",
    }.get(status, status.replace("_", " "))


def main():
    all_curves = pd.read_csv(DATA)
    frame = all_curves.copy()
    noise_frame = pd.read_csv(NOISE_DATA).sort_values("radius")
    frame = frame.loc[
        (frame["comparison"] == "radius")
        & frame["protocol_valid"].astype(bool)
        & frame["use_in_figure"].astype(bool)
    ].copy()
    series_meta = (
        frame.groupby("series_id", as_index=False)
        .agg(
            label=("label", "first"),
            radius=("radius", "first"),
            status=("status", "first"),
            last_step=("step", "max"),
        )
        .sort_values("radius")
    )
    common_step = int(series_meta["last_step"].min())
    baseline = all_curves.loc[
        (all_curves["series_id"] == "baseline")
        & (all_curves["step"] <= common_step),
        "value",
    ].tail(TAIL_POINTS)
    if len(baseline) != TAIL_POINTS:
        raise ValueError("No-noise R0 baseline lacks the matched tail window")
    baseline_value = float(baseline.mean())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titlepad": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig = plt.figure(figsize=(7.4, 7.0))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[2.15, 1.0], width_ratios=[1.15, 1.0],
        hspace=0.38, wspace=0.34,
    )
    ax_curve = fig.add_subplot(grid[0, :])
    ax_bar = fig.add_subplot(grid[1, 0])
    ax_noise = fig.add_subplot(grid[1, 1])
    styles = ["-", "--", "-.", ":"]
    matched_values = []

    for index, row in enumerate(series_meta.itertuples()):
        data = frame.loc[frame["series_id"] == row.series_id].sort_values("step")
        smooth = data["value"].rolling(ROLLING_POINTS, min_periods=1).mean()
        label = f"{row.label} | {status_text(row.status)} @ {row.last_step / 1e6:.1f}M"
        ax_curve.plot(
            data["step"] / 1e6,
            smooth,
            color=OKABE_ITO[index],
            linestyle=styles[index],
            linewidth=1.8,
            label=label,
        )
        ax_curve.scatter(
            data["step"].iloc[-1] / 1e6,
            smooth.iloc[-1],
            color=OKABE_ITO[index],
            s=22,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        shared = data.loc[data["step"] <= common_step, "value"].tail(TAIL_POINTS)
        matched_values.append(float(shared.mean()))

    ax_curve.set_title("End-to-end radius comparison with noise scale 3")
    ax_curve.set_xlabel("Environment steps (million)")
    ax_curve.set_ylabel("True60 system performance")
    ax_curve.set_xlim(0, series_meta["last_step"].max() / 1e6 * 1.015)
    ax_curve.yaxis.set_major_formatter(FuncFormatter(magnitude))
    ax_curve.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.75)
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.tick_params(direction="out", length=4)

    positions = np.arange(len(series_meta))
    bars = ax_bar.bar(
        positions,
        matched_values,
        color=OKABE_ITO,
        edgecolor="none",
        width=0.68,
    )
    ax_bar.bar_label(
        bars,
        labels=[f"{value / 1e3:.1f}k" for value in matched_values],
        padding=3,
        fontsize=8.5,
    )
    ax_bar.set_title(
        f"Matched budget: tail{TAIL_POINTS} at {common_step / 1e6:.2f}M steps\n"
        f"Dashed reference: no-noise R0 = {baseline_value / 1e3:.1f}k"
    )
    ax_bar.set_xlabel("Communication radius (m)")
    ax_bar.set_ylabel("Matched-budget True60")
    ax_bar.set_xticks(positions, [f"R{int(value)}" for value in series_meta["radius"]])
    ax_bar.axhline(
        baseline_value,
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=1.1,
        zorder=0,
    )
    ax_bar.set_xlim(-0.55, positions[-1] + 0.55)
    ax_bar.set_ylim(0, max(matched_values) * 1.16)
    ax_bar.yaxis.set_major_formatter(FuncFormatter(magnitude))
    ax_bar.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.75)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(direction="out", length=4)

    if noise_frame["common_step"].nunique() != 1 or int(
        noise_frame["common_step"].iloc[0]
    ) != common_step:
        raise ValueError("Noise-magnitude summary does not match the curve common step")
    if noise_frame["radius"].tolist() != series_meta["radius"].tolist():
        raise ValueError("Noise-magnitude radii do not match the plotted curve radii")
    noise_values = noise_frame["noise_magnitude_mean"].to_numpy()
    noise_bars = ax_noise.bar(
        positions,
        noise_values,
        color=OKABE_ITO,
        edgecolor="none",
        width=0.68,
    )
    ax_noise.bar_label(
        noise_bars,
        labels=["1.0" if value == 1.0 else f"{value:.2e}" for value in noise_values],
        padding=3,
        fontsize=7.7,
    )
    ax_noise.set_title(f"Effective noise magnitude $m_i$\n(tail{TAIL_POINTS}, same budget)")
    ax_noise.set_xlabel("Communication radius (m)")
    ax_noise.set_ylabel("Mean $m_i$ (log scale)")
    ax_noise.set_xticks(positions, [f"R{int(value)}" for value in series_meta["radius"]])
    ax_noise.set_yscale("log")
    ax_noise.set_ylim(1e-7, 3.0)
    ax_noise.grid(axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.65)
    ax_noise.set_axisbelow(True)
    ax_noise.spines[["top", "right"]].set_visible(False)
    ax_noise.tick_params(direction="out", length=4)

    handles, labels = ax_curve.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.96, bottom=0.16)
    fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    main()
