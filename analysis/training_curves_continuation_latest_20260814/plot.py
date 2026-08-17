"""Publication-style plot for total-step initial+continuation curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
COLORS = {20: "#0072B2", 30: "#E69F00", 40: "#009E73"}


def moving_average(values: np.ndarray, window: int = 21) -> np.ndarray:
    if len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def magnitude_millions(value, _position):
    return f"{value:g}"


def plot(data_path: Path, output_path: Path) -> None:
    data = pd.read_csv(data_path)
    data["total_step_m"] = data["total_step"] / 1e6
    data["v_max"] = data["v_max"].astype(int)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for vmax in (20, 30, 40):
        color = COLORS[vmax]
        for phase, linestyle in (("initial", "-"), ("continuation", "--")):
            part = data[(data["v_max"] == vmax) & (data["phase"] == phase)].sort_values("total_step")
            if part.empty:
                continue
            x = part["total_step_m"].to_numpy(dtype=float)
            y = part["performance"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, linewidth=0.6, alpha=0.16, marker=".", markersize=1.8,
                    linestyle="None", label="_nolegend_")
            ax.plot(x, moving_average(y), color=color, linewidth=1.65,
                    linestyle=linestyle, label=f"$v_{{\\max}}$={vmax} | {phase}")

        join = float(data.loc[data["v_max"] == vmax, "join_step"].iloc[0]) / 1e6
        ax.axvline(join, color=color, linewidth=0.75, linestyle=":", alpha=0.35, label="_nolegend_")

    ax.set_title("System performance during initial training and 40M-step continuation")
    ax.set_xlabel("Total training steps (million)")
    ax.set_ylabel("system_performance_true_all_GUs")
    ax.xaxis.set_major_formatter(FuncFormatter(magnitude_millions))
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4)

    series_handles = [
        Line2D([0], [0], color=COLORS[v], linewidth=1.8, label=f"$v_{{\\max}}$={v}")
        for v in (20, 30, 40)
    ]
    phase_handles = [
        Line2D([0], [0], color="#444444", linewidth=1.6, linestyle="-", label="initial run"),
        Line2D([0], [0], color="#444444", linewidth=1.6, linestyle="--", label="40M continuation"),
    ]
    ax.legend(handles=series_handles + phase_handles, loc="lower right", frameon=True,
              framealpha=0.92, facecolor="white", edgecolor="#BBBBBB", ncol=2,
              borderpad=0.45, labelspacing=0.35, handletextpad=0.55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE / "data" / "curves.csv")
    parser.add_argument("--output", type=Path, default=HERE / "plot.png")
    args = parser.parse_args()
    plot(args.data, args.output)


if __name__ == "__main__":
    main()
