from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
Q2_LOGS = (
    ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
    / "q2_cartesian_spatial_curriculum_completionpriority_seed2_100m_20260729"
    / "run2" / "logs"
)
PLOT_DATA = ROOT / "onpolicy" / "scripts" / "train" / "plot_data"
FIG4 = ROOT / "paper_artifacts" / "published_figures" / "fig4.png"
TAG = "agent0/system_performance_true_all_GUs"
DENOMINATOR = 64 * 400

# Pixel calibration of the canonical 2964 x 1945 Figure 4 raster.
X0, X3500 = 390.0, 2811.0
Y500K, Y400K = 397.0, 784.0
TOP, BOTTOM = 32.0, 1778.0
PX_PER_EPISODE = (X3500 - X0) / 3500.0
X4000 = X0 + 4000 * PX_PER_EPISODE


def smooth3(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        values[max(0, i - 1): min(len(values), i + 2)].mean()
        for i in range(len(values))
    ])


def load_q2() -> tuple[np.ndarray, np.ndarray]:
    event_file = sorted(Q2_LOGS.glob("events.out.tfevents.*"))[-1]
    acc = event_accumulator.EventAccumulator(
        str(event_file), size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()
    events = acc.Scalars(TAG)
    return (
        np.asarray([event.step for event in events], dtype=np.int64),
        np.asarray([event.value for event in events], dtype=float),
    )


def to_pixels(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = X0 + x * PX_PER_EPISODE
    py = Y500K + (500000.0 - y) * (Y400K - Y500K) / 100000.0
    return px, py


def main() -> None:
    steps, raw = load_q2()
    x = steps / DENOMINATOR
    smoothed = smooth3(raw)
    assert len(raw) == 1953 and steps[-1] == 99_968_000

    with (OUT / "plotted_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "series", "environment_steps", "learning_episode",
            "raw_value", "three_point_smoothed_value",
        ])
        for row in zip(steps, x, raw, smoothed):
            writer.writerow(["Q2", *row])

    historical = {
        "MAPPO": np.load(PLOT_DATA / "mappo.npy").astype(float).mean(axis=0),
        "DC-PPO": np.load(PLOT_DATA / "dcppo.npy").astype(float).mean(axis=0),
        "ARA": np.load(PLOT_DATA / "ara_extended.npy").astype(float).mean(axis=0),
    }
    historical_x = np.arange(1, 2 * len(next(iter(historical.values()))), 2, dtype=float)
    rolling20 = np.convolve(raw, np.ones(20) / 20.0, mode="valid")
    q2_at_3500 = float(np.interp(3500.0, x, smoothed))
    with (OUT / "curve_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "q2_points", "q2_final_step", "q2_final_episode", "q2_final_raw",
            "q2_last20_mean", "q2_last50_mean", "q2_best20_mean",
            "q2_smoothed_at_episode_3500", "historical_series",
            "historical_value_at_episode_3500", "q2_minus_historical_at_3500",
        ])
        for label, values in historical.items():
            old = float(np.interp(3500.0, historical_x, values))
            writer.writerow([
                len(raw), int(steps[-1]), float(x[-1]), float(raw[-1]),
                float(raw[-20:].mean()), float(raw[-50:].mean()),
                float(rolling20.max()), q2_at_3500, label, old, q2_at_3500 - old,
            ])

    background = mpimg.imread(FIG4)
    height, width = background.shape[:2]
    new_width = int(np.ceil(X4000 + 150))
    canvas = np.ones((height, new_width, background.shape[2]), dtype=background.dtype)
    canvas[:, :width] = background

    fig = plt.figure(figsize=(new_width / 300.0, height / 300.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, origin="upper")

    for y_value in (200000, 300000, 400000, 500000):
        _, py = to_pixels(np.asarray([0.0]), np.asarray([y_value]))
        ax.plot([X3500, X4000], [py[0], py[0]], color="#d0d0d0",
                linestyle="--", linewidth=0.8, zorder=1)
    ax.plot([X4000, X4000], [TOP, BOTTOM], color="#d0d0d0",
            linestyle="--", linewidth=0.8, zorder=1)
    ax.plot([X3500, X3500], [TOP, BOTTOM], color="#555555",
            linestyle=":", linewidth=1.1, zorder=4)
    ax.text(X3500 + 8, 82, "Historical Fig. 4 ends", fontsize=8,
            color="#444444", rotation=90, va="top")

    px, py = to_pixels(x, smoothed)
    visible = (px >= X0) & (px <= X4000) & (py >= TOP) & (py <= BOTTOM)
    ax.plot(px[visible], py[visible], color="#111111", linewidth=3.0,
            solid_capstyle="round",
            label="Q2: strict 1+5, true covered-MD objective", zorder=6)

    ax.text(X4000, 1838, "4000", ha="center", va="top", fontsize=10, color="#2b2b2b")
    ax.plot([X4000, X4000], [BOTTOM, BOTTOM + 12], color="#333333", linewidth=1)
    ax.set_xlim(-0.5, new_width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")
    ax.legend(
        loc="lower right", bbox_to_anchor=(0.975, 0.075),
        frameon=True, framealpha=0.94, facecolor="white", edgecolor="#cccccc",
        fontsize=8, title="Different environment; numerical scale only",
        title_fontsize=8, handlelength=3,
    )
    fig.savefig(OUT / "plot.png", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
