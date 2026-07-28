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
ONPOLICY = ROOT / "onpolicy"
RESULTS = ONPOLICY / "scripts" / "results" / "mec" / "mappo"
PLOT_DATA = ONPOLICY / "scripts" / "train" / "plot_data"
FIG4 = ROOT / "paper_artifacts" / "published_figures" / "fig4.png"
TAG = "agent0/system_performance_equivalent_full_GUs"
DENOMINATOR = 64 * 400

# Canonical 2964 x 1945 Fig. 4 tick calibration.
X0, X3500 = 390.0, 2811.0
Y500K, Y400K = 397.0, 784.0
LEFT, RIGHT, TOP, BOTTOM = 269.0, 2932.0, 32.0, 1778.0


def smooth3(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        result[i] = values[max(0, i - 1) : min(len(values), i + 2)].mean()
    return result


def load(log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    event_file = sorted(log_dir.glob("events.out.tfevents.*"))[-1]
    acc = event_accumulator.EventAccumulator(
        str(event_file), size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()
    events = acc.Scalars(TAG)
    x = np.asarray([event.step / DENOMINATOR for event in events], dtype=float)
    y = np.asarray([event.value for event in events], dtype=float)
    return x, y


def to_pixels(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = X0 + x * (X3500 - X0) / 3500.0
    py = Y500K + (500000.0 - y) * (Y400K - Y500K) / 100000.0
    return px, py


def main() -> None:
    specs = [
        (
            "MAPPO (strict 1+5, stochastic train)",
            RESULTS / "regional_dynamic_strict1p5_init110_220_330_440_mappo_gae095_seed2_50m" / "run1" / "logs",
            "#17becf",
            "-",
        ),
        (
            "DC-PPO (strict 1+5, stochastic train)",
            RESULTS / "regional_dynamic_strict1p5_init110_220_330_440_dcppo_gae095_seed2_50m" / "run1" / "logs",
            "#8c564b",
            "--",
        ),
    ]
    curves = []
    for label, log_dir, color, linestyle in specs:
        x, raw = load(log_dir)
        curves.append(
            {"label": label, "x": x, "raw": raw, "smooth": smooth3(raw), "color": color, "linestyle": linestyle}
        )

    with (OUT / "plotted_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "learning_episode", "raw_value", "three_point_smoothed_value"])
        for curve in curves:
            for x, raw, smooth in zip(curve["x"], curve["raw"], curve["smooth"]):
                writer.writerow([curve["label"], float(x), float(raw), float(smooth)])

    historical = {
        "MAPPO (Fig. 4)": np.load(PLOT_DATA / "mappo.npy").astype(float).mean(axis=0),
        "DC-PPO (Fig. 4)": np.load(PLOT_DATA / "dcppo.npy").astype(float).mean(axis=0),
        "ARA (Fig. 4)": np.load(PLOT_DATA / "ara_extended.npy").astype(float).mean(axis=0),
    }
    historical_x = np.arange(1, 2 * len(next(iter(historical.values()))), 2, dtype=float)
    with (OUT / "curve_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series",
                "points",
                "final_step",
                "final_episode",
                "final_raw",
                "last20_mean",
                "last50_mean",
                "last100_mean",
                "best20_mean",
                "best20_end_step",
                "historical_series",
                "historical_value_at_episode_1953",
            ]
        )
        for curve in curves:
            raw = curve["raw"]
            rolling20 = np.convolve(raw, np.ones(20) / 20.0, mode="valid")
            best_index = int(np.argmax(rolling20)) + 19
            for old_label, old_values in historical.items():
                writer.writerow(
                    [
                        curve["label"],
                        len(raw),
                        int(curve["x"][-1] * DENOMINATOR),
                        float(curve["x"][-1]),
                        float(raw[-1]),
                        float(raw[-20:].mean()),
                        float(raw[-50:].mean()),
                        float(raw[-100:].mean()),
                        float(rolling20.max()),
                        int(curve["x"][best_index] * DENOMINATOR),
                        old_label,
                        float(np.interp(curve["x"][-1], historical_x, old_values)),
                    ]
                )

    background = mpimg.imread(FIG4)
    height, width = background.shape[:2]
    fig = plt.figure(figsize=(width / 300.0, height / 300.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(background, origin="upper")
    for curve in curves:
        px, py = to_pixels(curve["x"], curve["smooth"])
        visible = (px >= LEFT) & (px <= RIGHT) & (py >= TOP) & (py <= BOTTOM)
        ax.plot(
            px[visible],
            py[visible],
            color=curve["color"],
            linestyle=curve["linestyle"],
            linewidth=2.8,
            solid_capstyle="round",
            label=curve["label"],
            zorder=5,
        )
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.965, 0.075),
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.5,
        title="Final training rollouts: equivalent 60-MD metric",
        title_fontsize=8.5,
        handlelength=3.0,
    )
    fig.savefig(OUT / "plot.png", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
