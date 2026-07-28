from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
ONPOLICY = REPO_ROOT / "onpolicy"
PLOT_DATA = ONPOLICY / "scripts" / "train" / "plot_data"
RESULTS = ONPOLICY / "scripts" / "results" / "mec" / "mappo"
PUBLISHED_FIG4 = REPO_ROOT / "paper_artifacts" / "published_figures" / "fig4.png"
TAG_CURRENT = "agent0/system_performance_equivalent_full_GUs"
EPISODE_DENOMINATOR = 64 * 400

# Pixel calibration of the canonical 2964 x 1945 Fig. 4 raster. The calibration
# comes from its labelled grid ticks, not from inferred historical raw values.
X_ZERO_PX = 390.0
X_3500_PX = 2811.0
Y_500K_PX = 397.0
Y_400K_PX = 784.0
PLOT_LEFT_PX = 269.0
PLOT_RIGHT_PX = 2932.0
PLOT_TOP_PX = 32.0
PLOT_BOTTOM_PX = 1778.0


def centered_smooth(values: np.ndarray, radius: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    for index in range(values.size):
        lo = max(0, index - radius)
        hi = min(values.size, index + radius + 1)
        result[index] = values[lo:hi].mean()
    return result


def load_event_series(log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file in {log_dir}")
    accumulator = event_accumulator.EventAccumulator(
        str(event_files[-1]), size_guidance={event_accumulator.SCALARS: 0}
    )
    accumulator.Reload()
    if TAG_CURRENT not in accumulator.Tags()["scalars"]:
        raise KeyError(f"Missing tag {TAG_CURRENT!r} in {event_files[-1]}")
    events = accumulator.Scalars(TAG_CURRENT)
    episodes = np.asarray([event.step / EPISODE_DENOMINATOR for event in events], dtype=float)
    values = np.asarray([event.value for event in events], dtype=float)
    return episodes, values


def data_to_pixels(episodes: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_scale = (X_3500_PX - X_ZERO_PX) / 3500.0
    y_scale = (Y_400K_PX - Y_500K_PX) / 100000.0
    x_pixels = X_ZERO_PX + episodes * x_scale
    y_pixels = Y_500K_PX + (500000.0 - values) * y_scale
    return x_pixels, y_pixels


def write_current_csv(series: list[dict[str, object]]) -> None:
    with (OUTPUT_DIR / "plotted_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "learning_episode", "raw_value", "three_point_smoothed_value"])
        for item in series:
            for x, raw, smooth in zip(item["episodes"], item["raw"], item["smooth"]):
                writer.writerow([item["label"], float(x), float(raw), float(smooth)])


def write_summary(series: list[dict[str, object]]) -> None:
    historical = {
        "MAPPO (Fig. 4)": np.load(PLOT_DATA / "mappo.npy").astype(float).mean(axis=0),
        "DC-PPO (Fig. 4)": np.load(PLOT_DATA / "dcppo.npy").astype(float).mean(axis=0),
        "ARA (Fig. 4)": np.load(PLOT_DATA / "ara_extended.npy").astype(float).mean(axis=0),
    }
    historical_x = np.arange(1, 2 * len(next(iter(historical.values()))), 2, dtype=float)
    with (OUTPUT_DIR / "curve_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "current_series",
                "latest_learning_episode",
                "latest_raw_value",
                "recent_20_raw_mean",
                "historical_series",
                "historical_value_at_current_endpoint",
                "difference_current_recent20_minus_historical",
            ]
        )
        for item in series:
            endpoint = float(item["episodes"][-1])
            recent = float(np.mean(item["raw"][-20:]))
            latest = float(item["raw"][-1])
            for historical_label, historical_values in historical.items():
                old_at_endpoint = float(np.interp(endpoint, historical_x, historical_values))
                writer.writerow(
                    [
                        item["label"],
                        endpoint,
                        latest,
                        recent,
                        historical_label,
                        old_at_endpoint,
                        recent - old_at_endpoint,
                    ]
                )


def main() -> None:
    current_specs = [
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
    series: list[dict[str, object]] = []
    for label, log_dir, color, linestyle in current_specs:
        episodes, raw = load_event_series(log_dir)
        series.append(
            dict(
                label=label,
                episodes=episodes,
                raw=raw,
                smooth=centered_smooth(raw),
                color=color,
                linestyle=linestyle,
            )
        )

    write_current_csv(series)
    write_summary(series)

    background = mpimg.imread(PUBLISHED_FIG4)
    height, width = background.shape[:2]
    fig = plt.figure(figsize=(width / 300.0, height / 300.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(background, origin="upper")
    for item in series:
        x_pixels, y_pixels = data_to_pixels(item["episodes"], item["smooth"])
        visible = (
            (x_pixels >= PLOT_LEFT_PX)
            & (x_pixels <= PLOT_RIGHT_PX)
            & (y_pixels >= PLOT_TOP_PX)
            & (y_pixels <= PLOT_BOTTOM_PX)
        )
        ax.plot(
            x_pixels[visible],
            y_pixels[visible],
            label=item["label"],
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=2.8,
            solid_capstyle="round",
            zorder=5,
        )

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")
    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.965, 0.075),
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=8.5,
        title="Training rollouts: equivalent 60-MD metric",
        title_fontsize=8.5,
        handlelength=3.0,
    )
    legend.set_zorder(10)
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
