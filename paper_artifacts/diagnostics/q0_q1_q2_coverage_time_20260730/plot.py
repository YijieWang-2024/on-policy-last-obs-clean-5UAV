from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / "onpolicy" / "scripts" / "results" / "mec" / "mappo"

METHODS = {
    "Q0": RESULTS
    / "q0_polar_standard_curriculum_completionpriority_seed2_100m_20260729"
    / "run1"
    / "eval_ranked10_step99968000_20260730",
    "Q1": RESULTS
    / "q1_polar_spatial_curriculum_completionpriority_seed2_100m_20260729"
    / "run1"
    / "eval_ranked10_step94233600_20260730",
    "Q2": RESULTS
    / "q2_cartesian_spatial_curriculum_completionpriority_seed2_100m_20260729"
    / "run2"
    / "eval_ranked10_step90188800_20260730",
}
COLORS = {"Q0": "#0072B2", "Q1": "#E69F00", "Q2": "#009E73"}
SLOTS = 400
RADIUS_SQ = 120.0**2


def rectangle_grid(x0: float, x1: float, y0: float, y1: float, n: int) -> np.ndarray:
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


LOWER_GRID = rectangle_grid(0.0, 175.0, 0.0, 175.0, 81)
UPPER_GRID = rectangle_grid(200.0, 600.0, 200.0, 600.0, 121)


def union_fraction(uavs: np.ndarray, grid: np.ndarray) -> float:
    delta = grid[:, None, :] - uavs[None, :, :]
    return np.any(np.sum(delta * delta, axis=2) <= RADIUS_SQ, axis=1).mean()


def coverage_series(npz_path: Path) -> np.ndarray:
    positions = np.load(npz_path)["uav_positions"][1 : SLOTS + 1]
    values = np.empty(SLOTS, dtype=float)
    for index, uavs in enumerate(positions):
        lower = union_fraction(uavs, LOWER_GRID)
        upper = union_fraction(uavs, UPPER_GRID)
        values[index] = (lower + 5.0 * upper) / 6.0
    return values


def admitted_series(csv_path: Path) -> np.ndarray:
    first_seen: dict[int, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            session_id = int(row["session_id"])
            slot = int(row["slot"])
            first_seen[session_id] = min(slot, first_seen.get(session_id, slot))
    first_slots = np.fromiter(first_seen.values(), dtype=int)
    return np.array([(first_slots <= slot).sum() for slot in range(1, SLOTS + 1)])


def stable_threshold(series: np.ndarray, threshold: float, window: int = 20) -> float:
    for start in range(0, len(series) - window + 1):
        if np.all(series[start : start + window] >= threshold):
            return float(start + 1)
    return np.nan


def load_method(path: Path) -> tuple[np.ndarray, np.ndarray]:
    episode_dirs = sorted(
        directory
        for directory in path.glob("seed_*")
        if (directory / "episode_data.npz").is_file()
        and (directory / "service_assignments.csv").is_file()
        and (directory / "episode_summary.json").is_file()
    )
    if len(episode_dirs) != 10:
        raise RuntimeError(f"Expected 10 complete episodes in {path}, found {len(episode_dirs)}")
    coverage = np.stack(
        [coverage_series(directory / "episode_data.npz") for directory in episode_dirs]
    )
    admitted = np.stack(
        [admitted_series(directory / "service_assignments.csv") for directory in episode_dirs]
    )
    return coverage, admitted


def write_derived_csv(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    with (HERE / "derived_time_series.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "slot",
                "coverage_mean",
                "coverage_std",
                "cumulative_admitted_mean",
                "cumulative_admitted_std",
            ]
        )
        for method, (coverage, admitted) in data.items():
            for slot in range(SLOTS):
                writer.writerow(
                    [
                        method,
                        slot + 1,
                        coverage[:, slot].mean(),
                        coverage[:, slot].std(),
                        admitted[:, slot].mean(),
                        admitted[:, slot].std(),
                    ]
                )


def main() -> None:
    data = {method: load_method(path) for method, path in METHODS.items()}
    write_derived_csv(data)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    slots = np.arange(1, SLOTS + 1)

    for method, (coverage, admitted) in data.items():
        color = COLORS[method]
        coverage_mean = coverage.mean(axis=0) * 100.0
        coverage_std = coverage.std(axis=0) * 100.0
        admitted_mean = admitted.mean(axis=0)
        admitted_std = admitted.std(axis=0)

        axes[0].plot(slots, coverage_mean, color=color, linewidth=1.6, label=method)
        axes[0].fill_between(
            slots,
            coverage_mean - coverage_std,
            coverage_mean + coverage_std,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        axes[1].plot(slots, admitted_mean, color=color, linewidth=1.6, label=method)
        axes[1].fill_between(
            slots,
            admitted_mean - admitted_std,
            admitted_mean + admitted_std,
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    q2_coverage = data["Q2"][0]
    q2_stable_slots = np.array([stable_threshold(row, 0.90) for row in q2_coverage])
    q2_mean_stable = float(np.nanmean(q2_stable_slots))
    axes[0].axhline(90.0, color="#555555", linestyle=":", linewidth=1.0)
    axes[0].axvline(q2_mean_stable, color=COLORS["Q2"], linestyle="--", linewidth=1.0)
    axes[0].annotate(
        f"Q2 stable >=90%\nmean slot {q2_mean_stable:.1f}",
        xy=(q2_mean_stable, 90.0),
        xytext=(82, 68),
        arrowprops={"arrowstyle": "->", "color": COLORS["Q2"], "lw": 0.8},
        color=COLORS["Q2"],
        fontsize=7.5,
    )

    axes[1].plot(slots, 6 * slots, color="#777777", linestyle=":", linewidth=1.0, label="Candidates")

    axes[0].set_title("(a) Formation of useful coverage")
    axes[0].set_xlabel("Slot")
    axes[0].set_ylabel("Demand-weighted union coverage (%)")
    axes[0].set_xlim(1, SLOTS)
    axes[0].set_ylim(20, 100)
    axes[0].set_yticks([20, 40, 60, 80, 90, 100])

    axes[1].set_title("(b) Resulting MD admission")
    axes[1].set_xlabel("Slot")
    axes[1].set_ylabel("Cumulative admitted MD sessions")
    axes[1].set_xlim(1, SLOTS)
    axes[1].set_ylim(0, 2450)

    for axis in axes:
        axis.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
