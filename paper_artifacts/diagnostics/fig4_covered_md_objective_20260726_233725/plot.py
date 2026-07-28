from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RESULTS = ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
OLD = RESULTS / "check"
DENOMINATOR = 64 * 400
N_AGENTS = 5


def smooth3(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        result[i] = values[max(0, i - 1) : min(len(values), i + 2)].mean()
    return result


def read_event(event_file: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    acc = event_accumulator.EventAccumulator(
        str(event_file), size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()
    events = acc.Scalars(tag)
    return (
        np.asarray([event.step / DENOMINATOR for event in events], dtype=float),
        np.asarray([event.value for event in events], dtype=float),
    )


def load_old_sum(run: str, metric: str = "system_performance_individual") -> tuple[np.ndarray, np.ndarray]:
    agents = []
    for agent in range(N_AGENTS):
        metric_dir = OLD / run / "logs" / f"agent{agent}" / metric / f"agent{agent}" / metric
        event_file = sorted(metric_dir.glob("events.out.tfevents.*"))[-1]
        agents.append(read_event(event_file, f"agent{agent}/{metric}"))
    count = min(len(values) for _, values in agents)
    x = agents[0][0][:count]
    summed = np.sum([values[:count] for _, values in agents], axis=0)
    return x, summed


def load_current_metric(run_dir: Path, metric: str, agent: int) -> tuple[np.ndarray, np.ndarray]:
    event_file = sorted((run_dir / "logs").glob("events.out.tfevents.*"))[-1]
    return read_event(event_file, f"agent{agent}/{metric}")


def load_current_sum(run_dir: Path, metric: str = "system_performance_individual") -> tuple[np.ndarray, np.ndarray]:
    agents = [load_current_metric(run_dir, metric, agent) for agent in range(N_AGENTS)]
    count = min(len(values) for _, values in agents)
    return agents[0][0][:count], np.sum([values[:count] for _, values in agents], axis=0)


def historical_group(runs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    for run in runs:
        x, values = load_old_sum(run)
        mask = x <= 3499
        curves.append((x[mask], smooth3(values[mask])))
    start = max(x[0] for x, _ in curves)
    end = min(x[-1] for x, _ in curves)
    count = min(len(x) for x, _ in curves)
    common_x = np.linspace(start, end, count)
    aligned = np.vstack([np.interp(common_x, x, values) for x, values in curves])
    return common_x, aligned.mean(axis=0), aligned.std(axis=0, ddof=0)


def magnitude(value: float, _position: float) -> str:
    return f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"


def main() -> None:
    current_dirs = {
        "MAPPO (strict 1+5)": RESULTS
        / "regional_dynamic_strict1p5_init110_220_330_440_mappo_gae095_seed2_50m"
        / "run1",
        "DC-PPO (strict 1+5)": RESULTS
        / "regional_dynamic_strict1p5_init110_220_330_440_dcppo_gae095_seed2_50m"
        / "run1",
    }
    hx_m, hm_m, hs_m = historical_group(["run313", "run321", "run322"])
    hx_d, hm_d, hs_d = historical_group(["run301", "run304", "run30901"])
    mx, my_raw = load_current_sum(current_dirs["MAPPO (strict 1+5)"])
    dx, dy_raw = load_current_sum(current_dirs["DC-PPO (strict 1+5)"])

    series = [
        {"label": "MAPPO (fixed 60, 3 seeds)", "kind": "historical", "x": hx_m, "mean": hm_m, "std": hs_m, "color": "#1f77b4", "style": "-"},
        {"label": "DC-PPO (fixed 60, 3 seeds)", "kind": "historical", "x": hx_d, "mean": hm_d, "std": hs_d, "color": "#ff7f0e", "style": "--"},
        {"label": "MAPPO (strict 1+5, seed2)", "kind": "current", "x": mx, "mean": smooth3(my_raw), "raw": my_raw, "std": None, "color": "#17becf", "style": "-"},
        {"label": "DC-PPO (strict 1+5, seed2)", "kind": "current", "x": dx, "mean": smooth3(dy_raw), "raw": dy_raw, "std": None, "color": "#8c564b", "style": "--"},
    ]

    with (OUT / "plotted_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "source_kind", "learning_episode", "mean", "std"])
        for item in series:
            for i, (x, mean) in enumerate(zip(item["x"], item["mean"])):
                writer.writerow([item["label"], item["kind"], float(x), float(mean), "" if item["std"] is None else float(item["std"][i])])

    with (OUT / "curve_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "endpoint_episode", "value_at_1953", "last20_mean_if_current", "last50_mean_if_current"])
        for item in series:
            value = float(np.interp(1953.0, item["x"], item["mean"]))
            if item["kind"] == "current":
                writer.writerow([item["label"], float(item["x"][-1]), value, float(item["raw"][-20:].mean()), float(item["raw"][-50:].mean())])
            else:
                writer.writerow([item["label"], float(item["x"][-1]), value, "", ""])

    with (OUT / "metric_identity_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "max_abs_sum_individual_minus_true_all", "mean_sum_individual_minus_true_all", "max_abs_sum_individual_minus_sum_cumulative"])
        for label, run_dir in current_dirs.items():
            x, summed_individual = load_current_sum(run_dir, "system_performance_individual")
            _, true_all = load_current_metric(run_dir, "system_performance_true_all_GUs", 0)
            _, summed_cumulative = load_current_sum(run_dir, "cumulative_individual_reward")
            count = min(len(summed_individual), len(true_all), len(summed_cumulative))
            delta_true = summed_individual[:count] - true_all[:count]
            delta_cumulative = summed_individual[:count] - summed_cumulative[:count]
            writer.writerow([label, float(np.max(np.abs(delta_true))), float(np.mean(delta_true)), float(np.max(np.abs(delta_cumulative)))])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for item in series:
        current = item["kind"] == "current"
        ax.plot(item["x"], item["mean"], color=item["color"], linestyle=item["style"], linewidth=2.6 if current else 2.0, label=item["label"], zorder=4 if current else 2)
        if item["std"] is not None:
            ax.fill_between(item["x"], item["mean"] - item["std"], item["mean"] + item["std"], color=item["color"], alpha=0.14, linewidth=0, zorder=1)
    ax.set_xlim(0, 3500)
    ax.set_xlabel("Learning episodes")
    ax.set_ylabel("Covered-MD system performance")
    ax.yaxis.set_major_formatter(FuncFormatter(magnitude))
    ax.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
