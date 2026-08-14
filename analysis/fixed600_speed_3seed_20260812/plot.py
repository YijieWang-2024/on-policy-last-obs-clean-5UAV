from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRIC = "agent0/system_performance_true_all_GUs"
TARGET_STEPS = 60_000_000
COMPLETE_THRESHOLD = 59_900_000

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
LOCAL_EXPERIMENTS = {
    (10, 2): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax10_seed2_60m_20260812",
    (20, 2): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax20_seed2_60m_20260812",
    (30, 2): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax30_seed2_60m_20260812",
    (10, 42): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax10_seed42_60m_20260811",
    (20, 42): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax20_seed42_60m_20260811",
    (30, 42): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax30_seed42_60m_20260811",
}

REMOTE_EXPERIMENTS = {
    (10, 32): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax10_seed32_60m_20260811_retry1",
    (20, 32): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax20_seed32_60m_20260811_retry1",
    (30, 32): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax30_seed32_60m_20260811_retry2",
    (40, 2): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed2_60m_20260812",
    (40, 32): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed32_60m_20260812",
    (40, 42): "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed42_60m_20260812",
}

EXPECTED_PROTOCOL = {
    "num_env_steps": 60_000_000,
    "episode_length": 400,
    "n_rollout_threads": 64,
    "n_GUs": 60,
    "x_min_gu": 0,
    "x_max_gu": 600,
    "y_min_gu": 0,
    "y_max_gu": 600,
    "hotspot_layout_mode": "episode_template4_600_200",
    "episode_layout_context": True,
    "episode_layout_context_units": "meters_v2",
    "md_arrivals_min": 5,
    "md_arrivals_max": 5,
    "md_arrivals_per_region": [1, 4],
    "md_lifetime_min": 12,
    "md_lifetime_max": 12,
    "mean_velocity": 3.0,
    "neighbor_distance": 520.0,
    "neighbor_R": 520,
    "noise_scale": 3.0,
    "advantage_mode": "per_agent_noise",
    "actor_message_mode": "task_summary",
    "actor_message_pool": "receiver_gated_sum",
    "actor_message_contract": "absolute_raw_v2",
    "ob_norm": True,
    "spatial_flight_actor": True,
}

COLORS = {
    10: "#56B4E9",  # Okabe-Ito sky blue
    20: "#0072B2",  # Okabe-Ito blue
    30: "#E69F00",  # Okabe-Ito orange
    40: "#CC79A7",  # Okabe-Ito reddish purple
}


def local_paths(experiment: str) -> tuple[Path, Path]:
    run = RESULTS / experiment / "run1"
    events = sorted((run / "logs").glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError(f"Expected one event file in {run / 'logs'}, found {len(events)}")
    return events[0], run / "args.json"


def load_scalar(event_path: Path) -> dict:
    if not event_path.is_file():
        raise FileNotFoundError(event_path)
    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if METRIC not in tags:
        raise KeyError(f"{METRIC!r} missing from {event_path}; available={tags}")

    by_step = {}
    for event in acc.Scalars(METRIC):
        by_step[int(event.step)] = (float(event.value), float(event.wall_time))
    steps = np.asarray(sorted(by_step), dtype=np.int64)
    values = np.asarray([by_step[int(step)][0] for step in steps], dtype=np.float64)
    wall_times = np.asarray([by_step[int(step)][1] for step in steps], dtype=np.float64)
    if len(steps) < 2:
        raise RuntimeError(f"Too few scalar points in {event_path}")
    return {"steps": steps, "values": values, "wall_times": wall_times}


def recent_speed(curve: dict) -> float:
    steps = curve["steps"]
    wall_times = curve["wall_times"]
    points = min(80, len(steps) - 1)
    delta_t = wall_times[-1] - wall_times[-1 - points]
    return float((steps[-1] - steps[-1 - points]) / delta_t) if delta_t > 0 else float("nan")


def fmt_step(step: float) -> str:
    return f"{step / 1_000_000:.2f}M"


def fmt_datetime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def magnitude_format(value, _position):
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def normalize_arg(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def availability_segments(grid: np.ndarray, counts: np.ndarray) -> list[dict]:
    segments = []
    start = 0
    for index in range(1, len(grid) + 1):
        if index == len(grid) or counts[index] != counts[start]:
            segments.append(
                {
                    "start_step": int(grid[start]),
                    "end_step": int(grid[index - 1]),
                    "seed_count": int(counts[start]),
                }
            )
            start = index
    return segments


def main() -> None:
    source_paths = {}
    for key, experiment in LOCAL_EXPERIMENTS.items():
        source_paths[key] = (*local_paths(experiment), "local", experiment)
    for key, experiment in REMOTE_EXPERIMENTS.items():
        event_path, args_path = local_paths(experiment)
        source_paths[key] = (
            event_path,
            args_path,
            "remote3090-archived",
            experiment,
        )

    expected_keys = {(speed, seed) for speed in (10, 20, 30, 40) for seed in (2, 32, 42)}
    if set(source_paths) != expected_keys:
        raise RuntimeError(f"Run map mismatch: {set(source_paths) ^ expected_keys}")

    curves = {}
    run_summaries = []
    protocol_errors = []
    for speed, seed in sorted(source_paths):
        event_path, args_path, host, source_name = source_paths[(speed, seed)]
        args = json.loads(args_path.read_text(encoding="utf-8"))
        for field, expected in EXPECTED_PROTOCOL.items():
            actual = normalize_arg(args.get(field))
            if actual != expected:
                protocol_errors.append(
                    f"v_max={speed}, seed={seed}: {field}={actual!r}, expected {expected!r}"
                )
        if int(args.get("seed")) != seed:
            protocol_errors.append(f"v_max={speed}, seed={seed}: args seed={args.get('seed')!r}")
        if float(args.get("v_max")) != float(speed):
            protocol_errors.append(f"v_max={speed}, seed={seed}: args v_max={args.get('v_max')!r}")

        curve = load_scalar(event_path)
        curves[(speed, seed)] = curve
        last_step = int(curve["steps"][-1])
        speed_sps = recent_speed(curve)
        complete = last_step >= COMPLETE_THRESHOLD
        remaining = max(TARGET_STEPS - last_step, 0)
        eta_seconds = 0.0 if complete else remaining / speed_sps
        predicted_finish = None if complete else curve["wall_times"][-1] + eta_seconds
        run_summaries.append(
            {
                "v_max": speed,
                "seed": seed,
                "host": host,
                "source": source_name,
                "event_path": str(event_path),
                "args_path": str(args_path),
                "points": int(len(curve["steps"])),
                "last_step": last_step,
                "progress_percent": 100.0 * min(last_step / TARGET_STEPS, 1.0),
                "last_value": float(curve["values"][-1]),
                "recent_steps_per_second": speed_sps,
                "last_event_time": fmt_datetime(float(curve["wall_times"][-1])),
                "curve_complete": complete,
                "eta_seconds_from_last_event": eta_seconds,
                "predicted_finish_from_last_event": (
                    None if predicted_finish is None else fmt_datetime(predicted_finish)
                ),
            }
        )

    if protocol_errors:
        raise AssertionError("Protocol mismatch:\n" + "\n".join(protocol_errors))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    aggregate_summaries = []

    for speed in (10, 20, 30, 40):
        speed_curves = [curves[(speed, seed)] for seed in (2, 32, 42)]
        grid = np.unique(np.concatenate([curve["steps"] for curve in speed_curves]))
        aligned = np.full((3, len(grid)), np.nan, dtype=np.float64)

        for row, curve in enumerate(speed_curves):
            in_range = (grid >= curve["steps"][0]) & (grid <= curve["steps"][-1])
            aligned[row, in_range] = np.interp(
                grid[in_range], curve["steps"], curve["values"]
            )

        counts = np.sum(np.isfinite(aligned), axis=0)
        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0, ddof=0)
        multi_seed = counts >= 2
        total_seed_count = len(speed_curves)
        terminal_count = int(counts[-1])
        terminal_mean = float(mean[-1])

        label = (
            f"v_max={speed} m/s: {terminal_mean:,.0f} at "
            f"{fmt_step(grid[-1])} (available={terminal_count}/{total_seed_count})"
        )
        color = COLORS[speed]
        ax.plot(grid, mean, color=color, linewidth=1.45, label=label, zorder=3)
        ax.fill_between(
            grid,
            mean - std,
            mean + std,
            where=multi_seed,
            interpolate=False,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        aggregate_summaries.append(
            {
                "v_max": speed,
                "seed_ids": [2, 32, 42],
                "total_seed_count": total_seed_count,
                "terminal_step": int(grid[-1]),
                "terminal_seed_count": terminal_count,
                "terminal_mean": terminal_mean,
                "terminal_std": float(std[-1]) if terminal_count >= 2 else None,
                "availability_segments": availability_segments(grid, counts),
            }
        )

    ax.set_title("Fixed600, MD lifetime=12, noise=3.0, R520: UAV speed comparison")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("System performance (all GUs)")
    ax.set_xlim(0, TARGET_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(magnitude_format))
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.legend(loc="lower right", frameon=True, framealpha=0.90, labelspacing=0.4)
    fig.text(
        0.01,
        0.01,
        "Line: mean of seeds available at each step; band: +/-1 population SD when n>=2; no smoothing.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    output_path = HERE / "plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric": METRIC,
        "target_steps": TARGET_STEPS,
        "aggregation": (
            "At each union-grid step, interpolate only within each seed's observed range; "
            "plot the mean over available seeds and population std only when >=2 seeds are available. "
            "Never extrapolate and never smooth."
        ),
        "protocol_verified": True,
        "expected_protocol": EXPECTED_PROTOCOL,
        "runs": run_summaries,
        "aggregates": aggregate_summaries,
        "output": str(output_path),
    }
    (HERE / "progress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
