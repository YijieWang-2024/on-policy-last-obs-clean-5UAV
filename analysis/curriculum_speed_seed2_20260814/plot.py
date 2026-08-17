from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RESULT_ROOT = REPO_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
TARGET_STEPS = 60_000_000
METRIC = "agent0/system_performance_true_all_GUs"

EXPERIMENTS = {
    10: "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_uavcurr_p0p7_10m_25m_vmax10_seed2_60m_20260813",
    20: "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_uavcurr_p0p7_10m_25m_vmax20_seed2_60m_20260813",
    30: "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_uavcurr_p0p7_10m_25m_vmax30_seed2_60m_20260813",
}
COLORS = {10: "#56B4E9", 20: "#0072B2", 30: "#E69F00"}


def local_datetime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def latest_run(experiment: str) -> tuple[Path, Path]:
    base = RESULT_ROOT / experiment
    candidates = []
    for run in sorted(base.glob("run*")):
        events = sorted((run / "logs").glob("events.out.tfevents.*"))
        if events:
            candidates.append((max(event.stat().st_mtime for event in events), run, events[-1]))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found under {base}")
    _, run, event = max(candidates, key=lambda item: (item[0], item[1].name))
    return run, event


def load_curve(event_path: Path) -> dict[str, np.ndarray]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"{METRIC!r} missing from {event_path}")
    by_step = {}
    for event in accumulator.Scalars(METRIC):
        by_step[int(event.step)] = event
    events = [by_step[step] for step in sorted(by_step)]
    return {
        "steps": np.asarray([event.step for event in events], dtype=np.int64),
        "values": np.asarray([event.value for event in events], dtype=np.float64),
        "wall_times": np.asarray([event.wall_time for event in events], dtype=np.float64),
    }


def summarize(speed: int, experiment: str, run: Path, event: Path, curve: dict) -> dict:
    steps = curve["steps"]
    values = curve["values"]
    wall_times = curve["wall_times"]
    points = min(80, len(steps) - 1)
    delta_t = wall_times[-1] - wall_times[-1 - points] if points else 0.0
    delta_steps = steps[-1] - steps[-1 - points] if points else 0
    steps_per_second = float(delta_steps / delta_t) if delta_t > 0 else math.nan
    remaining = max(TARGET_STEPS - int(steps[-1]), 0)
    eta_seconds = float(remaining / steps_per_second) if steps_per_second > 0 else math.nan
    finish = wall_times[-1] + eta_seconds if math.isfinite(eta_seconds) else None
    args_path = run / "args.json"
    args = json.loads(args_path.read_text(encoding="utf-8"))
    return {
        "v_max": speed,
        "seed": int(args["seed"]),
        "experiment": experiment,
        "selected_run": run.name,
        "event_path": str(event),
        "points": int(len(steps)),
        "last_step": int(steps[-1]),
        "progress_percent": float(100.0 * steps[-1] / TARGET_STEPS),
        "last_value": float(values[-1]),
        "recent_steps_per_second": steps_per_second,
        "last_event_time": local_datetime(float(wall_times[-1])),
        "predicted_finish": local_datetime(float(finish)) if finish is not None else None,
        "curve_complete": bool(steps[-1] >= TARGET_STEPS),
        "uav_reset_curriculum": bool(args.get("uav_reset_curriculum", False)),
        "uav_reset_curriculum_schedule": args.get("uav_reset_curriculum_schedule"),
    }


def main() -> None:
    curves = {}
    summaries = []
    for speed, experiment in EXPERIMENTS.items():
        run, event = latest_run(experiment)
        curve = load_curve(event)
        curves[speed] = curve
        summaries.append(summarize(speed, experiment, run, event, curve))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for summary in summaries:
        speed = summary["v_max"]
        curve = curves[speed]
        label = (
            f"v_max={speed} m/s: {summary['last_value']:,.0f} at "
            f"{summary['last_step'] / 1_000_000:.2f}M ({summary['selected_run']})"
        )
        ax.plot(
            curve["steps"] / 1_000_000.0,
            curve["values"],
            color=COLORS[speed],
            linewidth=1.55,
            label=label,
        )

    ax.set_title("Fixed600, MD lifetime=12, noise=3.0, R520: curriculum speed comparison")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("System performance (all GUs)")
    ax.set_xlim(left=0.0, right=TARGET_STEPS / 1_000_000.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.text(
        0.01,
        0.012,
        "Metric: agent0/system_performance_true_all_GUs; raw event points, no smoothing. "
        "v30 selects the newer run2 because run1 is an earlier stopped attempt.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    output_path = HERE / "plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric": METRIC,
        "target_steps": TARGET_STEPS,
        "selection": "newest TensorBoard event file under each experiment's run directory",
        "runs": summaries,
        "output": str(output_path),
    }
    (HERE / "progress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
