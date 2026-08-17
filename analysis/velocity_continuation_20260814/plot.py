"""Plot original and warm-start continuation curves on one cumulative-step axis."""

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
RESULT_ROOT = HERE.parent.parent / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
METRIC = "agent0/system_performance_true_all_GUs"
CONTINUATION_ARG_STEPS = 40_000_000
ROLLOUT_GRANULARITY = 64 * 400

RUNS = {
    20: {
        "seed": 2,
        "color": "#0072B2",
        "source": "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax20_seed2_60m_20260812/run1",
        "continuation": "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax20_seed2_resume40m_20260814/run1",
    },
    30: {
        "seed": 32,
        "color": "#E69F00",
        "source": "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax30_seed32_60m_20260811_retry2/run1",
        "continuation": "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax30_seed32_resume40m_20260814/run1",
    },
    40: {
        "seed": 32,
        "color": "#CC79A7",
        "source": "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed32_60m_20260812/run1",
        "continuation": "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed32_resume40m_20260814/run1",
    },
}


def load_curve(run_dir: Path) -> dict[str, np.ndarray]:
    events = sorted((run_dir / "logs").glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file under {run_dir}")
    event_path = max(events, key=lambda path: path.stat().st_mtime)
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
        "event_path": str(event_path),
    }


def manifest(run_dir: Path) -> dict:
    path = run_dir / "models" / "checkpoint_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def local_datetime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def summarize(speed: int, spec: dict) -> tuple[dict, dict]:
    source_run = RESULT_ROOT / spec["source"]
    continuation_run = RESULT_ROOT / spec["continuation"]
    source_manifest = manifest(source_run)
    continuation_manifest = manifest(continuation_run)
    source_steps = int(source_manifest["total_num_steps"])
    declared_source_steps = int(continuation_manifest["source_total_num_steps"])
    if source_steps != declared_source_steps:
        raise ValueError(
            f"{speed} m/s source-step mismatch: source={source_steps}, "
            f"continuation declares={declared_source_steps}"
        )

    source_curve = load_curve(source_run)
    continuation_curve = load_curve(continuation_run)
    cumulative_steps = np.concatenate(
        [source_curve["steps"], source_steps + continuation_curve["steps"]]
    )
    values = np.concatenate([source_curve["values"], continuation_curve["values"]])
    wall_times = np.concatenate(
        [source_curve["wall_times"], continuation_curve["wall_times"]]
    )

    points = min(80, len(continuation_curve["steps"]) - 1)
    if points > 0:
        delta_steps = continuation_curve["steps"][-1] - continuation_curve["steps"][-1 - points]
        delta_seconds = continuation_curve["wall_times"][-1] - continuation_curve["wall_times"][-1 - points]
    else:
        delta_steps = 0
        delta_seconds = 0.0
    steps_per_second = float(delta_steps / delta_seconds) if delta_seconds > 0 else math.nan
    actual_session_target = (CONTINUATION_ARG_STEPS // ROLLOUT_GRANULARITY) * ROLLOUT_GRANULARITY
    target_cumulative_steps = source_steps + actual_session_target
    current_cumulative_steps = int(cumulative_steps[-1])
    remaining = max(target_cumulative_steps - current_cumulative_steps, 0)
    eta_seconds = remaining / steps_per_second if steps_per_second > 0 else math.nan
    finish_timestamp = continuation_curve["wall_times"][-1] + eta_seconds if math.isfinite(eta_seconds) else None

    summary = {
        "v_max": speed,
        "seed": spec["seed"],
        "source_run": str(source_run),
        "continuation_run": str(continuation_run),
        "source_checkpoint_steps": source_steps,
        "continuation_manifest_steps": int(continuation_manifest["total_num_steps"]),
        "current_curve_cumulative_steps": current_cumulative_steps,
        "target_cumulative_steps": target_cumulative_steps,
        "progress_percent": 100.0 * current_cumulative_steps / target_cumulative_steps,
        "current_value": float(values[-1]),
        "recent_steps_per_second": steps_per_second,
        "last_event_time": local_datetime(float(wall_times[-1])),
        "predicted_finish": local_datetime(finish_timestamp) if finish_timestamp is not None else None,
        "source_event_path": source_curve["event_path"],
        "continuation_event_path": continuation_curve["event_path"],
    }
    curves = {
        "steps": cumulative_steps,
        "values": values,
        "source_steps": source_curve["steps"],
        "continuation_steps": source_steps + continuation_curve["steps"],
    }
    return summary, curves


def main() -> None:
    summaries = []
    curves = {}
    for speed, spec in RUNS.items():
        summary, curve = summarize(speed, spec)
        summaries.append(summary)
        curves[speed] = curve

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for summary in summaries:
        speed = summary["v_max"]
        spec = RUNS[speed]
        curve = curves[speed]
        label = (
            f"v_max={speed} m/s, seed={summary['seed']}: "
            f"{summary['current_value']:,.0f} at "
            f"{summary['current_curve_cumulative_steps'] / 1e6:.2f}M"
        )
        ax.plot(
            curve["steps"] / 1e6,
            curve["values"],
            color=spec["color"],
            linewidth=1.35,
            label=label,
        )
        ax.axvline(
            summary["source_checkpoint_steps"] / 1e6,
            color=spec["color"],
            linestyle="--",
            linewidth=0.8,
            alpha=0.22,
        )

    max_target = max(item["target_cumulative_steps"] for item in summaries)
    ax.set_title("Velocity comparison with warm-start continuation")
    ax.set_xlabel("Cumulative environment steps (M)")
    ax.set_ylabel("System performance (all GUs)")
    ax.set_xlim(0.0, max_target / 1e6)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.text(
        0.01,
        0.012,
        "Solid curves combine the original run and continuation; dashed vertical lines mark each warm-start boundary. "
        "Continuation event steps are offset by the source checkpoint total.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    output_path = HERE / "plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    output = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric": METRIC,
        "continuation_argument_steps": CONTINUATION_ARG_STEPS,
        "rollout_granularity": ROLLOUT_GRANULARITY,
        "runs": summaries,
        "output": str(output_path),
    }
    (HERE / "progress_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
