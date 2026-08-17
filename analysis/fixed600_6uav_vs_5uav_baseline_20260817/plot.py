"""Plot the live six-UAV run against the historical five-UAV psi=0.5 baseline."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
METRIC = "agent0/system_performance_true_all_GUs"
TOTAL_STEPS = 60_000_000
TZ = ZoneInfo("Asia/Shanghai")

CURRENT_EXPERIMENT = (
    "dcppoR520_fixed600_200_6uav_layoutctx_noactor_peragentnoise_s3p0_md12_"
    "vmax30_psi0p5_nofilter_seed2_60m_20260817"
)
BASELINE_EXPERIMENT = (
    "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
    "vmax30_psi0p5_nofilter_seed2_60m_20260814"
)

SERIES = (
    (
        "6-UAV current | psi=0.5 | deadline OFF",
        CURRENT_EXPERIMENT,
        "#0072B2",
        "-",
    ),
    (
        "5-UAV baseline | psi=0.5 | deadline OFF",
        BASELINE_EXPERIMENT,
        "#000000",
        "--",
    ),
)


def latest_event_file(experiment: str) -> Path:
    event_dir = RESULTS / experiment / "run1" / "logs"
    paths = sorted(
        event_dir.glob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event file found in {event_dir}")
    return paths[0]


def load_events(path: Path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    if METRIC not in available:
        raise KeyError(f"{METRIC!r} is missing from {path}; available={available}")
    events = accumulator.Scalars(METRIC)
    if not events:
        raise ValueError(f"{METRIC!r} has no scalar events in {path}")
    return events


def summarize(label: str, status: str, path: Path, events) -> dict:
    first, last = events[0], events[-1]
    recent = events[-min(25, len(events)) :]
    dt = recent[-1].wall_time - recent[0].wall_time
    ds = recent[-1].step - recent[0].step
    speed = ds / dt if dt > 0 else None
    remaining = max(TOTAL_STEPS - last.step, 0)
    eta = last.wall_time + remaining / speed if status == "running" and speed and speed > 0 else None
    return {
        "label": label,
        "status": status,
        "source": str(path),
        "metric": METRIC,
        "points": len(events),
        "first_step": int(first.step),
        "last_step": int(last.step),
        "progress_fraction": float(last.step / TOTAL_STEPS),
        "latest_value": float(last.value),
        "recent_steps_per_second": speed,
        "last_scalar_time": datetime.fromtimestamp(last.wall_time, TZ).isoformat(),
        "estimated_finish": datetime.fromtimestamp(eta, TZ).isoformat() if eta else None,
    }


def main() -> None:
    curves = []
    summaries = []
    for label, experiment, color, linestyle in SERIES:
        path = latest_event_file(experiment)
        events = load_events(path)
        status = "running" if experiment == CURRENT_EXPERIMENT else "completed_or_stopped"
        curves.append((label, color, linestyle, events))
        summaries.append(summarize(label, status, path, events))

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    for label, color, linestyle, events in curves:
        x = [event.step / 1_000_000 for event in events]
        y = [event.value for event in events]
        legend_label = f"{label} | latest={y[-1]:,.0f} @ {x[-1]:.1f}M"
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            alpha=0.95,
            label=legend_label,
        )

    ax.set_title("Fixed600-200: six-UAV run vs five-UAV psi=0.5 baseline", pad=10)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(0, TOTAL_STEPS / 1_000_000)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.96, fontsize=8.5)
    fig.subplots_adjust(bottom=0.20, left=0.12, right=0.98, top=0.91)
    fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    payload = {
        "generated_at": datetime.now(TZ).isoformat(),
        "metric": METRIC,
        "total_steps": TOTAL_STEPS,
        "curves": summaries,
        "outputs": {"plot": str(HERE / "plot.png")},
    }
    (HERE / "curve_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (HERE / "curve_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
