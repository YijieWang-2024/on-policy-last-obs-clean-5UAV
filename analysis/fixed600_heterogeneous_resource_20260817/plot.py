"""Plot the local heterogeneous-resource run against the remote baseline."""

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
TARGET_EXPERIMENT = (
    "dcppoR520_fixed600_200_het1_13_07_13_07_psi05_nf_cur_s2_60m_20260817"
)
REMOTE_EVENT = HERE / "data" / "remote_psi0p5_curriculum.tfevents"
METRIC = "agent0/system_performance_true_all_GUs"
TOTAL_STEPS = 60_000_000
TZ = ZoneInfo("Asia/Shanghai")


def latest_local_event(experiment: str) -> Path:
    candidates = []
    for run_dir in (RESULTS / experiment).glob("run*"):
        candidates.extend((run_dir / "logs").glob("events.out.tfevents.*"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No local event file found for {experiment}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def load_events(path: Path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"{METRIC!r} is missing from {path}")
    events = accumulator.Scalars(METRIC)
    if not events:
        raise ValueError(f"{METRIC!r} has no scalar events in {path}")
    return events


def summarize(label: str, source: Path, events, status: str) -> dict:
    last = events[-1]
    recent = events[-min(25, len(events)) :]
    elapsed = recent[-1].wall_time - recent[0].wall_time
    steps = recent[-1].step - recent[0].step
    speed = steps / elapsed if elapsed > 0 else None
    remaining = max(TOTAL_STEPS - int(last.step), 0)
    eta = (
        last.wall_time + remaining / speed
        if status == "running" and speed and speed > 0
        else None
    )
    return {
        "label": label,
        "status": status,
        "source": str(source),
        "metric": METRIC,
        "points": len(events),
        "last_step": int(last.step),
        "progress_fraction": float(last.step / TOTAL_STEPS),
        "latest_value": float(last.value),
        "recent_steps_per_second": float(speed) if speed else None,
        "last_scalar_time": datetime.fromtimestamp(last.wall_time, TZ).isoformat(),
        "estimated_finish": (
            datetime.fromtimestamp(eta, TZ).isoformat() if eta else None
        ),
    }


def main() -> None:
    if not REMOTE_EVENT.is_file():
        raise FileNotFoundError(
            f"Fetch the remote baseline event first: {REMOTE_EVENT}"
        )

    local_source = latest_local_event(TARGET_EXPERIMENT)
    loaded = [
        (
            "local heterogeneous | [1.0,1.3,0.7,1.3,0.7] | psi=0.5 | curriculum",
            local_source,
            "#0072B2",
            "-",
            "running",
        ),
        (
            "remote homogeneous | psi=0.5 | curriculum | deadline OFF",
            REMOTE_EVENT,
            "#000000",
            "--",
            "remote_latest_snapshot",
        ),
    ]

    summaries = []
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    for label, source, color, linestyle, status in loaded:
        events = load_events(source)
        x = [event.step / 1_000_000 for event in events]
        y = [event.value for event in events]
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            alpha=0.95,
            label=f"{label} | latest={y[-1]:,.0f} @ {x[-1]:.2f}M",
        )
        summaries.append(summarize(label, source, events, status))

    ax.set_title("Fixed600-200: heterogeneous resources vs homogeneous curriculum baseline", pad=10)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(0, TOTAL_STEPS / 1_000_000)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.96)
    fig.subplots_adjust(bottom=0.18, left=0.12, right=0.98, top=0.91)
    output = HERE / "plot.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    payload = {
        "generated_at": datetime.now(TZ).isoformat(),
        "metric": METRIC,
        "total_steps": TOTAL_STEPS,
        "curves": summaries,
        "output": str(output),
    }
    (HERE / "curve_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (HERE / "curve_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
