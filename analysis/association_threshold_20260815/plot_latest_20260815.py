"""Plot the latest association-threshold checkpoints.

The three remote curves use the event snapshots copied from the 3090 on
2026-08-15.  Local curves are read from the newest event file in each live
run directory.  The script deliberately keeps the remote stopped-run status
separate from the curve itself: a stopped run has no valid future ETA.
"""

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
REMOTE_CURRENT = HERE / "data" / "remote" / "current_20260815_final"
REMOTE_BASELINE = HERE / "data" / "remote" / "baseline_R520_psi0p5_filter.tfevents"
METRIC = "agent0/system_performance_true_all_GUs"
TOTAL_STEPS = 60_000_000
TZ = ZoneInfo("Asia/Shanghai")

# Okabe-Ito palette: readable for common red-green colour-vision deficiencies.
COLORS = {
    "remote_0.3": "#0072B2",
    "remote_0.7": "#E69F00",
    "remote_0.9": "#D55E00",
    "baseline": "#000000",
    "local_0.3": "#009E73",
    "local_0.5": "#CC79A7",
    "local_0.7": "#56B4E9",
}

LOCAL_EXPERIMENTS = {
    "local_0.3": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p3_nofilter_seed2_60m_20260814",
    "local_0.5": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed2_60m_20260814",
    "local_0.7": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed2_60m_20260814",
}


def local_event(experiment: str) -> Path:
    paths = sorted(
        (RESULTS / experiment / "run1" / "logs").glob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No event file found for {experiment}")
    return paths[0]


def load_events(path: Path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"{METRIC!r} is missing from {path}")
    events = accumulator.Scalars(METRIC)
    if not events:
        raise ValueError(f"{METRIC!r} has no events in {path}")
    return events


def summarize(label: str, status: str, path: Path, events) -> dict:
    first, last = events[0], events[-1]
    recent = events[-min(25, len(events)) :]
    dt = recent[-1].wall_time - recent[0].wall_time
    ds = recent[-1].step - recent[0].step
    speed = ds / dt if dt > 0 else None
    remaining = max(TOTAL_STEPS - last.step, 0)
    eta = last.wall_time + remaining / speed if speed and speed > 0 and status == "running" else None
    return {
        "label": label,
        "status": status,
        "source": str(path),
        "metric": METRIC,
        "points": len(events),
        "first_step": int(first.step),
        "last_step": int(last.step),
        "remaining_steps": int(remaining),
        "progress_fraction": float(last.step / TOTAL_STEPS),
        "latest_value": float(last.value),
        "recent_steps_per_second": speed,
        "last_scalar_time": datetime.fromtimestamp(last.wall_time, TZ).isoformat(),
        "estimated_finish": datetime.fromtimestamp(eta, TZ).isoformat() if eta else None,
    }


def curve(label: str, key: str, path: Path, status: str, linestyle: str = "-") -> dict:
    return {
        "label": label,
        "key": key,
        "path": path,
        "status": status,
        "color": COLORS[key],
        "linestyle": linestyle,
        "events": load_events(path),
    }


def plot_curves(curves: list[dict], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    for item in curves:
        events = item["events"]
        x = [event.step / 1_000_000 for event in events]
        y = [event.value for event in events]
        label = f"{item['label']} | latest={y[-1]:,.0f} @ {x[-1]:.1f}M"
        ax.plot(
            x,
            y,
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=1.8,
            alpha=0.95,
            label=label,
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(0, TOTAL_STEPS / 1_000_000)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, -0.06),
        frameon=True,
        framealpha=0.96,
        fontsize=7.8 if len(curves) >= 7 else 8.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.34 if len(curves) < 7 else 0.43, left=0.12, right=0.98, top=0.91)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    remote = [
        curve(
            "remote psi=0.3 | deadline ON",
            "remote_0.3",
            REMOTE_CURRENT / "remote_psi0p3.tfevents",
            "stopped/no active PID",
        ),
        curve(
            "remote psi=0.7 | deadline ON",
            "remote_0.7",
            REMOTE_CURRENT / "remote_psi0p7.tfevents",
            "stopped/no active PID",
        ),
        curve(
            "remote psi=0.9 | deadline ON",
            "remote_0.9",
            REMOTE_CURRENT / "remote_psi0p9.tfevents",
            "stopped/no active PID",
        ),
    ]
    baseline = curve(
        "baseline psi=0.5 | deadline ON",
        "baseline",
        REMOTE_BASELINE,
        "completed checkpoint",
        linestyle="--",
    )
    local = [
        curve(
            "local psi=0.3 | deadline OFF",
            "local_0.3",
            local_event(LOCAL_EXPERIMENTS["local_0.3"]),
            "running",
            linestyle="-.",
        ),
        curve(
            "local psi=0.5 | deadline OFF",
            "local_0.5",
            local_event(LOCAL_EXPERIMENTS["local_0.5"]),
            "running",
            linestyle="-.",
        ),
        curve(
            "local psi=0.7 | deadline OFF",
            "local_0.7",
            local_event(LOCAL_EXPERIMENTS["local_0.7"]),
            "running",
            linestyle="-.",
        ),
    ]

    outputs = {
        "remote_vs_baseline": HERE / "remote_psi_thresholds_vs_baseline_latest_20260815.png",
        "local_nofilter": HERE / "local_psi_thresholds_deadline_off_latest_20260815.png",
        "all_seven": HERE / "all_seven_psi_thresholds_latest_20260815.png",
    }
    plot_curves(remote + [baseline], "Association threshold: remote sweep vs deadline-filter baseline", outputs["remote_vs_baseline"])
    plot_curves(local, "Association threshold: local deadline-filter OFF", outputs["local_nofilter"])
    plot_curves(remote + local + [baseline], "Association threshold: all seven runs", outputs["all_seven"])

    all_curves = remote + local + [baseline]
    summaries = [summarize(item["label"], item["status"], item["path"], item["events"]) for item in all_curves]
    payload = {
        "generated_at": datetime.now(TZ).isoformat(),
        "metric": METRIC,
        "total_steps": TOTAL_STEPS,
        "remote_snapshot_note": "Remote event files are read-only snapshots copied from the stopped 3090 runs on 2026-08-15.",
        "curves": summaries,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (HERE / "curve_summary_latest_20260815.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (HERE / "curve_summary_latest_20260815.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
