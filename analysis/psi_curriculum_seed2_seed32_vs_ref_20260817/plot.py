"""Plot the six live curriculum runs plus the requested historical reference."""

from __future__ import annotations

import csv
import json
import math
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

RUNS = [
    {
        "label": "local psi=0.1 | seed=2 | curriculum ON",
        "color": "#0072B2",
        "linestyle": "-",
        "path": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
        "kind": "live",
    },
    {
        "label": "local psi=0.1 | seed=32 | curriculum ON",
        "color": "#56B4E9",
        "linestyle": "-",
        "path": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed32_60m_20260816" / "run1",
        "kind": "live",
    },
    {
        "label": "local psi=0.3 | seed=2 | curriculum ON",
        "color": "#009E73",
        "linestyle": "-",
        "path": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p3_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
        "kind": "live",
    },
    {
        "label": "remote psi=0.3 | seed=42 | previous reference",
        "color": "#000000",
        "linestyle": "--",
        "path": PROJECT / "analysis" / "psi_curriculum_all_running_20260816" / "data" / "remote" / "psi0p3_seed42_baseline.tfevents",
        "kind": "reference",
    },
    {
        "label": "remote psi=0.5 | seed=2 | curriculum ON",
        "color": "#E69F00",
        "linestyle": "-",
        "path": HERE / "data" / "remote" / "psi0p5.tfevents",
        "kind": "live",
    },
    {
        "label": "remote psi=0.7 | seed=2 | curriculum ON",
        "color": "#D55E00",
        "linestyle": "-",
        "path": HERE / "data" / "remote" / "psi0p7.tfevents",
        "kind": "live",
    },
    {
        "label": "remote psi=0.9 | seed=2 | curriculum ON",
        "color": "#CC79A7",
        "linestyle": "-",
        "path": HERE / "data" / "remote" / "psi0p9.tfevents",
        "kind": "live",
    },
]


def event_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        (path / "logs").glob("events.out.tfevents.*"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file under {path}")
    return candidates[0]


def load_events(path: Path):
    actual = event_path(path)
    accumulator = EventAccumulator(str(actual), size_guidance={"scalars": 0})
    accumulator.Reload()
    tag = METRIC if METRIC in accumulator.Tags().get("scalars", []) else next(
        (item for item in accumulator.Tags().get("scalars", []) if item.endswith("/system_performance_true_all_GUs")),
        None,
    )
    if tag is None:
        raise KeyError(f"{METRIC!r} is missing from {actual}")
    events = sorted(accumulator.Scalars(tag), key=lambda item: item.step)
    if not events:
        raise RuntimeError(f"No scalar points for {tag} in {actual}")
    return actual, tag, events


def estimate_speed(events):
    if len(events) < 2:
        return math.nan
    tail = events[-min(20, len(events)):]
    dt = tail[-1].wall_time - tail[0].wall_time
    ds = tail[-1].step - tail[0].step
    return ds / dt if dt > 0 else math.nan


def main() -> None:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 8.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })
    summaries = []
    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    for run in RUNS:
        actual, tag, events = load_events(run["path"])
        steps = [event.step for event in events]
        values = [event.value for event in events]
        latest = events[-1]
        speed = estimate_speed(events)
        eta = None
        if run["kind"] == "live" and speed == speed and speed > 0:
            eta = latest.wall_time + max(TOTAL_STEPS - latest.step, 0) / speed
        label = f"{run['label']} | {latest.step / 1e6:.2f}M, {latest.value:,.0f}"
        ax.plot(
            [step / 1e6 for step in steps],
            values,
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=2.0 if run["kind"] == "live" else 2.2,
            alpha=0.95,
            label=label,
        )
        summaries.append({
            "label": run["label"],
            "kind": run["kind"],
            "event_file": str(actual),
            "metric": tag,
            "points": len(events),
            "latest_step": int(latest.step),
            "progress_fraction": float(latest.step / TOTAL_STEPS),
            "latest_value": float(latest.value),
            "recent_steps_per_second": None if speed != speed else float(speed),
            "last_scalar_time": datetime.fromtimestamp(latest.wall_time, TZ).isoformat(),
            "estimated_finish": datetime.fromtimestamp(eta, TZ).isoformat() if eta else None,
        })

    ax.set_title("Association-threshold comparison with curriculum (deadline filter OFF)", pad=12)
    ax.set_xlabel("Environment steps (million)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(0, TOTAL_STEPS / 1e6)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.94, borderpad=0.7, handlelength=2.8)
    fig.tight_layout()
    fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    (HERE / "curve_summary.json").write_text(
        json.dumps({"generated_at": datetime.now(TZ).isoformat(), "metric": METRIC, "curves": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (HERE / "curve_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
