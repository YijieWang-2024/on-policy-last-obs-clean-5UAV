"""Refreshable training-curve plot for the five curriculum-on runs plus a reference."""

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

# Okabe-Ito palette plus line styles: readable for red-green colour-vision deficiency.
RUNS = [
    {
        "label": "local $\\psi=0.1$, seed=2 | curriculum ON | deadline OFF",
        "color": "#0072B2",
        "linestyle": "-",
        "path": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
        "status": "finished",
    },
    {
        "label": "local $\\psi=0.3$, seed=2 | curriculum ON | deadline OFF",
        "color": "#E69F00",
        "linestyle": "-",
        "path": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p3_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
        "status": "running",
    },
    {
        "label": "remote $\\psi=0.5$, seed=2 | curriculum ON | deadline OFF",
        "color": "#009E73",
        "linestyle": "-",
        "path": HERE / "data" / "remote" / "psi0p5.tfevents",
        "status": "finished",
    },
    {
        "label": "remote $\\psi=0.7$, seed=2 | curriculum ON | deadline OFF",
        "color": "#D55E00",
        "linestyle": "-",
        "path": HERE / "data" / "remote" / "psi0p7.tfevents",
        "status": "finished",
    },
    {
        "label": "remote $\\psi=0.9$, seed=2 | curriculum ON | deadline OFF",
        "color": "#CC79A7",
        "linestyle": "-",
        "path": HERE / "data" / "remote" / "psi0p9.tfevents",
        "status": "finished",
    },
    {
        "label": "reference remote $\\psi=0.3$, seed=42 | historical",
        "color": "#000000",
        "linestyle": "--",
        "path": HERE / "data" / "reference" / "psi0p3_seed42.tfevents",
        "status": "reference",
    },
]


def event_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        path.glob("logs/events.out.tfevents.*"),
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
    tags = accumulator.Tags().get("scalars", [])
    tag = METRIC if METRIC in tags else next(
        (item for item in tags if item.endswith("/system_performance_true_all_GUs")),
        None,
    )
    if tag is None:
        raise KeyError(f"{METRIC!r} is missing from {actual}")
    events = sorted(accumulator.Scalars(tag), key=lambda item: item.step)
    if not events:
        raise RuntimeError(f"No scalar points for {tag} in {actual}")
    return actual, tag, events


def estimate_speed(events) -> float:
    if len(events) < 2:
        return math.nan
    tail = events[-min(20, len(events)) :]
    delta_t = tail[-1].wall_time - tail[0].wall_time
    delta_steps = tail[-1].step - tail[0].step
    return delta_steps / delta_t if delta_t > 0 else math.nan


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 7.4,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )

    summaries = []
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for run in RUNS:
        actual, tag, events = load_events(run["path"])
        latest = events[-1]
        speed = estimate_speed(events)
        eta = None
        if run["status"] == "running" and math.isfinite(speed) and speed > 0:
            eta = latest.wall_time + max(TOTAL_STEPS - latest.step, 0) / speed

        label = (
            f"{run['label']} | {latest.step / 1e6:.2f}M | "
            f"{latest.value:,.0f}"
        )
        ax.plot(
            [item.step / 1e6 for item in events],
            [item.value for item in events],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=2.0 if run["status"] != "reference" else 2.2,
            alpha=0.95,
            label=label,
        )
        summaries.append(
            {
                "label": run["label"],
                "status": run["status"],
                "event_file": str(actual),
                "metric": tag,
                "points": len(events),
                "latest_step": int(latest.step),
                "progress_fraction": float(latest.step / TOTAL_STEPS),
                "latest_value": float(latest.value),
                "recent_steps_per_second": None if not math.isfinite(speed) else float(speed),
                "last_scalar_time": datetime.fromtimestamp(latest.wall_time, TZ).isoformat(),
                "estimated_finish": datetime.fromtimestamp(eta, TZ).isoformat() if eta else None,
            }
        )

    ax.set_title(
        "Association-threshold comparison with reset curriculum (deadline filter OFF)",
        pad=12,
    )
    ax.set_xlabel("Environment steps (million)")
    ax.set_ylabel("System performance (true all GUs)")
    ax.set_xlim(0, TOTAL_STEPS / 1e6)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.94, borderpad=0.7, handlelength=2.8)
    fig.tight_layout()
    fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    payload = {
        "generated_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "metric": METRIC,
        "total_steps": TOTAL_STEPS,
        "curves": summaries,
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
