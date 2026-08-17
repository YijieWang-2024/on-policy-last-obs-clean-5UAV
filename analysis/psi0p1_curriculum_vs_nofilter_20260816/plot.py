"""Plot the two live curriculum runs against the five stopped psi baselines."""

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
OLD_ANALYSIS = PROJECT / "analysis" / "association_threshold_seed42_seed2_latest_20260816"
REMOTE_DATA = OLD_ANALYSIS / "data" / "remote"
METRIC = "agent0/system_performance_true_all_GUs"
TOTAL_STEPS = 60_000_000
TZ = ZoneInfo("Asia/Shanghai")

# Okabe-Ito colors plus black; readable for red-green colour-vision deficiency.
CURVES = [
    {
        "key": "curr_seed2",
        "label": "curriculum ON | deadline OFF | psi=0.1 | seed=2",
        "color": "#0072B2",
        "linestyle": "-",
        "source": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1" / "logs",
    },
    {
        "key": "curr_seed32",
        "label": "curriculum ON | deadline OFF | psi=0.1 | seed=32",
        "color": "#D55E00",
        "linestyle": "-",
        "source": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed32_60m_20260816" / "run1" / "logs",
    },
    {
        "key": "off_remote_01",
        "label": "curriculum OFF | remote | deadline OFF | psi=0.1 | seed=42",
        "color": "#009E73",
        "linestyle": "--",
        "source": REMOTE_DATA / "remote_psi0p1.tfevents",
    },
    {
        "key": "off_remote_03",
        "label": "curriculum OFF | remote | deadline OFF | psi=0.3 | seed=42",
        "color": "#E69F00",
        "linestyle": "--",
        "source": REMOTE_DATA / "remote_psi0p3.tfevents",
    },
    {
        "key": "off_remote_05",
        "label": "curriculum OFF | remote | deadline OFF | psi=0.5 | seed=42",
        "color": "#56B4E9",
        "linestyle": "--",
        "source": REMOTE_DATA / "remote_psi0p5.tfevents",
    },
    {
        "key": "off_local_07",
        "label": "curriculum OFF | local | deadline OFF | psi=0.7 | seed=42",
        "color": "#CC79A7",
        "linestyle": ":",
        "source": RESULTS / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed42_60m_20260816" / "run1" / "logs",
    },
    {
        "key": "off_local_09",
        "label": "curriculum OFF | local | deadline OFF | psi=0.9 | seed=42",
        "color": "#000000",
        "linestyle": ":",
        "source": RESULTS / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p9_nofilter_seed42_60m_20260816" / "run1" / "logs",
    },
]


def event_file(source: Path) -> Path:
    if source.is_file():
        return source
    files = sorted(source.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found under {source}")
    return files[0]


def load_events(path: Path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"{METRIC!r} is missing from {path}")
    events = accumulator.Scalars(METRIC)
    if not events:
        raise ValueError(f"{METRIC!r} has no events in {path}")
    return events


def summarize(item: dict, path: Path, events: list) -> dict:
    first, last = events[0], events[-1]
    recent = events[-min(25, len(events)) :]
    dt = recent[-1].wall_time - recent[0].wall_time
    ds = recent[-1].step - recent[0].step
    speed = ds / dt if dt > 0 else None
    remaining = max(TOTAL_STEPS - last.step, 0)
    eta = last.wall_time + remaining / speed if speed and speed > 0 else None
    return {
        "key": item["key"],
        "label": item["label"],
        "source": str(path),
        "points": len(events),
        "last_step": int(last.step),
        "progress_fraction": float(last.step / TOTAL_STEPS),
        "latest_value": float(last.value),
        "recent_steps_per_second": speed,
        "last_scalar_time": datetime.fromtimestamp(last.wall_time, TZ).isoformat(),
        "estimated_finish": datetime.fromtimestamp(eta, TZ).isoformat() if eta else None,
    }


def main() -> None:
    loaded = []
    summaries = []
    for item in CURVES:
        path = event_file(item["source"])
        events = load_events(path)
        loaded.append((item, events))
        summaries.append(summarize(item, path, events))

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    for item, events in loaded:
        x = [event.step / 1_000_000 for event in events]
        y = [event.value for event in events]
        label = f"{item['label']} | latest={y[-1]:,.0f} @ {x[-1]:.2f}M"
        ax.plot(
            x,
            y,
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=2.0 if item["key"].startswith("curr_") else 1.7,
            alpha=0.95,
            label=label,
        )

    ax.set_title("psi threshold comparison: curriculum ON vs curriculum OFF", pad=10)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(0, TOTAL_STEPS / 1_000_000)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, -0.08),
        frameon=True,
        framealpha=0.96,
        fontsize=7.2,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.38, left=0.10, right=0.98, top=0.91)
    output = HERE / "plot.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    payload = {
        "generated_at": datetime.now(TZ).isoformat(),
        "metric": METRIC,
        "total_steps": TOTAL_STEPS,
        "curve_count": len(summaries),
        "curves": summaries,
        "output": str(output),
    }
    (HERE / "curve_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (HERE / "curve_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
