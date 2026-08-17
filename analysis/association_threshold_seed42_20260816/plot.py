"""Plot the five current seed-42 association-threshold training curves."""

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

# Okabe-Ito palette: colorblind-friendly and consistent with the previous
# association-threshold figures.
CURVES = [
    ("remote psi=0.1 | deadline OFF", "psi0p1", "#0072B2", "-", HERE / "data" / "remote" / "remote_psi0p1.tfevents"),
    ("remote psi=0.3 | deadline OFF", "psi0p3", "#E69F00", "-", HERE / "data" / "remote" / "remote_psi0p3.tfevents"),
    ("remote psi=0.5 | deadline OFF", "psi0p5", "#D55E00", "-", HERE / "data" / "remote" / "remote_psi0p5.tfevents"),
    ("local psi=0.7 | deadline OFF", "psi0p7", "#009E73", "--", None),
    ("local psi=0.9 | deadline OFF", "psi0p9", "#CC79A7", "--", None),
]

LOCAL_NAMES = {
    "psi0p7": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed42_60m_20260816",
    "psi0p9": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p9_nofilter_seed42_60m_20260816",
}


def local_event(name: str) -> Path:
    paths = sorted(
        (RESULTS / name / "run1" / "logs").glob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event file found for {name}")
    return paths[0]


def load_events(path: Path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"{METRIC!r} is missing from {path}")
    events = accumulator.Scalars(METRIC)
    if not events:
        raise ValueError(f"{METRIC!r} has no scalar events in {path}")
    return events


def summarize(label: str, source: Path, events) -> dict:
    last = events[-1]
    recent = events[-min(25, len(events)):]
    dt = recent[-1].wall_time - recent[0].wall_time
    ds = recent[-1].step - recent[0].step
    speed = ds / dt if dt > 0 else None
    remaining = max(TOTAL_STEPS - last.step, 0)
    eta = last.wall_time + remaining / speed if speed and speed > 0 else None
    return {
        "label": label,
        "source": str(source),
        "metric": METRIC,
        "points": len(events),
        "last_step": int(last.step),
        "remaining_steps": int(remaining),
        "progress_fraction": float(last.step / TOTAL_STEPS),
        "latest_value": float(last.value),
        "recent_steps_per_second": float(speed) if speed else None,
        "last_scalar_time": datetime.fromtimestamp(last.wall_time, TZ).isoformat(),
        "estimated_finish": datetime.fromtimestamp(eta, TZ).isoformat() if eta else None,
    }


def main() -> None:
    loaded = []
    for label, key, color, linestyle, source in CURVES:
        if source is None:
            source = local_event(LOCAL_NAMES[key])
        events = load_events(source)
        loaded.append({"label": label, "key": key, "color": color, "linestyle": linestyle, "source": source, "events": events})

    summaries = []
    fig, ax = plt.subplots(figsize=(10.6, 6.1))
    for item in loaded:
        events = item["events"]
        x = [event.step / 1_000_000 for event in events]
        y = [event.value for event in events]
        ax.plot(
            x,
            y,
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=1.9,
            alpha=0.95,
            label=f"{item['label']} | latest={y[-1]:,.0f} @ {x[-1]:.2f}M",
        )
        summaries.append(summarize(item["label"], item["source"], events))

    ax.set_title("Association-threshold sweep: seed=42, deadline filter OFF", pad=10)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(0, TOTAL_STEPS / 1_000_000)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=8.2,
        framealpha=0.96,
        ncol=1,
    )
    fig.subplots_adjust(bottom=0.39, left=0.12, right=0.98, top=0.91)
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
    (HERE / "curve_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (HERE / "curve_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
