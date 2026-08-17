"""Refresh and plot the five live curriculum runs plus the psi=0.3 reference."""

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
PROJECT = HERE.parents[2]
RESULTS = PROJECT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
METRIC = "agent0/system_performance_true_all_GUs"
TOTAL_STEPS = 60_000_000
TZ = ZoneInfo("Asia/Shanghai")

RUNS = [
    ("local psi=0.1 | seed=2", "#0072B2", "-", RESULTS / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1", True),
    ("local psi=0.3 | seed=2", "#E69F00", "-", RESULTS / "dcppoR520_fixed600_200_noactor_psi0p3_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1", True),
    ("remote psi=0.5 | seed=2", "#009E73", "-", HERE / "data" / "remote" / "psi0p5", True),
    ("remote psi=0.7 | seed=2", "#D55E00", "-", HERE / "data" / "remote" / "psi0p7", True),
    ("remote psi=0.9 | seed=2", "#CC79A7", "-", HERE / "data" / "remote" / "psi0p9", True),
    ("remote psi=0.3 | seed=42 | reference", "#000000", "--", HERE / "data" / "remote" / "psi0p3_seed42", False),
]


def event_path(path: Path) -> Path:
    candidates = list(path.glob("events.out.tfevents*")) + list(path.glob("logs/events.out.tfevents*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file under {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def manifest_step(path: Path) -> int | None:
    candidates = [path / "models" / "checkpoint_manifest.json", path / "checkpoint_snapshot" / "checkpoint_manifest.json"]
    for candidate in candidates:
        if candidate.is_file():
            return int(json.loads(candidate.read_text(encoding="utf-8"))["total_num_steps"])
    return None


def load_curve(path: Path):
    actual = event_path(path)
    accumulator = EventAccumulator(str(actual), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    tag = METRIC if METRIC in tags else next((item for item in tags if item.endswith("/system_performance_true_all_GUs")), None)
    if tag is None:
        raise KeyError(f"{METRIC!r} is missing from {actual}")
    events = sorted(accumulator.Scalars(tag), key=lambda item: item.step)
    if not events:
        raise RuntimeError(f"No scalar points for {tag} in {actual}")
    return actual, tag, events


def recent_speed(events) -> float:
    if len(events) < 2:
        return math.nan
    tail = events[-min(20, len(events)) :]
    delta_t = tail[-1].wall_time - tail[0].wall_time
    delta_steps = tail[-1].step - tail[0].step
    return delta_steps / delta_t if delta_t > 0 else math.nan


def main() -> None:
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 7.8, "figure.dpi": 120, "savefig.dpi": 300})
    summaries = []
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for label, color, linestyle, path, live in RUNS:
        actual, tag, events = load_curve(path)
        latest = events[-1]
        speed = recent_speed(events)
        checkpoint = manifest_step(path)
        if checkpoint is None and path.parent.name == "remote":
            checkpoint = manifest_step(HERE / "remote_checkpoints" / path.name)
        eta = latest.wall_time + max(TOTAL_STEPS - latest.step, 0) / speed if live and math.isfinite(speed) and speed > 0 else None
        legend_label = f"{label} | {latest.step / 1e6:.2f}M | {latest.value:,.0f}"
        ax.plot([item.step / 1e6 for item in events], [item.value for item in events], color=color, linestyle=linestyle, linewidth=2.0 if live else 2.2, alpha=0.95, label=legend_label)
        summaries.append({
            "label": label,
            "live": live,
            "event_file": str(actual),
            "metric": tag,
            "points": len(events),
            "latest_step": int(latest.step),
            "checkpoint_step": checkpoint,
            "progress_fraction": float(latest.step / TOTAL_STEPS),
            "latest_value": float(latest.value),
            "recent_steps_per_second": None if not math.isfinite(speed) else float(speed),
            "last_scalar_time": datetime.fromtimestamp(latest.wall_time, TZ).isoformat(timespec="seconds"),
            "estimated_finish": datetime.fromtimestamp(eta, TZ).isoformat(timespec="seconds") if eta else None,
        })
    ax.set_title("Association-threshold comparison with curriculum (deadline filter OFF)", pad=12)
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
    payload = {"generated_at": datetime.now(TZ).isoformat(timespec="seconds"), "metric": METRIC, "total_steps": TOTAL_STEPS, "curves": summaries}
    (HERE / "curve_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (HERE / "curve_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
