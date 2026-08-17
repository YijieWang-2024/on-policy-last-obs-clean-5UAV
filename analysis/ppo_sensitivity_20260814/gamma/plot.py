"""Plot the completed gamma sensitivity against the matched baseline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "remote"
METRIC = "agent0/system_performance_true_all_GUs"
TOTAL_STEPS = 60_000_000
CURVES = [
    ("Baseline · gamma=0.99 · seed=2", DATA / "baseline_R520_seed2.tfevents", "#0072B2", "-"),
    ("Variant · gamma=0.95 · seed=32", DATA / "gamma0p95_seed32.tfevents", "#E69F00", "--"),
    ("Variant · gamma=0.90 · seed=32", DATA / "gamma0p90_seed32.tfevents", "#CC79A7", "-.") ,
]


def load_events(path: Path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"{METRIC!r} is missing from {path}")
    return accumulator.Scalars(METRIC)


def summarize(label: str, path: Path, events) -> dict:
    first, last = events[0], events[-1]
    recent = events[-min(100, len(events)) :]
    dt = recent[-1].wall_time - recent[0].wall_time
    ds = recent[-1].step - recent[0].step
    fps = ds / dt if dt > 0 else None
    remaining = max(TOTAL_STEPS - last.step, 0)
    finish = last.wall_time + remaining / fps if fps else last.wall_time
    complete = last.step >= TOTAL_STEPS - 2 * 25_600
    return {
        "label": label,
        "source": str(path),
        "metric": METRIC,
        "points": len(events),
        "last_step": last.step,
        "last_value": last.value,
        "progress_fraction": last.step / TOTAL_STEPS,
        "complete": complete,
        "remaining_steps": remaining,
        "recent_steps_per_second": fps,
        "last_scalar_time": datetime.fromtimestamp(last.wall_time, ZoneInfo("Asia/Shanghai")).isoformat(),
        "estimated_finish": datetime.fromtimestamp(finish, ZoneInfo("Asia/Shanghai")).isoformat(),
    }


def main() -> None:
    loaded = [(label, path, color, linestyle, load_events(path)) for label, path, color, linestyle in CURVES]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    summaries = []
    for label, path, color, linestyle, events in loaded:
        x = [event.step / 1_000_000 for event in events]
        y = [event.value for event in events]
        last = events[-1]
        is_complete = last.step >= TOTAL_STEPS - 2 * 25_600
        suffix = f"final={y[-1]:,.0f}" if is_complete else f"latest={y[-1]:,.0f} @ {last.step / 1_000_000:.2f}M"
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.8,
                label=f"{label} ({suffix})")
        summaries.append(summarize(label, path, events))

    ax.set_title("PPO sensitivity: discount factor", pad=10)
    ax.set_xlabel("Training steps (million)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(0, TOTAL_STEPS / 1_000_000)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.035), frameon=True,
              framealpha=0.96, fontsize=8.1, borderaxespad=0.0)
    fig.subplots_adjust(bottom=0.33, left=0.12, right=0.98, top=0.91)
    output = HERE / "plot.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    (HERE / "curve_summary.json").write_text(
        json.dumps({"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    "metric": METRIC, "curves": summaries, "output": str(output)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {output}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
