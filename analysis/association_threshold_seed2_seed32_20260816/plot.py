"""Plot the four explicitly requested psi curves."""

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

RUNS = {
    "local_psi0p5_seed2": (
        "local $\\psi=0.5$ | seed 2 | OFF",
        "#0072B2",
        "-",
        RESULTS
        / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed2_60m_20260814"
        / "run1"
        / "logs",
    ),
    "local_psi0p7_seed2": (
        "local $\\psi=0.7$ | seed 2 | OFF",
        "#D55E00",
        "-",
        RESULTS
        / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed2_60m_20260814"
        / "run1"
        / "logs",
    ),
    "local_psi0p9_seed32": (
        "local $\\psi=0.9$ | seed 32 | OFF",
        "#009E73",
        "--",
        RESULTS
        / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p9_nofilter_seed32_60m_20260815"
        / "run1"
        / "logs",
    ),
    "remote_psi0p3_seed32": (
        "remote $\\psi=0.3$ | seed 32 | OFF",
        "#CC79A7",
        "--",
        HERE / "data" / "remote" / "remote_psi0p3_seed32.tfevents",
    ),
}


def event_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        path.glob("events.out.tfevents.*"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file under {path}")
    return candidates[0]


def load_events(path: Path):
    source = event_file(path)
    accumulator = EventAccumulator(str(source), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"{METRIC!r} is missing from {source}")
    return source, accumulator.Scalars(METRIC)


def main() -> None:
    summaries = []
    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    for key, (label, color, linestyle, source_path) in RUNS.items():
        source, events = load_events(source_path)
        x = [item.step / 1_000_000 for item in events]
        y = [item.value for item in events]
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            alpha=0.95,
            label=f"{label} | final={y[-1]:,.0f} @ {x[-1]:.2f}M",
        )
        recent = events[-min(25, len(events)) :]
        dt_seconds = recent[-1].wall_time - recent[0].wall_time
        ds = recent[-1].step - recent[0].step
        fps = ds / dt_seconds if dt_seconds > 0 else None
        remaining = max(TOTAL_STEPS - events[-1].step, 0)
        eta = events[-1].wall_time + remaining / fps if fps and fps > 0 else None
        summaries.append(
            {
                "key": key,
                "label": label,
                "source": str(source),
                "metric": METRIC,
                "points": len(events),
                "last_step": int(events[-1].step),
                "progress_fraction": float(events[-1].step / TOTAL_STEPS),
                "latest_value": float(events[-1].value),
                "recent_steps_per_second": float(fps) if fps else None,
                "last_scalar_time": datetime.fromtimestamp(events[-1].wall_time, TZ).isoformat(),
                "estimated_finish": datetime.fromtimestamp(eta, TZ).isoformat() if eta else None,
            }
        )

    ax.set_title("Association threshold comparison: seed 2 vs seed 32", pad=10)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(0, TOTAL_STEPS / 1_000_000)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.4, framealpha=0.96)
    fig.subplots_adjust(bottom=0.34, left=0.12, right=0.98, top=0.91)
    fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    payload = {
        "generated_at": datetime.now(TZ).isoformat(),
        "metric": METRIC,
        "total_steps": TOTAL_STEPS,
        "curves": summaries,
        "output": str(HERE / "plot.png"),
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
