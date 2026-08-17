"""Reproducibly plot the six live curriculum runs plus the requested reference.

The script deliberately reads the event files copied into ``data``.  It does
not touch live training directories, so it can be rerun while training is in
progress after refreshing those copies.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT_PNG = HERE / "training_curves_7.png"
OUT_CSV = HERE / "curve_summary.csv"
OUT_JSON = HERE / "curve_summary.json"
TAG = "agent0/system_performance_true_all_GUs"

RUNS = [
    ("Local ψ=0.1, seed=2 | curriculum ON", DATA / "local" / "psi0p1_seed2.tfevents", "#0072B2", "-", "live"),
    ("Local ψ=0.3, seed=2 | curriculum ON", DATA / "local" / "psi0p3_seed2.tfevents", "#D55E00", "-", "live"),
    ("Local ψ=0.1, seed=32 | curriculum ON", DATA / "local" / "psi0p1_seed32.tfevents", "#009E73", "-", "live"),
    ("Remote ψ=0.5, seed=2 | curriculum ON", DATA / "remote" / "psi0p5_seed2.tfevents", "#E69F00", "-", "live"),
    ("Remote ψ=0.7, seed=2 | curriculum ON", DATA / "remote" / "psi0p7_seed2.tfevents", "#56B4E9", "-", "live"),
    ("Remote ψ=0.9, seed=2 | curriculum ON", DATA / "remote" / "psi0p9_seed2.tfevents", "#CC79A7", "-", "live"),
    ("Remote ψ=0.3, seed=42 | curriculum OFF reference", DATA / "remote" / "psi0p3_seed42_baseline.tfevents", "#000000", "--", "reference"),
]


def load_scalars(path: Path):
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    tag = TAG if TAG in tags else next((t for t in tags if t.endswith("/system_performance_true_all_GUs")), None)
    if tag is None:
        raise RuntimeError(f"{path} has no system-performance scalar; available tags: {tags[:20]}")
    points = sorted(acc.Scalars(tag), key=lambda x: x.step)
    if not points:
        raise RuntimeError(f"{path} has no points for {tag}")
    return points, tag


def estimate_speed(points):
    if len(points) < 2:
        return math.nan
    tail = points[-min(20, len(points)):]
    dt = tail[-1].wall_time - tail[0].wall_time
    ds = tail[-1].step - tail[0].step
    return ds / dt if dt > 0 else math.nan


def main():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    })

    summaries = []
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for label, path, color, linestyle, kind in RUNS:
        if not path.exists():
            raise FileNotFoundError(path)
        points, actual_tag = load_scalars(path)
        steps = [p.step for p in points]
        values = [p.value for p in points]
        latest = points[-1]
        speed = estimate_speed(points)
        legend_label = f"{label} | {latest.step/1e6:.2f}M, {latest.value:,.0f}"
        ax.plot([s / 1e6 for s in steps], values, color=color, linestyle=linestyle,
                linewidth=1.8 if kind == "live" else 2.1, label=legend_label,
                alpha=0.95 if kind == "live" else 0.9)
        summaries.append({
            "label": label,
            "kind": kind,
            "event_file": str(path),
            "tag": actual_tag,
            "points": len(points),
            "latest_step": int(latest.step),
            "latest_value": float(latest.value),
            "latest_wall_time": float(latest.wall_time),
            "recent_steps_per_second": None if math.isnan(speed) else float(speed),
        })

    ax.set_title("Association-threshold training comparison (curriculum runs)")
    ax.set_xlabel("Environment steps (million)")
    ax.set_ylabel("system_performance_true_all_GUs")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", framealpha=0.92, borderpad=0.6, handlelength=2.8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    OUT_JSON.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    # PowerShell may expose a legacy code page; keep console output portable.
    print(json.dumps({"png": str(OUT_PNG), "summary": summaries}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
