"""Plot the fixed-layout actor-message ablation.

The remote event files are refreshed local snapshots.  Re-run the refresh
step before rerunning this script if the remote runs have progressed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REMOTE_DATA = HERE / "data" / "remote"
RESULTS = PROJECT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
METRIC = "agent0/system_performance_true_all_GUs"
TOTAL_STEPS = 60_000_000

# Okabe-Ito palette: suitable for common red-green colour-vision deficiencies.
COLORS = {"R0": "#0072B2", "R260": "#E69F00", "R520": "#009E73"}

REMOTE_FILES = {
    "R0": REMOTE_DATA / "R0.tfevents",
    "R260": REMOTE_DATA / "R260.tfevents",
    "R520": REMOTE_DATA / "R520.tfevents",
}
LOCAL_EXPERIMENT = (
    "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_"
    "vmax30_seed2_60m_20260810"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def latest_local_event() -> Path:
    log_dir = RESULTS / LOCAL_EXPERIMENT / "run1" / "logs"
    paths = sorted(log_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event file found in {log_dir}")
    return paths[0]


def load_events(path: Path, metric: str) -> list[Any]:
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if metric not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"Metric {metric!r} is absent from {path}")
    events = accumulator.Scalars(metric)
    if not events:
        raise ValueError(f"Metric {metric!r} has no values in {path}")
    return events


def make_curve(label: str, path: Path, metric: str, color: str, linestyle: str = "-") -> dict[str, Any]:
    return {
        "label": label,
        "path": path,
        "color": color,
        "linestyle": linestyle,
        "events": load_events(path, metric),
    }


def summarize(curve: dict[str, Any], metric: str) -> dict[str, Any]:
    events = curve["events"]
    first, last = events[0], events[-1]
    recent = events[-min(25, len(events)):]
    dt = recent[-1].wall_time - recent[0].wall_time
    ds = recent[-1].step - recent[0].step
    fps = ds / dt if dt > 0 else None
    remaining = max(TOTAL_STEPS - last.step, 0)
    eta = last.wall_time + remaining / fps if fps and fps > 0 else None
    path = Path(curve["path"])
    return {
        "label": curve["label"],
        "path": str(path),
        "metric": metric,
        "points": len(events),
        "first_step": first.step,
        "last_step": last.step,
        "progress_fraction": last.step / TOTAL_STEPS,
        "last_value": last.value,
        "recent_fps_steps_per_sec": fps,
        "last_scalar_time": datetime.fromtimestamp(last.wall_time).astimezone().isoformat(),
        "source_file_mtime": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
        "estimated_remaining_hours": remaining / fps / 3600 if fps and fps > 0 else None,
        "estimated_finish": datetime.fromtimestamp(eta).astimezone().isoformat() if eta else None,
    }


def plot_curves(curves: list[dict[str, Any]], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for curve in curves:
        events = curve["events"]
        steps = [event.step / 1_000_000 for event in events]
        values = [event.value for event in events]
        final = values[-1]
        label = f"{curve['label']} | final={final:,.0f} @ {steps[-1]:.1f}M"
        ax.plot(
            steps,
            values,
            label=label,
            color=curve["color"],
            linestyle=curve["linestyle"],
            linewidth=1.8,
            alpha=0.95,
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(METRIC)
    ax.set_xlim(left=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    # Put the requested lower-right legend below the axes so it cannot cover data.
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, -0.04),
        frameon=True,
        framealpha=0.96,
        fontsize=8.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.30, left=0.12, right=0.98, top=0.91)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = args.metric

    remote_curves = [
        make_curve(f"{radius} | no_actor_message", path, metric, COLORS[radius])
        for radius, path in REMOTE_FILES.items()
    ]
    local_curve = make_curve(
        "R520 | actor_message",
        latest_local_event(),
        metric,
        COLORS["R520"],
        linestyle="--",
    )

    remote_path = output_dir / "remote_no_actor_radii.png"
    four_way_path = output_dir / "no_actor_vs_actor_r520.png"
    r520_two_way_path = output_dir / "r520_no_actor_vs_actor.png"
    plot_curves(
        remote_curves,
        remote_path,
        "Fixed 600 x 600 | noise=3 | no actor_message | R0/R260/R520",
    )
    plot_curves(
        [*remote_curves, local_curve],
        four_way_path,
        "Fixed 600 x 600 | noise=3 | actor_message ablation",
    )
    r520_remote_curve = next(
        curve for curve in remote_curves if curve["label"].startswith("R520 |")
    )
    local_curve_two_way = {**local_curve, "color": "#CC79A7"}
    plot_curves(
        [r520_remote_curve, local_curve_two_way],
        r520_two_way_path,
        "Fixed 600 x 600 | noise=3 | R520 actor_message comparison",
    )
    # Keep the conventional paper-figures filename as an alias of the main
    # four-curve comparison while retaining the two descriptive filenames.
    shutil.copyfile(four_way_path, output_dir / "plot.png")

    curves = [*remote_curves, local_curve]
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "metric": metric,
        "total_steps": TOTAL_STEPS,
        "remote_snapshot_note": "Remote event files are local snapshots from the refresh manifest.",
        "curves": [summarize(curve, metric) for curve in curves],
        "outputs": [
            str(remote_path),
            str(four_way_path),
            str(r520_two_way_path),
            str(output_dir / "plot.png"),
        ],
    }
    (output_dir / "curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
