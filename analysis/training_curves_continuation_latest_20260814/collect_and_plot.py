#!/usr/bin/env python
"""Collect and plot initial+continuation training curves on one total-step axis."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
METRIC = "agent0/system_performance_true_all_GUs"

SPECS = {
    "vmax20_seed2": {
        "v_max": 20,
        "seed": 2,
        "base_run": RESULTS / (
            "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
            "md12_vmax20_seed2_60m_20260812/run1"
        ),
        "continuation_run": RESULTS / (
            "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
            "md12_vmax20_seed2_resume40m_20260814/run1"
        ),
    },
    "vmax30_seed32": {
        "v_max": 30,
        "seed": 32,
        "base_run": RESULTS / (
            "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
            "md12_vmax30_seed32_60m_20260811_retry2/run1"
        ),
        "continuation_run": RESULTS / (
            "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
            "md12_vmax30_seed32_resume40m_20260814/run1"
        ),
    },
    "vmax40_seed32": {
        "v_max": 40,
        "seed": 32,
        "base_run": RESULTS / (
            "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
            "md12_vmax40_seed32_60m_20260812/run1"
        ),
        "continuation_run": RESULTS / (
            "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
            "md12_vmax40_seed32_resume40m_20260814/run1"
        ),
    },
}


def latest_event(run_dir: Path) -> Path:
    paths = sorted((run_dir / "logs").glob("events.out.tfevents*"))
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event file under {run_dir / 'logs'}")
    return paths[-1]


def manifest(run_dir: Path) -> dict:
    path = run_dir / "models" / "checkpoint_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def stable_copy(source: Path, destination: Path) -> None:
    """Copy a live event file only when its size/mtime are stable."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        shutil.copy2(source, destination)
        after = (source.stat().st_size, source.stat().st_mtime_ns)
        if before == after:
            return
    raise RuntimeError(f"Event file kept changing while snapshotting: {source}")


def load_curve(event_path: Path) -> tuple[list[int], list[float]]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"Metric {METRIC!r} is absent from {event_path}")
    events = accumulator.Scalars(METRIC)
    steps = [int(event.step) for event in events]
    values = [float(event.value) for event in events]
    # TensorBoard files can contain repeated steps after a restart; preserve
    # the final value for each step and keep the curve monotone.
    dedup = {}
    for step, value in zip(steps, values):
        dedup[step] = value
    steps = sorted(dedup)
    return steps, [dedup[step] for step in steps]


def collect() -> None:
    data_dir = HERE / "data"
    event_dir = data_dir / "event_snapshots"
    data_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = {"metric": METRIC, "series": {}}

    for label, spec in SPECS.items():
        base_run = spec["base_run"]
        continuation_run = spec["continuation_run"]
        base_manifest = manifest(base_run)
        continuation_manifest = manifest(continuation_run)
        base_event = latest_event(base_run)
        continuation_event = latest_event(continuation_run)
        base_snapshot = event_dir / f"{label}_initial.tfevents"
        continuation_snapshot = event_dir / f"{label}_continuation.tfevents"
        stable_copy(base_event, base_snapshot)
        stable_copy(continuation_event, continuation_snapshot)
        base_steps, base_values = load_curve(base_snapshot)
        continuation_steps, continuation_values = load_curve(continuation_snapshot)
        offset = int(continuation_manifest["source_total_num_steps"])
        join_step = offset

        for step, value in zip(base_steps, base_values):
            rows.append({
                "series": label,
                "v_max": spec["v_max"],
                "seed": spec["seed"],
                "phase": "initial",
                "local_step": step,
                "total_step": step,
                "performance": value,
                "join_step": join_step,
                "event_snapshot": str(base_snapshot),
            })
        for step, value in zip(continuation_steps, continuation_values):
            rows.append({
                "series": label,
                "v_max": spec["v_max"],
                "seed": spec["seed"],
                "phase": "continuation",
                "local_step": step,
                "total_step": offset + step,
                "performance": value,
                "join_step": join_step,
                "event_snapshot": str(continuation_snapshot),
            })
        metadata["series"][label] = {
            "v_max": spec["v_max"],
            "seed": spec["seed"],
            "base_run": str(base_run),
            "continuation_run": str(continuation_run),
            "base_event_snapshot": str(base_snapshot),
            "continuation_event_snapshot": str(continuation_snapshot),
            "source_total_num_steps": offset,
            "base_manifest_total_num_steps": int(base_manifest["total_num_steps"]),
            "continuation_manifest_total_num_steps": int(continuation_manifest["total_num_steps"]),
            "base_points": len(base_steps),
            "continuation_points": len(continuation_steps),
            "base_last_logged_step": base_steps[-1],
            "continuation_last_logged_step": continuation_steps[-1],
            "absolute_last_logged_step": offset + continuation_steps[-1],
            "join_step": join_step,
        }

    rows.sort(key=lambda row: (row["v_max"], row["total_step"], row["phase"]))
    with (data_dir / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (data_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    collect()
