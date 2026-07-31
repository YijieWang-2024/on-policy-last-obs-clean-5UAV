from __future__ import annotations

import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "onpolicy" / "scripts" / "results" / "mec" / "mappo"

RUNS = {
    "DC-PPO (50M+50M)": {
        "status": "complete",
        "family": "earlier baseline",
        "segments": [
            ("regional_dynamic_strict1p5_init110_220_330_440_dcppo_gae095_seed2_50m", 0),
            ("regional_dynamic_strict1p5_init110_220_330_440_dcppo_gae095_seed2_warm50to100m", 50_000_000),
        ],
    },
    "MAPPO (50M+50M)": {
        "status": "complete",
        "family": "earlier baseline",
        "segments": [
            ("regional_dynamic_strict1p5_init110_220_330_440_mappo_gae095_seed2_50m", 0),
            ("regional_dynamic_strict1p5_init110_220_330_440_mappo_gae095_seed2_warm50to100m", 50_000_000),
        ],
    },
    "Cartesian MAPPO": {
        "status": "stopped diagnostic",
        "family": "earlier diagnostic",
        "segments": [("regional_dynamic_strict1p5_cartesian_mappo_seed2_100m_20260727", 0)],
    },
    "E1: spatial": {
        "status": "stopped diagnostic",
        "family": "mechanism experiment",
        "segments": [("regional_dynamic_strict1p5_cartesian_spatialflight_e1_seed2_100m_r64_20260727", 0)],
    },
    "E2: spatial + curriculum": {
        "status": "complete",
        "family": "mechanism experiment",
        "segments": [("regional_dynamic_strict1p5_cartesian_spatialflight_curriculum_e2_seed2_100m_r64_20260727", 0)],
    },
    "B0: standard + curriculum": {
        "status": "running snapshot",
        "family": "current ablation",
        "segments": [("regional_dynamic_strict1p5_cartesian_curriculum_distance_b0_seed2_100m_r64_20260728", 0)],
    },
    "B1: B0 + neighbor positions": {
        "status": "running snapshot",
        "family": "current ablation",
        "segments": [("regional_dynamic_strict1p5_cartesian_curriculum_distance_neighbor_b1_seed2_100m_r64_20260728", 0)],
    },
}

TAGS = {
    "reward": "agent0/cumulative_reward",
    "true60": "agent0/system_performance_true_all_GUs",
    "equivalent60": "agent0/system_performance_equivalent_full_GUs",
    "system_performance": "agent0/system_performance",
    "admission": "agent0/md_admission_ratio",
    "upper_admission": "agent0/md_admission_ratio_upper_right",
    "completion": "agent0/complete_task_ratio",
    "active_mds": "agent0/average_active_mds",
    "projection": "agent0/flight_disk_projection_ratio",
}


def load_accumulator(run_name: str) -> EventAccumulator:
    log_dir = RESULTS / run_name / "run1" / "logs"
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def scalar_map(accumulator: EventAccumulator, tag: str, offset: int) -> dict[int, float]:
    return {int(item.step + offset): float(item.value) for item in accumulator.Scalars(tag)}


curve_rows: list[dict[str, object]] = []
manifest_rows: list[dict[str, object]] = []

for experiment, definition in RUNS.items():
    values_by_metric = {metric: {} for metric in TAGS}
    source_names = []
    for run_name, offset in definition["segments"]:
        source_names.append(run_name)
        accumulator = load_accumulator(run_name)
        for metric, tag in TAGS.items():
            if tag in accumulator.Tags()["scalars"]:
                values_by_metric[metric].update(scalar_map(accumulator, tag, offset))

    max_step = max(max(values) for values in values_by_metric.values() if values)
    manifest_rows.append({
        "experiment": experiment,
        "family": definition["family"],
        "status": definition["status"],
        "max_step": max_step,
        "source_runs": " | ".join(source_names),
    })
    for metric, values in values_by_metric.items():
        for step, value in sorted(values.items()):
            curve_rows.append({
                "experiment": experiment,
                "step": step,
                "metric": metric,
                "value": value,
            })

with (HERE / "training_curves.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["experiment", "step", "metric", "value"])
    writer.writeheader()
    writer.writerows(curve_rows)

with (HERE / "run_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["experiment", "family", "status", "max_step", "source_runs"])
    writer.writeheader()
    writer.writerows(manifest_rows)

tail_rows: list[dict[str, object]] = []
for experiment in RUNS:
    row: dict[str, object] = {"experiment": experiment}
    manifest = next(item for item in manifest_rows if item["experiment"] == experiment)
    row.update({"status": manifest["status"], "max_step": manifest["max_step"]})
    for metric in TAGS:
        points = [item for item in curve_rows if item["experiment"] == experiment and item["metric"] == metric]
        points.sort(key=lambda item: int(item["step"]))
        tail = points[-100:]
        row[metric] = "" if not tail else sum(float(item["value"]) for item in tail) / len(tail)
    tail_rows.append(row)

tail_fields = ["experiment", "status", "max_step", *TAGS]
with (HERE / "tail100_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=tail_fields)
    writer.writeheader()
    writer.writerows(tail_rows)

print(f"wrote {len(curve_rows):,} curve rows and {len(tail_rows)} tail summaries to {HERE}")
