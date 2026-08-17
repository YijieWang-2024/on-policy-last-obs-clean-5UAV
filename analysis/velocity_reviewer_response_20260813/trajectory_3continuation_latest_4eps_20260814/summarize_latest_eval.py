#!/usr/bin/env python
"""Summarize the four latest-checkpoint evaluation episodes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
TAIL_SLOTS = 100


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    catalog = json.loads((ROOT / "trajectory_catalog.json").read_text(encoding="utf-8"))
    detail = []
    episode_metrics = []
    for record in catalog:
        data = np.load(record["episode_data"])
        positions = np.asarray(data["uav_positions"], dtype=float)
        tail = positions[-TAIL_SLOTS:]
        center = tail.mean(axis=0)
        distances = np.linalg.norm(tail - center[None, :, :], axis=2)
        step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=2)
        for uav in range(positions.shape[1]):
            detail.append({
                "label": record["label"],
                "episode": record["episode_index"],
                "evaluation_seed": record["evaluation_seed"],
                "training_step": record["training_step"],
                "uav": uav + 1,
                "initial_x_m": positions[0, uav, 0],
                "initial_y_m": positions[0, uav, 1],
                "final_x_m": positions[-1, uav, 0],
                "final_y_m": positions[-1, uav, 1],
                "tail_center_x_m": center[uav, 0],
                "tail_center_y_m": center[uav, 1],
                "tail_r95_m": np.percentile(distances[:, uav], 95),
                "tail_rmax_m": distances[:, uav].max(),
                "tail_mean_step_distance_m": step_distances[-TAIL_SLOTS:, uav].mean(),
                "total_path_length_m": step_distances[:, uav].sum(),
            })
        episode_metrics.append({
            "label": record["label"],
            "episode": record["episode_index"],
            "evaluation_seed": record["evaluation_seed"],
            "training_step": record["training_step"],
            "true_all_GUs": record["true_all_GUs"],
            "complete_task_ratio": record["complete_task_ratio"],
            "md_admission_ratio": record["md_admission_ratio"],
        })

    detail_path = ROOT / "deployment_tail_summary.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    metrics_path = ROOT / "episode_metrics_summary.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_metrics[0]))
        writer.writeheader()
        writer.writerows(episode_metrics)

    lines = [
        "# Latest continuation checkpoint evaluation summary",
        "",
        f"Four deterministic evaluation episodes were run per speed (seeds 1001--1004). The tail diagnostic uses the last {TAIL_SLOTS} slots.",
        "The tail center is descriptive: it is used to assess whether the policy repeatedly remains in a compact deployment region, not as a new training target.",
        "",
        "| speed / training seed | checkpoint total steps | performance mean +/- SD | completion mean +/- SD | MD admission mean +/- SD | mean tail r95 (m) | max tail r95 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in sorted({row["label"] for row in episode_metrics}):
        ms = [row for row in episode_metrics if row["label"] == label]
        ds = [row for row in detail if row["label"] == label]
        perf = mean_sd([row["true_all_GUs"] for row in ms])
        comp = mean_sd([row["complete_task_ratio"] for row in ms])
        adm = mean_sd([row["md_admission_ratio"] for row in ms])
        r95 = [row["tail_r95_m"] for row in ds]
        step = ms[0]["training_step"]
        vmax_seed = label.replace("vmax", "v_max=").replace("_seed", ", seed=")
        lines.append(
            f"| {vmax_seed} | {step / 1e6:.3f}M | {perf[0]:.1f} +/- {perf[1]:.1f} | "
            f"{comp[0]:.4f} +/- {comp[1]:.4f} | {adm[0]:.4f} +/- {adm[1]:.4f} | "
            f"{np.mean(r95):.2f} | {np.max(r95):.2f} |"
        )
    lines += [
        "",
        "Interpretation: all three policies form five repeatable deployment clusters in the 600 m x 600 m area. The 20 m/s policy has visibly longer terminal wandering for UAV 2/UAV 5 and lower evaluation performance, while 30/40 m/s are more compact and higher-performing in this four-episode snapshot. This is descriptive evidence, not a cross-seed statistical claim.",
    ]
    (ROOT / "latest_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
