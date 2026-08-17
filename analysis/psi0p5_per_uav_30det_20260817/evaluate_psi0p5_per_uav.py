"""Evaluate the latest remote psi=0.5 checkpoint with UAV-wise diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "psi_reviewer_final_metrics_20260816" / "evaluate_final_checkpoints.py"
CHECKPOINT_DIR = HERE / "remote_snapshots" / "remote_psi0p5" / "checkpoint_snapshot"

UAV_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
UAV_HATCHES = ["", "//", "..", "xx", "\\\\"]
PER_UAV_FIELDS = (
    "service_md_count",
    "successful_service_md_count",
    "service_success_rate",
    "service_steps",
    "service_step_fraction",
    "allocated_bandwidth",
    "useful_bandwidth",
    "allocated_cpu",
    "useful_cpu",
    "episode_reward",
    "mean_reward",
    "system_performance_individual",
    "cumulative_individual_reward",
)


def load_base():
    spec = importlib.util.spec_from_file_location("psi_reviewer_metrics_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=7001)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results_30det_latest_per_uav",
    )
    return parser.parse_args()


def write_protocol(output_dir: Path, seeds: list[int], checkpoint_step: int) -> None:
    text = f"""# psi=0.5 UAV-wise deterministic evaluation

The latest manifest-safe remote checkpoint at {checkpoint_step:,} environment
steps was evaluated deterministically on {len(seeds)} common seeds
{seeds[0]}--{seeds[-1]}.

For each time step, an active MD is counted as served by UAV i when the
post-transformation association matrix has a positive entry at row i for
that active MD.  A served MD is counted as successful for UAV i when the
environment complete_task flag is true after the step.  Therefore the
per-UAV service count is an association/load statistic, not a count of all
MDs visible in the UAV observation.

- `service_md_count`: served active MD assignments for one UAV in one episode;
- `successful_service_md_count`: those assignments completed on time;
- `service_success_rate`: successful service count divided by service count;
- `episode_reward`: sum of the per-UAV reward returned by env.step;
- `system_performance_individual`: the environment's accumulated UAV-view
  performance array, i.e. R_task_delay + R_task_energy + R_fly_energy;
- `cumulative_individual_reward`: the environment's final cumulative local
  reward array, retained as a cross-check of the per-step reward sum.

`system_performance_true_all_GUs` remains a global metric.  It is not split
among UAVs by the environment and is therefore not reported as a per-UAV
quantity.
"""
    (output_dir / "metric_protocol.md").write_text(text, encoding="utf-8")


def make_per_uav_rows(rows: list[dict], n_uavs: int) -> list[dict]:
    result = []
    for row in rows:
        arrays = {
            "service_md_count": row["per_uav_offloaded_task_count"],
            "successful_service_md_count": row["per_uav_successful_offloaded_task_count"],
            "service_success_rate": row["per_uav_service_success_rate"],
            "service_steps": row["per_uav_service_steps"],
            "service_step_fraction": row["per_uav_service_step_fraction"],
            "allocated_bandwidth": row["per_uav_allocated_bandwidth"],
            "useful_bandwidth": row["per_uav_useful_bandwidth"],
            "allocated_cpu": row["per_uav_allocated_cpu"],
            "useful_cpu": row["per_uav_useful_cpu"],
            "episode_reward": row["per_uav_episode_reward"],
            "mean_reward": row["per_uav_mean_reward"],
            "system_performance_individual": row["per_uav_system_performance_individual"],
            "cumulative_individual_reward": row["per_uav_cumulative_individual_reward"],
        }
        for uav_id in range(n_uavs):
            item = {
                "model": "psi0p5_seed2_remote",
                "evaluation_seed": row["evaluation_seed"],
                "checkpoint_step": row["checkpoint_step"],
                "uav_id": uav_id,
            }
            item.update({name: float(values[uav_id]) for name, values in arrays.items()})
            result.append(item)
    return result


def aggregate_per_uav(rows: list[dict]) -> list[dict]:
    result = []
    for uav_id in sorted({int(row["uav_id"]) for row in rows}):
        selected = [row for row in rows if int(row["uav_id"]) == uav_id]
        item = {
            "model": "psi0p5_seed2_remote",
            "uav_id": uav_id,
            "episodes": len(selected),
        }
        for field in PER_UAV_FIELDS:
            values = np.asarray([float(row[field]) for row in selected], dtype=float)
            values = values[np.isfinite(values)]
            item[f"{field}_n"] = int(values.size)
            item[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
            item[f"{field}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            item[f"{field}_se"] = item[f"{field}_std"] / np.sqrt(values.size) if values.size > 1 else 0.0
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(output_dir: Path, rows: list[dict]) -> None:
    x = np.arange(len(rows))
    labels = [f"UAV {int(row['uav_id'])}" for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    def bars(ax, field, title, ylabel, scale=1.0):
        means = [row[f"{field}_mean"] * scale for row in rows]
        errors = [row[f"{field}_std"] * scale for row in rows]
        ax.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=UAV_COLORS,
            hatch=UAV_HATCHES,
            edgecolor="#333333",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    bars(axes[0, 0], "service_md_count", "Served active MDs per episode", "MD count")
    bars(axes[0, 1], "successful_service_md_count", "On-time served MDs per episode", "MD count")
    bars(axes[0, 2], "service_success_rate", "Per-UAV service success rate", "rate (%)", 100.0)
    bars(axes[1, 0], "episode_reward", "Per-UAV episode reward", "reward")
    bars(axes[1, 1], "mean_reward", "Per-UAV mean step reward", "reward / step")
    bars(axes[1, 2], "system_performance_individual", "Per-UAV system performance", "performance")
    fig.suptitle("psi=0.5 | seed=2 | latest checkpoint | 30 deterministic episodes", fontsize=14)
    fig.savefig(output_dir / "psi0p5_per_uav_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    if cli.episodes < 20:
        raise ValueError("Use at least 20 common test episodes")
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    if not (CHECKPOINT_DIR / "args.json").is_file():
        raise FileNotFoundError(
            f"Missing remote checkpoint snapshot: {CHECKPOINT_DIR}. Run refresh first."
        )
    cli.output_dir.mkdir(parents=True)
    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    base = load_base()
    torch.set_num_threads(1)

    checkpoint_step = base.checkpoint_manifest_step(CHECKPOINT_DIR)
    if checkpoint_step is None:
        raise RuntimeError(f"Missing verified manifest: {CHECKPOINT_DIR}")
    write_protocol(cli.output_dir, seeds, checkpoint_step)
    args, env, actors, normers = base.load_policy(CHECKPOINT_DIR, CHECKPOINT_DIR)
    rows = []
    try:
        for index, seed in enumerate(seeds, 1):
            row = base.evaluate_episode(args, env, actors, normers, seed, checkpoint_step)
            row["model"] = "psi0p5_seed2_remote"
            rows.append(row)
            if index == 1 or index % 5 == 0 or index == len(seeds):
                print(json.dumps({"episode": index, "episodes": len(seeds)}), flush=True)
    finally:
        env.close()

    scalar_keys = ["model", "checkpoint_step", "evaluation_seed", *base.METRICS]
    scalar_rows = [{key: row[key] for key in scalar_keys} for row in rows]
    overall = base.aggregate(rows)
    per_uav_rows = make_per_uav_rows(rows, int(args.n_UAVs))
    per_uav_overall = aggregate_per_uav(per_uav_rows)
    write_csv(cli.output_dir / "episode_metrics.csv", scalar_rows)
    write_csv(cli.output_dir / "aggregate_metrics.csv", overall)
    write_csv(cli.output_dir / "per_uav_episode_metrics.csv", per_uav_rows)
    write_csv(cli.output_dir / "per_uav_aggregate.csv", per_uav_overall)
    make_plot(cli.output_dir, per_uav_overall)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "episodes": cli.episodes,
        "test_seeds": seeds,
        "deterministic": True,
        "run": "remote psi=0.5 | seed=2 | curriculum ON | deadline OFF | actor_message disabled",
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_step": checkpoint_step,
        "aggregate_metrics": overall,
        "per_uav_aggregate": per_uav_overall,
    }
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(cli.output_dir),
        "checkpoint_step": checkpoint_step,
        "rows": len(rows),
        "per_uav_rows": len(per_uav_rows),
        "plot": str(cli.output_dir / "psi0p5_per_uav_summary.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
