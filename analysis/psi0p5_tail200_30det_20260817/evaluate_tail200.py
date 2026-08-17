"""Evaluate the latest remote psi=0.5 checkpoint on the final 200 slots."""

from __future__ import annotations

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
PER_UAV_SCRIPT = PROJECT_ROOT / "analysis" / "psi0p5_per_uav_30det_20260817" / "evaluate_psi0p5_per_uav.py"
CHECKPOINT_DIR = PROJECT_ROOT / "analysis" / "psi0p5_per_uav_30det_20260817" / "remote_snapshots" / "remote_psi0p5" / "checkpoint_snapshot"

UAV_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
UAV_HATCHES = ["", "//", "..", "xx", "\\\\"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=7001)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results_tail200_30det_latest",
    )
    return parser.parse_args()


def write_protocol(output_dir: Path, seeds: list[int], checkpoint_step: int, steps: int, start: int) -> None:
    measured = steps - start
    text = f"""# psi=0.5 deterministic tail-window evaluation

The latest manifest-safe remote checkpoint at {checkpoint_step:,} environment
steps was evaluated deterministically on {len(seeds)} common seeds
{seeds[0]}--{seeds[-1]}.

Every evaluation episode still executes all {steps} environment slots.  The
task, resource, reward, system-performance, and per-UAV service metrics in
this directory are accumulated only for zero-based steps {start}--{steps - 1}
({measured} slots).  This is a warm-up-free comparison of the final part of
each episode.

For each measured step, an active MD is counted as served by UAV i when the
post-transformation association matrix has a positive entry at row i for
that active MD.  Thus `service_md_count / {measured}` is the average number
of active MD service assignments per slot for that UAV in the tail window.
Successful service is counted from the environment's post-step
`complete_task` flag.  The final-position deployment fields remain endpoint
diagnostics after slot {steps - 1}; they are not tail-window averages.

The cumulative system-performance, delay, and energy values are computed from
before/after deltas of the environment's cumulative counters, so they cannot
include the first {start} warm-up slots.  Dynamic-MD admission counts use the
same tail-window delta when the environment counters are available.
"""
    (output_dir / "metric_protocol.md").write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(output_dir: Path, rows: list[dict], tail_slots: int) -> None:
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

    bars(axes[0, 0], "service_md_count", f"Served active MDs in final {tail_slots} slots", "MD count")
    bars(axes[0, 1], "successful_service_md_count", "On-time served MDs in tail", "MD count")
    bars(axes[0, 2], "service_success_rate", "Per-UAV tail service success rate", "rate (%)", 100.0)
    bars(axes[1, 0], "episode_reward", "Per-UAV tail reward", "reward")
    bars(axes[1, 1], "mean_reward", "Per-UAV tail mean reward", "reward / slot")
    bars(axes[1, 2], "system_performance_individual", "Per-UAV tail system performance", "performance")
    fig.suptitle(
        f"psi=0.5 | seed=2 | latest checkpoint | 30 deterministic episodes | final {tail_slots} slots",
        fontsize=14,
    )
    fig.savefig(output_dir / "psi0p5_tail200_per_uav_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    if cli.episodes < 20:
        raise ValueError("Use at least 20 common test episodes")
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    if not (CHECKPOINT_DIR / "args.json").is_file():
        raise FileNotFoundError(f"Missing remote checkpoint snapshot: {CHECKPOINT_DIR}")

    cli.output_dir.mkdir(parents=True)
    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    base = load_module(BASE_SCRIPT, "psi_reviewer_metrics_base_tail200")
    per_uav = load_module(PER_UAV_SCRIPT, "psi0p5_per_uav_helpers_tail200")
    torch.set_num_threads(1)

    checkpoint_step = base.checkpoint_manifest_step(CHECKPOINT_DIR)
    if checkpoint_step is None:
        raise RuntimeError(f"Missing verified manifest: {CHECKPOINT_DIR}")

    args, env, actors, normers = base.load_policy(CHECKPOINT_DIR, CHECKPOINT_DIR)
    steps = int(args.episode_length)
    measurement_start_step = steps // 2
    write_protocol(cli.output_dir, seeds, checkpoint_step, steps, measurement_start_step)

    rows = []
    try:
        for index, seed in enumerate(seeds, 1):
            row = base.evaluate_episode(
                args,
                env,
                actors,
                normers,
                seed,
                checkpoint_step,
                measurement_start_step=measurement_start_step,
            )
            row["model"] = "psi0p5_seed2_remote_tail200"
            rows.append(row)
            if index == 1 or index % 5 == 0 or index == len(seeds):
                print(json.dumps({"episode": index, "episodes": len(seeds)}), flush=True)
    finally:
        env.close()

    scalar_keys = ["model", "checkpoint_step", "evaluation_seed", *base.METRICS]
    scalar_rows = [{key: row[key] for key in scalar_keys} for row in rows]
    overall = base.aggregate(rows)
    per_uav_rows = per_uav.make_per_uav_rows(rows, int(args.n_UAVs))
    for row in per_uav_rows:
        row["model"] = "psi0p5_seed2_remote_tail200"
    per_uav_overall = per_uav.aggregate_per_uav(per_uav_rows)
    for row in per_uav_overall:
        row["model"] = "psi0p5_seed2_remote_tail200"

    write_csv(cli.output_dir / "episode_metrics.csv", scalar_rows)
    write_csv(cli.output_dir / "aggregate_metrics.csv", overall)
    write_csv(cli.output_dir / "per_uav_episode_metrics.csv", per_uav_rows)
    write_csv(cli.output_dir / "per_uav_aggregate.csv", per_uav_overall)
    make_plot(cli.output_dir, per_uav_overall, steps - measurement_start_step)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "episodes": cli.episodes,
        "test_seeds": seeds,
        "deterministic": True,
        "measurement_start_step": measurement_start_step,
        "measurement_end_step_exclusive": steps,
        "measurement_slots": steps - measurement_start_step,
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
        "measurement_slots": steps - measurement_start_step,
        "rows": len(rows),
        "per_uav_rows": len(per_uav_rows),
        "plot": str(cli.output_dir / "psi0p5_tail200_per_uav_summary.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
