"""Evaluate the five curriculum-on psi checkpoints on common deterministic episodes."""

from __future__ import annotations

import argparse
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
RESULTS = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"

RUNS = {
    "psi0p1_seed2_local": {
        "label": "psi=0.1 | local | seed=2",
        "run_dir": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
        "remote": False,
    },
    "psi0p3_seed2_local": {
        "label": "psi=0.3 | local | seed=2",
        "run_dir": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p3_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
        "remote": False,
    },
    "psi0p5_seed2_remote": {
        "label": "psi=0.5 | remote | seed=2",
        "remote_key": "psi0p5",
        "remote_dir": HERE / "remote_snapshots" / "remote_psi0p5" / "checkpoint_snapshot",
        "remote": True,
    },
    "psi0p7_seed2_remote": {
        "label": "psi=0.7 | remote | seed=2",
        "remote_key": "psi0p7",
        "remote_dir": HERE / "remote_snapshots" / "remote_psi0p7" / "checkpoint_snapshot",
        "remote": True,
    },
    "psi0p9_seed2_remote": {
        "label": "psi=0.9 | remote | seed=2",
        "remote_key": "psi0p9",
        "remote_dir": HERE / "remote_snapshots" / "remote_psi0p9" / "checkpoint_snapshot",
        "remote": True,
    },
}

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
HATCHES = ["", "//", "..", "xx", "\\\\"]


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
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_30det_20260817")
    parser.add_argument(
        "--remote-snapshot-subdir",
        default="remote_snapshots",
        help="Subdirectory under this analysis folder containing refreshed remote snapshots.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actor actions instead of using deterministic distribution means",
    )
    return parser.parse_args()


def protocol(output_dir: Path, seeds: list[int], deterministic: bool) -> None:
    mode = "deterministic" if deterministic else "stochastic actor sampling"
    text = f"""# Curriculum-on psi reviewer metrics

Five checkpoints (curriculum ON, deadline filter OFF, actor_message disabled,
R520, noise=3, lifetime=12, mean MD velocity=3 m/s, seed=2) were evaluated
using **{mode}** on the same {len(seeds)} test seeds: {seeds[0]}--{seeds[-1]}.

For each step, `active_task_count` is the number of active MD tasks before
the action.  The transformed action partitions every active task into exactly
one of two branches:

- offloaded: the task has a nonzero association after action transformation;
- not offloaded/local: the task is not selected for offloading and therefore
  follows the environment's local-execution branch.

The explicit rates are:

- OffloadRatio = offloaded / active;
- task offloading success rate (OSR) = on-time completed offloaded / offloaded;
- OffloadYield = on-time completed offloaded / active;
- LocalShare = not-offloaded / active;
- local completion rate = on-time completed not-offloaded / not-offloaded;
- LocalYield = on-time completed not-offloaded / active;
- OverallCompletion = (on-time completed offloaded + on-time completed
  not-offloaded) / active.

The per-episode CSV retains the numerator and denominator counts.  Resource
allocation/useful/waste fields follow the current environment action path:
allocated is x times normalized allocation; useful is x times s times
normalized allocation; failed-offload waste is allocated resource assigned to
an offloaded task whose environment completion flag is false.  These are
code-level resource-usage measures and are reported alongside, not confused
with, the task success rates.
"""
    (output_dir / "metric_protocol.md").write_text(text, encoding="utf-8")


def _means(rows, metric, scale=1.0):
    return [row[f"{metric}_mean"] * scale for row in rows]


def _stds(rows, metric, scale=1.0):
    return [row[f"{metric}_std"] * scale for row in rows]


def make_plot(output_dir: Path, aggregate_rows: list[dict], deterministic: bool) -> None:
    labels = [RUNS[row["model"]]["label"].replace(" | ", "\n") for row in aggregate_rows]
    x = np.arange(len(aggregate_rows))
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5), constrained_layout=True)
    width = 0.24

    def grouped(ax, specs, title, ylabel="rate (%)", ylim=(0, 105)):
        for index, (metric, label, color, hatch) in enumerate(specs):
            offset = (index - (len(specs) - 1) / 2) * width
            ax.bar(
                x + offset,
                _means(aggregate_rows, metric, 100.0),
                width,
                yerr=_stds(aggregate_rows, metric, 100.0),
                capsize=3,
                color=color,
                hatch=hatch,
                edgecolor="#333333",
                linewidth=0.45,
                label=label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    gain = _means(aggregate_rows, "system_performance_true_all_GUs")
    gain_err = _stds(aggregate_rows, "system_performance_true_all_GUs")
    axes[0, 0].bar(x, gain, yerr=gain_err, capsize=4, color=COLORS, edgecolor="#333333")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0, 0].set_title("Overall system gain")
    axes[0, 0].set_ylabel("system_performance_true_all_GUs")
    axes[0, 0].grid(axis="y", alpha=0.25)

    grouped(
        axes[0, 1],
        (
            ("offload_ratio", "OffloadRatio", "#0072B2", ""),
            ("local_share", "LocalShare", "#E69F00", "//"),
            ("all_task_completion_rate", "OverallCompletion", "#009E73", ".."),
        ),
        "Active-task branch shares",
    )
    grouped(
        axes[0, 2],
        (
            ("task_offloading_success_rate", "OSR", "#D55E00", ""),
            ("local_completion_rate", "Local completion", "#CC79A7", "//"),
            ("all_task_completion_rate", "Overall completion", "#0072B2", ".."),
        ),
        "Conditional completion rates",
    )
    grouped(
        axes[0, 3],
        (
            ("offload_yield", "OffloadYield", "#0072B2", ""),
            ("local_yield", "LocalYield", "#E69F00", "//"),
            ("all_task_completion_rate", "OverallCompletion", "#009E73", ".."),
        ),
        "Successful-task yields over all active tasks",
    )

    def resource(ax, allocated, useful, title):
        ax.bar(x - width / 2, _means(aggregate_rows, allocated, 100.0), width,
               yerr=_stds(aggregate_rows, allocated, 100.0), capsize=3,
               color="#56B4E9", label="allocated")
        ax.bar(x + width / 2, _means(aggregate_rows, useful, 100.0), width,
               yerr=_stds(aggregate_rows, useful, 100.0), capsize=3,
               color="#009E73", hatch="//", label="useful")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("capacity fraction (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    resource(axes[1, 0], "bandwidth_allocated_utilization", "bandwidth_useful_utilization", "Bandwidth allocated vs useful")
    resource(axes[1, 1], "cpu_allocated_utilization", "cpu_useful_utilization", "CPU allocated vs useful")
    grouped(
        axes[1, 2],
        (
            ("bandwidth_failure_waste_ratio", "Bandwidth waste", "#D55E00", ""),
            ("cpu_failure_waste_ratio", "CPU waste", "#CC79A7", "//"),
        ),
        "Resource assigned to failed offloads",
        ylim=(0, 10),
    )
    grouped(
        axes[1, 3],
        (
            ("md_admission_ratio", "MD admission", "#0072B2", ""),
            ("deployment_1plus4_final", "1+4 final deployment", "#E69F00", "//"),
        ),
        "Admission and final deployment diagnostic",
    )

    mode = "deterministic" if deterministic else "stochastic actor sampling"
    fig.suptitle(
        f"Curriculum ON: association-threshold impact, psi=0.1–0.9, seed=2 ({mode})",
        fontsize=15,
    )
    fig.savefig(output_dir / "psi_curriculum_reviewer_metrics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    if cli.episodes < 20:
        raise ValueError("Use at least 20 common test episodes")
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    cli.output_dir.mkdir(parents=True)
    (cli.output_dir / "snapshots").mkdir()
    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    protocol(cli.output_dir, seeds, deterministic=not cli.stochastic)

    base = load_base()
    torch.set_num_threads(1)
    rows = []
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "episodes": cli.episodes,
        "test_seeds": seeds,
        "deterministic": not cli.stochastic,
        "runs": {},
    }

    for model, spec in RUNS.items():
        if spec["remote"]:
            snapshot_subdir = cli.remote_snapshot_subdir
            if cli.stochastic and snapshot_subdir == "remote_snapshots":
                snapshot_subdir = "remote_snapshots_stochastic"
            run_dir = HERE / snapshot_subdir / f"remote_{spec['remote_key']}" / "checkpoint_snapshot"
            checkpoint_dir = run_dir
            if not (checkpoint_dir / "args.json").is_file():
                raise FileNotFoundError(checkpoint_dir / "args.json")
            load_run_dir = run_dir
            checkpoint_files = sorted(path.name for path in checkpoint_dir.iterdir() if path.is_file())
        else:
            run_dir = Path(spec["run_dir"])
            if not run_dir.is_dir():
                raise FileNotFoundError(run_dir)
            checkpoint_dir, checkpoint_files_meta = base.snapshot_checkpoint(
                run_dir,
                cli.output_dir / "snapshots" / model,
            )
            load_run_dir = run_dir
            checkpoint_files = checkpoint_files_meta

        checkpoint_step = base.checkpoint_manifest_step(checkpoint_dir)
        if checkpoint_step is None:
            raise RuntimeError(f"Missing verified checkpoint manifest: {checkpoint_dir}")
        args, env, actors, normers = base.load_policy(load_run_dir, checkpoint_dir)
        metadata["runs"][model] = {
            "label": spec["label"],
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": checkpoint_step,
            "trained_psi": float(args.association_threshold),
            "checkpoint_files": checkpoint_files,
        }
        try:
            for index, seed in enumerate(seeds, 1):
                row = base.evaluate_episode(
                    args,
                    env,
                    actors,
                    normers,
                    seed,
                    checkpoint_step,
                    deterministic=not cli.stochastic,
                )
                row["model"] = model
                rows.append(row)
                if index == 1 or index % 5 == 0 or index == len(seeds):
                    print(json.dumps({"model": model, "episode": index, "episodes": len(seeds)}), flush=True)
        finally:
            env.close()

    aggregates = base.aggregate(rows)
    base.write_csv(cli.output_dir / "episode_metrics.csv", rows)
    base.write_csv(cli.output_dir / "aggregate_metrics.csv", aggregates)
    make_plot(cli.output_dir, aggregates, deterministic=not cli.stochastic)
    metadata["aggregate_metrics"] = aggregates
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(cli.output_dir),
        "rows": len(rows),
        "plot": str(cli.output_dir / "psi_curriculum_reviewer_metrics.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
