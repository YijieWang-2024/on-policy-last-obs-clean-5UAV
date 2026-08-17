"""Evaluate the five latest curriculum-on psi checkpoints on common seeds.

This wrapper reuses the established reviewer-metric evaluator.  It freezes the
two local checkpoints at evaluation start, uses the already-frozen remote
checkpoint copies, and then writes both episode-wise counts and pooled rates.
The pooled rates are the primary values for reviewer reporting because their
numerators and denominators are retained explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "psi_reviewer_final_metrics_20260816" / "evaluate_final_checkpoints.py"
RESULTS = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
REMOTE_STAGING = HERE / "remote_checkpoints_latest_20260817"

RUNS = {
    "psi0p1_seed2_local": {
        "label": "psi=0.1 | local | seed=2",
        "remote": False,
        "run_dir": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
    },
    "psi0p3_seed2_local": {
        "label": "psi=0.3 | local | seed=2",
        "remote": False,
        "run_dir": RESULTS / "dcppoR520_fixed600_200_noactor_psi0p3_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
    },
    "psi0p5_seed2_remote": {
        "label": "psi=0.5 | remote | seed=2",
        "remote": True,
        "run_dir": REMOTE_STAGING / "psi0p5",
        "checkpoint_dir": REMOTE_STAGING / "psi0p5" / "models",
    },
    "psi0p7_seed2_remote": {
        "label": "psi=0.7 | remote | seed=2",
        "remote": True,
        "run_dir": REMOTE_STAGING / "psi0p7",
        "checkpoint_dir": REMOTE_STAGING / "psi0p7" / "models",
    },
    "psi0p9_seed2_remote": {
        "label": "psi=0.9 | remote | seed=2",
        "remote": True,
        "run_dir": REMOTE_STAGING / "psi0p9",
        "checkpoint_dir": REMOTE_STAGING / "psi0p9" / "models",
    },
}

# These are the runtime/environment fields that must match for a fair psi
# comparison.  association_threshold is intentionally the only expected
# difference.
COMPARISON_KEYS = (
    "association_threshold",
    "B",
    "F_m",
    "actor_message_contract",
    "actor_message_mode",
    "actor_message_pool",
    "actor_neighbor_obs",
    "advantage_mode",
    "noise_scale",
    "consensus_alpha",
    "average_neighbor_advantage",
    "dynamic_md",
    "episode_layout_context",
    "episode_layout_context_units",
    "episode_length",
    "hotspot_layout_mode",
    "fix_hotspot",
    "md_arrivals_min",
    "md_arrivals_max",
    "md_arrivals_per_region",
    "md_lifetime_min",
    "md_lifetime_max",
    "md_velocity_init_min_factor",
    "md_velocity_init_max_factor",
    "md_velocity_init_std_factor",
    "md_velocity_update_clip_min_factor",
    "md_velocity_update_clip_max_factor",
    "mean_velocity",
    "n_GUs",
    "n_UAVs",
    "neighbor_R",
    "neighbor_distance",
    "offload_deadline_filter",
    "uav_reset_curriculum",
    "uav_reset_curriculum_schedule",
    "uav_start_positions",
    "v_max",
    "ob_norm",
    "use_shared_obs",
    "use_centralized_V",
    "use_atten_critic",
    "critic_hops",
    "ppo_epoch",
    "clip_param",
    "gamma",
    "n_rollout_threads",
    "n_training_threads",
    "n_rollout_workers",
    "num_env_steps",
    "seed",
)


def load_base():
    spec = importlib.util.spec_from_file_location("psi_reviewer_metrics_base_25", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=25)
    # Reuse the first 25 seeds from the previous 30-deterministic evaluation.
    parser.add_argument("--seed-start", type=int, default=8001)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results_25det_latest_20260817",
    )
    return parser.parse_args()


def read_args(run_dir: Path) -> dict:
    with (run_dir / "args.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def comparable(value):
    """Make JSON/list/NumPy values comparable and serializable."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def build_config_audit():
    configs = {model: read_args(spec["run_dir"]) for model, spec in RUNS.items()}
    reference_model = next(iter(configs))
    reference = configs[reference_model]
    differences = {}
    for key in COMPARISON_KEYS:
        values = {model: comparable(config.get(key)) for model, config in configs.items()}
        if len({json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values.values()}) > 1:
            differences[key] = values
    unexpected = {key: value for key, value in differences.items() if key != "association_threshold"}
    return {
        "reference_model": reference_model,
        "comparison_keys": list(COMPARISON_KEYS),
        "expected_difference": "association_threshold",
        "differences": differences,
        "unexpected_differences": unexpected,
        "fair_except_threshold": not unexpected,
        "configs": configs,
    }


def write_protocol(output_dir: Path, seeds: list[int]) -> None:
    text = f"""# Curriculum-on psi reviewer metrics: latest checkpoints

Five checkpoints were evaluated deterministically on the same {len(seeds)} test
seeds ({seeds[0]}--{seeds[-1]}): curriculum ON, deadline filter OFF,
actor_message disabled, R520 consensus/per-agent noise scale 3, lifetime 12,
mean MD velocity 3 m/s, strict 1+4 arrivals, Fixed600-200 layout, and seed 2
training runs.  Only association_threshold (psi_min) differs across runs.

The per-episode CSV preserves the task counts needed to audit every rate:

- active_task_count: active MD tasks immediately before the action;
- offloaded_task_count: active tasks selected for offloading after action
  transformation;
- successful_offloaded_task_count: offloaded tasks whose environment
  completion flag is true after the step;
- not_offloaded_task_count: active tasks that followed the local branch;
- successful_local_task_count: not-offloaded tasks whose completion flag is
  true after the step;
- local_failure_count: not-offloaded tasks not completed on time.

The pooled reviewer metrics are:

- OffloadRatio = offloaded / active;
- task offloading success rate (OSR) = successful_offloaded / offloaded;
- OffloadYield = successful_offloaded / active;
- LocalShare = not_offloaded / active;
- local completion rate = successful_local / not_offloaded;
- LocalYield = successful_local / active;
- OverallCompletion = (successful_offloaded + successful_local) / active.

Resource fields are reported as allocated utilization, useful utilization,
and failed-offload waste for bandwidth and CPU.  Because the current action
path normalizes resources per serving UAV, these are code-level allocation and
usefulness measures; they are not claimed to be independent physical
busy-time utilization.  The environment's cumulative
system_performance_true_all_GUs is reported as overall system gain, together
with gain per active task.
"""
    (output_dir / "metric_protocol.md").write_text(text, encoding="utf-8")


def copy_remote_snapshot(source: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    for source_file in sorted(source.iterdir()):
        if source_file.is_file():
            shutil.copy2(source_file, destination / source_file.name)
            files.append(source_file.name)
    return files


def main() -> None:
    cli = parse_args()
    if cli.episodes < 20:
        raise ValueError("Use at least 20 common test episodes")
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    cli.output_dir.mkdir(parents=True)
    (cli.output_dir / "snapshots").mkdir()
    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    write_protocol(cli.output_dir, seeds)

    base = load_base()
    torch.set_num_threads(1)
    rows = []
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "episodes": cli.episodes,
        "test_seeds": seeds,
        "deterministic": True,
        "runs": {},
    }

    audit = build_config_audit()
    (cli.output_dir / "configuration_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not audit["fair_except_threshold"]:
        raise RuntimeError(
            "Unexpected runtime/config differences found; see configuration_audit.json"
        )

    for model, spec in RUNS.items():
        run_dir = Path(spec["run_dir"])
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)

        if spec["remote"]:
            source_checkpoint = Path(spec["checkpoint_dir"])
            checkpoint_dir = cli.output_dir / "snapshots" / model
            checkpoint_files = copy_remote_snapshot(source_checkpoint, checkpoint_dir)
            load_run_dir = run_dir
        else:
            checkpoint_dir, checkpoint_files = base.snapshot_checkpoint(
                run_dir, cli.output_dir / "snapshots" / model
            )
            load_run_dir = run_dir

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
                    deterministic=True,
                )
                row["model"] = model
                rows.append(row)
                if index == 1 or index % 5 == 0 or index == len(seeds):
                    print(
                        json.dumps(
                            {"model": model, "episode": index, "episodes": len(seeds)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            env.close()

    aggregates = base.aggregate(rows)
    base.write_csv(cli.output_dir / "episode_metrics.csv", rows)
    base.write_csv(cli.output_dir / "aggregate_metrics.csv", aggregates)
    metadata["aggregate_metrics"] = aggregates
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(cli.output_dir),
                "rows": len(rows),
                "checkpoint_steps": {
                    model: data["checkpoint_step"]
                    for model, data in metadata["runs"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
