#!/usr/bin/env python
"""Evaluate a separated 5-UAV checkpoint on deterministic fixed resets.

This script is deliberately separate from training: it snapshots a coherent
checkpoint, freezes the saved observation statistics, disables the reset
curriculum, and evaluates every candidate on exactly the same explicit seeds.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_attention import (
    R_Actor_Attention,
)
from onpolicy.envs.mec.env_maker import WrappedMECEnv
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.scripts.eval.render_dynamic_mappo_episode import (
    frozen_normalize,
    snapshot_checkpoint,
)


SUMMARY_METRICS = (
    "system_performance_true_all_GUs",
    "complete_task_ratio",
    "md_admission_ratio",
    "md_admission_ratio_lower_left",
    "md_admission_ratio_upper_right",
    "average_active_mds",
    "n_GUs_by_coverd",
    "delay_true_all_GUs",
    "energy_true_all_GUs",
    "total_uav_trajectory_length",
    "final_position_stability_slot_10m",
    "last50_uav_trajectory_length",
)

METRIC_DIRECTIONS = {
    "system_performance_true_all_GUs": 1,
    "complete_task_ratio": 1,
    "md_admission_ratio": 1,
    "md_admission_ratio_lower_left": 1,
    "md_admission_ratio_upper_right": 1,
    "average_active_mds": 1,
    "n_GUs_by_coverd": 1,
    "delay_true_all_GUs": -1,
    "energy_true_all_GUs": -1,
    "total_uav_trajectory_length": -1,
    "final_position_stability_slot_10m": -1,
    "last50_uav_trajectory_length": -1,
}

MATCHED_CONFIG_EXCLUSIONS = {
    "advantage_mode",
    "device",
    "experiment_name",
    "externality_beta",
    "model_dir",
    "neighbor_R",
    "neighbor_distance",
    "seed",
    "user_name",
}


def matched_config(source_args):
    return {
        key: value
        for key, value in sorted(source_args.items())
        if key not in MATCHED_CONFIG_EXCLUSIONS
    }


def config_sha256(config):
    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--training-step", type=int, required=True)
    parser.add_argument(
        "--confirm-checkpoint-frozen",
        action="store_true",
        help=(
            "Required acknowledgement that the trainer has stopped writing run-dir/models. "
            "The evaluator refuses live model directories because the five independent "
            "actors and normers are saved sequentially."
        ),
    )
    return parser.parse_args()


def scalar_info(info, key):
    value = np.asarray(info.get(key, np.nan), dtype=np.float64)
    return float(np.mean(value)) if value.size else float("nan")


def final_position_stability_slot(positions, tolerance=10.0):
    """Earliest slot after which every UAV stays within tolerance of its endpoint."""
    distance_to_final = np.linalg.norm(positions - positions[-1:], axis=-1)
    within = np.all(distance_to_final <= tolerance, axis=1)
    suffix_within = np.logical_and.accumulate(within[::-1])[::-1]
    candidates = np.flatnonzero(suffix_within)
    return int(candidates[0]) if candidates.size else int(len(positions) - 1)


def aggregate_rows(rows):
    aggregate = {}
    for metric in SUMMARY_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if not finite.size:
            aggregate[metric] = {
                "count": 0,
                "mean": None,
                "std": None,
                "ci95_half_width": None,
            }
            continue
        std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        critical = (
            float(student_t.ppf(0.975, finite.size - 1))
            if finite.size > 1
            else 0.0
        )
        aggregate[metric] = {
            "count": int(finite.size),
            "mean": float(np.mean(finite)),
            "std": std,
            "ci95_half_width": float(critical * std / np.sqrt(finite.size)),
        }
    return aggregate


def load_actor(path, actor_class, args, obs_space, action_space, device):
    actor = actor_class(args, obs_space, action_space, device)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def evaluate_episode(env, actors, normers, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    obs, states, available_actions, _, attention_mask = env.reset()
    if bool(getattr(env, "curriculum_random_reset", False)):
        raise RuntimeError("fixed evaluation unexpectedly used a random reset")
    obs, states = frozen_normalize(normers, obs, states)

    n_uavs = int(args.n_UAVs)
    rnn_states = np.zeros(
        (n_uavs, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    masks = np.ones((n_uavs, 1), dtype=np.float32)
    positions = [env.uav_positions[:, :2].copy()]
    reward_sum = np.zeros(n_uavs, dtype=np.float64)
    final_info = None

    with torch.no_grad():
        for _ in range(int(args.episode_length)):
            raw_actions = []
            for agent_id, actor in enumerate(actors):
                kwargs = {}
                if args.use_atten_actor:
                    kwargs["attention_active_mask"] = attention_mask[
                        agent_id:agent_id + 1
                    ]
                action, _, next_rnn = actor(
                    obs[agent_id:agent_id + 1],
                    rnn_states[agent_id:agent_id + 1],
                    masks[agent_id:agent_id + 1],
                    available_actions[agent_id:agent_id + 1],
                    deterministic=True,
                    **kwargs,
                )
                raw_actions.append(action.cpu().numpy()[0])
                rnn_states[agent_id] = next_rnn.cpu().numpy()[0]

            step_result = env.step(np.asarray(raw_actions))
            (
                obs,
                states,
                rewards,
                dones,
                final_info,
                available_actions,
                _,
                attention_mask,
            ) = step_result
            reward_sum += np.asarray(rewards, dtype=np.float64).reshape(n_uavs)
            positions.append(env.uav_positions[:, :2].copy())
            obs, states = frozen_normalize(normers, obs, states)
            masks[:] = 0.0 if np.all(dones) else 1.0

    positions = np.asarray(positions, dtype=np.float64)
    per_uav_distance = np.sum(
        np.linalg.norm(np.diff(positions, axis=0), axis=-1), axis=0
    )
    last50_start = max(0, len(positions) - 51)
    last50_distance = np.sum(
        np.linalg.norm(np.diff(positions[last50_start:], axis=0), axis=-1)
    )
    row = {
        "seed": int(seed),
        "fixed_reset": not bool(final_info.get("curriculum_random_reset", False)),
        "system_performance_true_all_GUs": scalar_info(
            final_info, "system_performance_true_all_GUs"
        ),
        "complete_task_ratio": scalar_info(final_info, "complete_task_ratio"),
        "md_admission_ratio": scalar_info(final_info, "md_admission_ratio"),
        "md_admission_ratio_lower_left": scalar_info(
            final_info, "md_admission_ratio_lower_left"
        ),
        "md_admission_ratio_upper_right": scalar_info(
            final_info, "md_admission_ratio_upper_right"
        ),
        "average_active_mds": scalar_info(final_info, "average_active_mds"),
        "n_GUs_by_coverd": scalar_info(final_info, "n_GUs_by_coverd"),
        "delay_true_all_GUs": scalar_info(final_info, "delay_true_all_GUs"),
        "energy_true_all_GUs": scalar_info(final_info, "energy_true_all_GUs"),
        "total_actor_reward": float(np.sum(reward_sum)),
        "total_uav_trajectory_length": float(np.sum(per_uav_distance)),
        "final_position_stability_slot_10m": final_position_stability_slot(
            positions
        ),
        "last50_uav_trajectory_length": float(last50_distance),
    }
    for agent_id in range(n_uavs):
        row[f"uav{agent_id}_trajectory_length"] = float(per_uav_distance[agent_id])
        row[f"uav{agent_id}_initial_x"] = float(positions[0, agent_id, 0])
        row[f"uav{agent_id}_initial_y"] = float(positions[0, agent_id, 1])
        row[f"uav{agent_id}_final_x"] = float(positions[-1, agent_id, 0])
        row[f"uav{agent_id}_final_y"] = float(positions[-1, agent_id, 1])
    return row


def main():
    cli = parse_cli()
    run_dir = cli.run_dir.resolve()
    output_dir = cli.output_dir.resolve()
    if len(set(cli.seeds)) != len(cli.seeds):
        raise ValueError("evaluation seeds must be unique")
    if not cli.confirm_checkpoint_frozen:
        raise ValueError(
            "refusing a potentially live checkpoint; stop checkpoint writes and pass "
            "--confirm-checkpoint-frozen"
        )
    if cli.training_step <= 0:
        raise ValueError("training step must be positive")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    checkpoint_dir, checkpoint_files = snapshot_checkpoint(run_dir, output_dir)
    with (checkpoint_dir / "args.json").open("r", encoding="utf-8") as handle:
        source_args = json.load(handle)
    comparison_config = matched_config(source_args)
    args = Namespace(**source_args)
    args.n_rollout_threads = 1
    args.n_training_threads = 1
    args.model_dir = str(checkpoint_dir)
    args.uav_reset_curriculum_training = False

    torch.set_num_threads(1)
    device = torch.device("cpu")
    env = WrappedMECEnv(args=args)
    actor_class = R_Actor_Attention if args.use_atten_actor else R_Actor
    actors = []
    normers = []
    for agent_id in range(args.n_UAVs):
        actors.append(load_actor(
            checkpoint_dir / f"actor_agent{agent_id}.pt",
            actor_class,
            args,
            env.observation_space[agent_id],
            env.action_space[agent_id],
            device,
        ))
        normer = Normer(
            args=args,
            obs_space=env.observation_space[agent_id],
            states_space=env.share_observation_space[agent_id],
        )
        normer.load(checkpoint_dir / f"normer{agent_id}.pkl")
        normers.append(normer)

    rows = []
    try:
        for seed in cli.seeds:
            rows.append(evaluate_episode(env, actors, normers, args, seed))
    finally:
        env.close()

    for row in rows:
        for metric in SUMMARY_METRICS:
            if not np.isfinite(float(row[metric])):
                raise ValueError(
                    f"seed {row['seed']} produced non-finite metric {metric}"
                )

    with (output_dir / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "completed",
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_files": checkpoint_files,
        "declared_training_step": cli.training_step,
        "checkpoint_write_state": "caller-confirmed frozen before snapshot",
        "matched_config_exclusions": sorted(MATCHED_CONFIG_EXCLUSIONS),
        "matched_config": comparison_config,
        "matched_config_sha256": config_sha256(comparison_config),
        "actor_mode": "deterministic",
        "reset_mode": "fixed",
        "normalization": "frozen saved statistics",
        "seed_pairing_note": (
            "The same seed initializes each policy evaluation, but policy-dependent MD "
            "admission changes later RNG consumption. Treat pairs as matched seed labels, "
            "not as identical exogenous user trajectories."
        ),
        "seeds": cli.seeds,
        "episode_count": len(rows),
        "aggregate": aggregate_rows(rows),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
