#!/usr/bin/env python
"""Evaluate the three user-selected continuation checkpoints for 15 episodes.

The evaluator intentionally reads the frozen ``run1 - 副本`` directories
directly.  It loads one coherent checkpoint per speed, verifies the hashes in
``checkpoint_manifest.json``, and then records the full trajectory and the
slot-level quantities needed by the reviewer-response figures.
"""

from __future__ import annotations

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_attention import (
    R_Actor_Attention,
)
from onpolicy.envs.mec.env_maker import WrappedMECEnv
from onpolicy.envs.mec.mec import (
    A,
    P1,
    P2,
    U_tip,
    d0,
    g,
    rho,
    v0,
    _cartesian_flight_velocity,
)
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.scripts.eval import evaluate_dynamic_mappo_fixed as base


OUTPUT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = OUTPUT_ROOT / "data"
EVAL_SEEDS = list(range(1001, 1016))

# These are the exact directories specified by the user.  The copied
# directories are used as immutable evaluation sources while the live
# continuation runs may continue writing their original ``run1`` folders.
MODELS = [
    {
        "v_max": 20,
        "train_seed": 2,
        "label": "vmax20_seed2",
        "run_dir": Path(
            r"D:\wyj\Projects\on-policy-last-obs-clean-5UAV\onpolicy\scripts\results\mec\mappo\dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax20_seed2_resume40m_20260814\run1 - 副本"
        ),
    },
    {
        "v_max": 30,
        "train_seed": 32,
        "label": "vmax30_seed32",
        "run_dir": Path(
            r"D:\wyj\Projects\on-policy-last-obs-clean-5UAV\onpolicy\scripts\results\mec\mappo\dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax30_seed32_resume40m_20260814\run1 - 副本"
        ),
    },
    {
        "v_max": 40,
        "train_seed": 32,
        "label": "vmax40_seed32",
        "run_dir": Path(
            r"D:\wyj\Projects\on-policy-last-obs-clean-5UAV\onpolicy\scripts\results\mec\mappo\dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed32_resume40m_20260814\run1 - 副本"
        ),
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(run_dir: Path) -> tuple[Path, dict]:
    checkpoint_dir = run_dir / "models"
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = {}
    for name, metadata in manifest.get("files", {}).items():
        path = checkpoint_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        expected = metadata.get("sha256")
        if expected and actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checked[name] = {"bytes": path.stat().st_size, "sha256": actual}
    manifest["verified_files"] = checked
    return checkpoint_dir, manifest


def load_actor(path: Path, actor_class, args, obs_space, action_space, device):
    actor = actor_class(args, obs_space, action_space, device)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def propulsion_power(speed):
    speed = np.asarray(speed, dtype=np.float64)
    return (
        P1 * (1.0 + 3.0 * speed**2 / U_tip**2)
        + P2
        * (
            np.sqrt(1.0 + speed**4 / (4.0 * v0**4))
            - speed**2 / (2.0 * v0**2)
        )
        ** 0.5
        + 0.5 * d0 * rho * g * A * speed**3
    )


def graph_stats(positions, radius):
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    adjacency = (distances <= radius) & (~np.eye(len(positions), dtype=bool))
    degrees = adjacency.sum(axis=1).astype(np.float64)
    min_distance = float(np.min(distances[~np.eye(len(positions), dtype=bool)]))
    return degrees, adjacency.astype(np.uint8), min_distance


def scalar_info(info, key, default=np.nan) -> float:
    value = np.asarray(info.get(key, default), dtype=np.float64)
    return float(np.mean(value)) if value.size else float(default)


def evaluate_episode(env, actors, normers, args, seed, checkpoint_meta):
    """Run one deterministic episode and return arrays plus scalar metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    obs, states, available_actions, _, attention_mask = env.reset()
    if bool(getattr(env, "curriculum_random_reset", False)):
        raise RuntimeError("fixed evaluation unexpectedly used a random reset")
    obs, states = base.prepare_policy_inputs(normers, obs, states)

    n_uavs = int(args.n_UAVs)
    steps = int(args.episode_length)
    dt = float(args.Delta_t)
    radius = float(args.neighbor_distance)
    positions = np.zeros((steps + 1, n_uavs, 2), dtype=np.float64)
    speeds = np.zeros((steps, n_uavs), dtype=np.float64)
    command_speeds = np.zeros((steps, n_uavs), dtype=np.float64)
    propulsion_energy = np.zeros((steps, n_uavs), dtype=np.float64)
    degrees = np.zeros((steps, n_uavs), dtype=np.float64)
    adjacency = np.zeros((steps, n_uavs, n_uavs), dtype=np.uint8)
    min_separation = np.zeros(steps, dtype=np.float64)
    rewards = np.zeros((steps, n_uavs), dtype=np.float64)
    system_performance_cumulative = np.zeros(steps + 1, dtype=np.float64)
    task_energy_cumulative = np.zeros(steps + 1, dtype=np.float64)
    task_plus_flight_energy_cumulative = np.zeros(steps + 1, dtype=np.float64)
    positions[0] = env.uav_positions[:, :2]
    rnn_states = np.zeros(
        (n_uavs, int(args.recurrent_N), int(args.hidden_size)), dtype=np.float32
    )
    masks = np.ones((n_uavs, 1), dtype=np.float32)
    final_info = {}

    with torch.no_grad():
        for step in range(steps):
            raw_actions = []
            for agent_id, actor in enumerate(actors):
                kwargs = {}
                if args.use_atten_actor:
                    kwargs["attention_active_mask"] = attention_mask[
                        agent_id : agent_id + 1
                    ]
                action, _, next_rnn = actor(
                    obs[agent_id : agent_id + 1],
                    rnn_states[agent_id : agent_id + 1],
                    masks[agent_id : agent_id + 1],
                    available_actions[agent_id : agent_id + 1],
                    deterministic=True,
                    **kwargs,
                )
                raw_actions.append(action.cpu().numpy()[0])
                rnn_states[agent_id] = next_rnn.cpu().numpy()[0]

            raw_actions = np.asarray(raw_actions, dtype=np.float64)
            velocity_vectors = _cartesian_flight_velocity(
                raw_actions[:, :2], float(args.v_max)
            )
            command_speeds[step] = np.linalg.norm(velocity_vectors, axis=1)
            propulsion_energy[step] = propulsion_power(command_speeds[step]) * dt
            (
                obs,
                states,
                reward,
                dones,
                final_info,
                available_actions,
                _,
                attention_mask,
            ) = env.step(raw_actions)
            rewards[step] = np.asarray(reward, dtype=np.float64).reshape(n_uavs)
            positions[step + 1] = env.uav_positions[:, :2]
            system_performance_cumulative[step + 1] = scalar_info(
                {"value": env.system_performance_true_all_GUs}, "value"
            )
            task_energy_cumulative[step + 1] = scalar_info(
                {"value": env.energy_true_all_GUs}, "value"
            )
            task_plus_flight_energy_cumulative[step + 1] = scalar_info(
                {"value": env.energy_all_GUs_UAVs}, "value"
            )
            speeds[step] = np.linalg.norm(
                positions[step + 1] - positions[step], axis=1
            ) / dt
            degrees[step], adjacency[step], min_separation[step] = graph_stats(
                positions[step + 1], radius
            )
            obs, states = base.prepare_policy_inputs(normers, obs, states)
            masks[:] = 0.0 if np.all(dones) else 1.0

    edge_union = adjacency[1:] | adjacency[:-1]
    edge_intersection = adjacency[1:] & adjacency[:-1]
    union_count = edge_union.sum(axis=(1, 2)).astype(np.float64)
    intersection_count = edge_intersection.sum(axis=(1, 2)).astype(np.float64)
    edge_turnover = np.divide(
        union_count - intersection_count,
        union_count,
        out=np.zeros_like(union_count),
        where=union_count > 0,
    )
    connected = np.asarray(
        [
            np.all(
                np.linalg.matrix_power(
                    adj.astype(np.int64) + np.eye(n_uavs, dtype=np.int64),
                    n_uavs - 1,
                )
                > 0
            )
            for adj in adjacency
        ],
        dtype=bool,
    )
    boundary_tol = 1e-6
    in_bounds = (
        (positions[:, :, 0] >= float(args.x_min_uav) - boundary_tol)
        & (positions[:, :, 0] <= float(args.x_max_uav) + boundary_tol)
        & (positions[:, :, 1] >= float(args.y_min_uav) - boundary_tol)
        & (positions[:, :, 1] <= float(args.y_max_uav) + boundary_tol)
    )
    collision_free = min_separation > float(getattr(args, "Dis_min", 0.0))
    speed_flat = speeds.reshape(-1)
    system_increment = np.diff(system_performance_cumulative)
    task_energy_increment = np.diff(task_energy_cumulative)
    task_plus_flight_energy_increment = np.diff(task_plus_flight_energy_cumulative)
    metrics = {
        "v_max": int(args.v_max),
        "train_seed": int(checkpoint_meta["train_seed"]),
        "checkpoint_step": int(checkpoint_meta["checkpoint_step"]),
        "eval_seed": int(seed),
        "system_performance_true_all_GUs": scalar_info(
            final_info, "system_performance_true_all_GUs"
        ),
        "complete_task_ratio": scalar_info(final_info, "complete_task_ratio"),
        "md_admission_ratio": scalar_info(final_info, "md_admission_ratio"),
        "average_active_mds": scalar_info(final_info, "average_active_mds"),
        "delay_true_all_GUs": scalar_info(final_info, "delay_true_all_GUs"),
        "energy_true_all_GUs": scalar_info(final_info, "energy_true_all_GUs"),
        "actual_speed_mean_mps": float(np.mean(speed_flat)),
        "actual_speed_p50_mps": float(np.quantile(speed_flat, 0.50)),
        "actual_speed_p95_mps": float(np.quantile(speed_flat, 0.95)),
        "actual_speed_max_mps": float(np.max(speed_flat)),
        "speed_utilization_mean": float(np.mean(speed_flat) / float(args.v_max)),
        "speed_saturation_fraction": float(
            np.mean(speed_flat >= 0.99 * float(args.v_max))
        ),
        "command_speed_mean_mps": float(np.mean(command_speeds)),
        "command_speed_max_mps": float(np.max(command_speeds)),
        "total_trajectory_length_m": float(np.sum(speeds) * dt),
        "propulsion_energy_proxy_j": float(np.sum(propulsion_energy)),
        "mean_neighbor_degree": float(np.mean(degrees)),
        "mean_directed_message_edge_fraction": float(
            np.mean(degrees) / max(n_uavs - 1, 1)
        ),
        "mean_neighbor_edge_turnover": float(np.mean(edge_turnover)),
        "connected_slot_fraction": float(np.mean(connected)),
        "minimum_uav_separation_m": float(np.min(min_separation)),
        "collision_violation_slot_fraction": float(np.mean(~collision_free)),
        "out_of_bounds_position_fraction": float(np.mean(~in_bounds)),
        "system_performance_early_mean_per_slot": float(
            np.mean(system_increment[:100])
        ),
        "system_performance_late_mean_per_slot": float(
            np.mean(system_increment[100:])
        ),
        "system_performance_full_mean_per_slot": float(
            np.mean(system_increment)
        ),
        "uav_initial_positions": positions[0].tolist(),
        "uav_final_positions": positions[-1].tolist(),
    }
    arrays = {
        "positions": positions,
        "speeds": speeds,
        "command_speeds": command_speeds,
        "propulsion_energy": propulsion_energy,
        "degrees": degrees,
        "adjacency": adjacency,
        "min_separation": min_separation,
        "edge_turnover": edge_turnover,
        "connected": connected,
        "rewards": rewards,
        "system_performance_cumulative": system_performance_cumulative,
        "system_performance_increment": system_increment,
        "task_energy_cumulative": task_energy_cumulative,
        "task_energy_increment": task_energy_increment,
        "task_plus_flight_energy_cumulative": task_plus_flight_energy_cumulative,
        "task_plus_flight_energy_increment": task_plus_flight_energy_increment,
    }
    return arrays, metrics


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    model_records = []

    torch.set_num_threads(1)
    device = torch.device("cpu")
    for model_index, model in enumerate(MODELS, start=1):
        run_dir = model["run_dir"].resolve()
        checkpoint_dir, manifest = verify_checkpoint(run_dir)
        source_args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
        args = Namespace(**source_args)
        args.n_rollout_threads = 1
        args.n_training_threads = 1
        args.model_dir = str(checkpoint_dir)
        args.uav_reset_curriculum_training = False
        checkpoint_step = int(manifest["total_num_steps"])
        if int(args.v_max) != int(model["v_max"]):
            raise RuntimeError(f"v_max mismatch for {run_dir}")
        checkpoint_meta = {
            "v_max": int(model["v_max"]),
            "train_seed": int(model["train_seed"]),
            "label": model["label"],
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": checkpoint_step,
            "episode_index": manifest.get("episode_index"),
            "source_total_num_steps": manifest.get("source_total_num_steps"),
            "session_total_num_steps": manifest.get("session_total_num_steps"),
            "manifest": manifest,
        }
        model_records.append(checkpoint_meta)
        model_data_root = DATA_ROOT / model["label"]
        model_data_root.mkdir(parents=True, exist_ok=True)
        (model_data_root / "checkpoint_metadata.json").write_text(
            json.dumps(checkpoint_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        env = WrappedMECEnv(args=args)
        actor_class = R_Actor_Attention if args.use_atten_actor else R_Actor
        actors, normers = [], []
        for agent_id in range(int(args.n_UAVs)):
            actors.append(
                load_actor(
                    checkpoint_dir / f"actor_agent{agent_id}.pt",
                    actor_class,
                    args,
                    env.observation_space[agent_id],
                    env.action_space[agent_id],
                    device,
                )
            )
            normer = Normer(
                args=args,
                obs_space=env.observation_space[agent_id],
                states_space=env.share_observation_space[agent_id],
            )
            normer.load(checkpoint_dir / f"normer{agent_id}.pkl")
            normers.append(normer)

        try:
            for episode_index, eval_seed in enumerate(EVAL_SEEDS, start=1):
                arrays, metrics = evaluate_episode(
                    env, actors, normers, args, eval_seed, checkpoint_meta
                )
                episode_dir = model_data_root / f"eval{eval_seed}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(episode_dir / "speed_trace.npz", **arrays)
                (episode_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "v_max": int(model["v_max"]),
                            "train_seed": int(model["train_seed"]),
                            "checkpoint_step": checkpoint_step,
                            "evaluation_seed": int(eval_seed),
                            "checkpoint_dir": str(checkpoint_dir),
                            "delta_t_s": float(args.Delta_t),
                            "episode_length_slots": int(args.episode_length),
                            "neighbor_distance_m": float(args.neighbor_distance),
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                all_metrics.append(metrics)
                print(
                    json.dumps(
                        {
                            "model": f"{model_index}/{len(MODELS)}",
                            "v_max": model["v_max"],
                            "train_seed": model["train_seed"],
                            "eval": f"{episode_index}/{len(EVAL_SEEDS)}",
                            "eval_seed": eval_seed,
                            "performance": metrics[
                                "system_performance_true_all_GUs"
                            ],
                            "mean_speed_mps": metrics["actual_speed_mean_mps"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        finally:
            env.close()

    fields = list(all_metrics[0])
    with (OUTPUT_ROOT / "episode_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, list)
                    else value
                    for key, value in row.items()
                }
            )

    (OUTPUT_ROOT / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "evaluation_seeds": EVAL_SEEDS,
                "episodes_per_model": len(EVAL_SEEDS),
                "models": model_records,
                "data_root": str(DATA_ROOT),
                "note": (
                    "The three selected copied checkpoints were loaded directly; "
                    "all statistics are deterministic-policy episode statistics."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "models": len(MODELS),
                "episodes": len(all_metrics),
                "episode_metrics": str(OUTPUT_ROOT / "episode_metrics.csv"),
                "data_root": str(DATA_ROOT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
