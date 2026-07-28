#!/usr/bin/env python
"""Paired deterministic rollouts with UAV5 forced to navigate to fixed targets."""

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_attention import R_Actor_Attention
from onpolicy.envs.mec.env_maker import WrappedMECEnv
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.scripts.eval.render_dynamic_mappo_episode import frozen_normalize


CONDITIONS = {
    "policy": None,
    "uav5_500_500": (500.0, 500.0),
    "uav5_520_480": (520.0, 480.0),
}
METRICS = (
    "true_performance",
    "equivalent_performance",
    "active_last200",
    "completion_last200",
    "reward_per_slot_last200",
    "reward_per_active_last200",
    "admission_ratio",
    "upper_admission_ratio",
    "coverage_departures",
)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def first_value(info, key):
    return float(np.asarray(info[key]).reshape(-1)[0])


def load_models(args, env, checkpoint_dir):
    device = torch.device("cpu")
    actor_class = R_Actor_Attention if args.use_atten_actor else R_Actor
    actors, normers = [], []
    for agent_id in range(args.n_UAVs):
        actor = actor_class(args, env.observation_space[agent_id], env.action_space[agent_id], device)
        actor.load_state_dict(torch.load(
            checkpoint_dir / f"actor_agent{agent_id}.pt", map_location=device, weights_only=True
        ))
        actor.eval()
        actors.append(actor)

        normer = Normer(args=args, obs_space=env.observation_space[agent_id],
                        states_space=env.share_observation_space[agent_id])
        normer.load(checkpoint_dir / f"normer{agent_id}.pkl")
        normers.append(normer)
    return actors, normers


def force_navigation(action, position, target, args):
    delta = np.asarray(target) - position
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0:
        action[:2] = (0.0, 0.0)
        return True
    direction = np.arctan2(delta[1], delta[0]) % (2 * np.pi)
    speed = min(float(args.v_max), distance / float(args.Delta_t))
    action[:2] = (direction / (2 * np.pi), speed / float(args.v_max))
    return False


def rollout(args, checkpoint_dir, actors, normers, seed, condition, target):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = WrappedMECEnv(args=args)
    env.seed(seed)
    obs, states, available_actions, _, attention_mask = env.reset()
    obs, states = frozen_normalize(normers, obs, states)

    n_uavs = int(args.n_UAVs)
    rnn_states = np.zeros((n_uavs, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((n_uavs, 1), dtype=np.float32)
    active_counts, completed_counts, rewards, positions = [], [], [], [env.uav_positions[:, :2].copy()]
    reached_slot = None
    final_info = None

    with torch.no_grad():
        for step in range(int(args.episode_length)):
            decision_active = env.active_md_mask.copy()
            raw_actions = []
            for agent_id, actor in enumerate(actors):
                kwargs = {"attention_active_mask": attention_mask[agent_id:agent_id + 1]} \
                    if args.use_atten_actor else {}
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

            if target is not None:
                reached = force_navigation(raw_actions[4], env.uav_positions[4, :2], target, args)
                if reached and reached_slot is None:
                    reached_slot = step + 1

            obs, states, reward, dones, final_info, available_actions, _, attention_mask = env.step(
                np.asarray(raw_actions)
            )
            active_counts.append(int(decision_active.sum()))
            completed_counts.append(int(np.sum(env.complete_task[decision_active])))
            rewards.append(np.asarray(reward).reshape(n_uavs))
            positions.append(env.uav_positions[:, :2].copy())
            obs, states = frozen_normalize(normers, obs, states)
            masks[:] = 0.0 if np.all(dones) else 1.0

    env.close()
    active_counts = np.asarray(active_counts)
    completed_counts = np.asarray(completed_counts)
    rewards = np.asarray(rewards)
    positions = np.asarray(positions)
    tail = slice(len(active_counts) // 2, None)
    tail_tasks = max(int(active_counts[tail].sum()), 1)
    boundary = (
        (positions[:, :, 0] <= 1e-8) | (positions[:, :, 0] >= 600 - 1e-8)
        | (positions[:, :, 1] <= 1e-8) | (positions[:, :, 1] >= 600 - 1e-8)
    )

    return {
        "seed": seed,
        "condition": condition,
        "true_performance": first_value(final_info, "system_performance_true_all_GUs"),
        "equivalent_performance": first_value(final_info, "system_performance_equivalent_full_GUs"),
        "active_last200": float(active_counts[tail].mean()),
        "completion_last200": float(completed_counts[tail].sum() / tail_tasks),
        "reward_per_slot_last200": float(rewards[tail].sum(axis=1).mean()),
        "reward_per_active_last200": float(rewards[tail].sum() / tail_tasks),
        "admission_ratio": float(final_info["md_admission_ratio"]),
        "upper_admission_ratio": float(final_info["md_admission_ratio_upper_right"]),
        "coverage_departures": int(final_info["coverage_departed_md_accesses"]),
        "uav5_reached_slot": reached_slot,
        "uav5_final_x": float(positions[-1, 4, 0]),
        "uav5_final_y": float(positions[-1, 4, 1]),
        "uav4_boundary_slots": int(boundary[:, 3].sum()),
        "uav5_boundary_slots": int(boundary[:, 4].sum()),
        "uav_rewards": rewards.sum(axis=0).tolist(),
    }


def summarize(rows):
    summary = {"conditions": {}, "paired_deltas_vs_policy": {}}
    by_condition = {name: [row for row in rows if row["condition"] == name] for name in CONDITIONS}
    for name, group in by_condition.items():
        summary["conditions"][name] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in group])),
                "std": float(np.std([row[metric] for row in group], ddof=1)),
            }
            for metric in METRICS
        }
        summary["conditions"][name]["uav4_boundary_episodes"] = int(sum(row["uav4_boundary_slots"] > 0 for row in group))
        summary["conditions"][name]["uav5_boundary_episodes"] = int(sum(row["uav5_boundary_slots"] > 0 for row in group))

    baseline = {row["seed"]: row for row in by_condition["policy"]}
    for name in ("uav5_500_500", "uav5_520_480"):
        group = by_condition[name]
        paired = {}
        for metric in METRICS:
            deltas = np.asarray([row[metric] - baseline[row["seed"]][metric] for row in group])
            paired[metric] = {
                "mean": float(deltas.mean()),
                "std": float(deltas.std(ddof=1)),
                "ci95_half_width": float(1.96 * deltas.std(ddof=1) / np.sqrt(len(deltas))),
                "win_rate": float(np.mean(deltas > 0)),
            }
        summary["paired_deltas_vs_policy"][name] = paired
    return summary


def main():
    cli = parse_cli()
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    cli.output_dir.mkdir(parents=True)
    with (cli.run_dir / "args.json").open("r", encoding="utf-8") as handle:
        args = Namespace(**json.load(handle))
    args.n_rollout_threads = 1
    args.n_training_threads = 1
    args.model_dir = str(cli.checkpoint_dir)
    torch.set_num_threads(1)

    probe_env = WrappedMECEnv(args=args)
    actors, normers = load_models(args, probe_env, cli.checkpoint_dir)
    probe_env.close()

    rows = []
    for condition, target in CONDITIONS.items():
        for index, seed in enumerate(cli.seeds, 1):
            rows.append(rollout(args, cli.checkpoint_dir, actors, normers, seed, condition, target))
            print(f"{condition}: {index}/{len(cli.seeds)} seed={seed}", flush=True)

    with (cli.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "uav_rewards"])
        writer.writeheader()
        writer.writerows({key: jsonable(value) for key, value in row.items() if key != "uav_rewards"} for row in rows)

    result = {
        "protocol": {
            "run_dir": str(cli.run_dir.resolve()),
            "checkpoint_dir": str(cli.checkpoint_dir.resolve()),
            "seeds": cli.seeds,
            "conditions": {key: value for key, value in CONDITIONS.items()},
            "forced_agent": 5,
            "navigation": "straight-line at up to v_max, then exact zero-speed hover within 1 m",
            "policy_actions_retained": "all UAVs; UAV5 resource actions; only UAV5 flight direction/speed overridden",
        },
        "summary": summarize(rows),
        "episodes": rows,
    }
    with (cli.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=jsonable)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
