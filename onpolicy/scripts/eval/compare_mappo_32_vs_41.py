#!/usr/bin/env python
"""Paired 3-large/2-small versus 4-large/1-small MAPPO counterfactual.

The evaluator supplies both deployments with the same candidate-MD births,
mobility noise, and task sequence.  It delegates action projection and every
reward/channel/delay/energy calculation to the production MEC environment.
"""

import argparse
import csv
import json
import math
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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.mec.env_maker import WrappedMECEnv
from onpolicy.envs.mec.vec_normalize import Normer


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--baseline-episode", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=410000)
    parser.add_argument("--steady-slots", type=int, default=400)
    return parser.parse_args()


def load_args(checkpoint_dir):
    with (checkpoint_dir / "args.json").open("r", encoding="utf-8") as handle:
        args = Namespace(**json.load(handle))
    args.n_rollout_threads = 1
    args.n_training_threads = 1
    args.model_dir = str(checkpoint_dir)
    assert args.dynamic_md and args.md_arrivals_min == args.md_arrivals_max == 3
    assert args.md_lifetime_min == args.md_lifetime_max == 20
    assert args.n_UAVs == 5 and args.n_GUs == 60 and not args.use_atten_actor
    return args


def load_policy(args, checkpoint_dir):
    probe = WrappedMECEnv(args=args)
    device = torch.device("cpu")
    actors, normers = [], []
    for agent_id in range(args.n_UAVs):
        actor = R_Actor(args, probe.observation_space[agent_id], probe.action_space[agent_id], device)
        try:
            weights = torch.load(
                checkpoint_dir / f"actor_agent{agent_id}.pt", map_location=device, weights_only=True
            )
        except TypeError:
            weights = torch.load(checkpoint_dir / f"actor_agent{agent_id}.pt", map_location=device)
        actor.load_state_dict(weights)
        actor.eval()
        actors.append(actor)

        normer = Normer(
            args=args,
            obs_space=probe.observation_space[agent_id],
            states_space=probe.share_observation_space[agent_id],
        )
        normer.load(checkpoint_dir / f"normer{agent_id}.pkl")
        normers.append(normer)
    probe.close()
    return actors, normers


def normalize_obs(normers, obs):
    obs = np.asarray(obs, dtype=np.float32).copy()
    for agent_id, normer in enumerate(normers):
        if normer.ob_rms is None:
            continue
        start = normer.not_norm
        obs[agent_id, start:] = np.clip(
            (obs[agent_id, start:] - normer.ob_rms.mean[start:])
            / np.sqrt(normer.ob_rms.var[start:] + normer.epsilon),
            -normer.clipob,
            normer.clipob,
        )
    return obs


def make_trace(args, seed, slots):
    """Generate immutable candidate sessions for a paired replay."""
    rng = np.random.default_rng(seed)
    life = args.md_lifetime_max
    arrivals = args.md_arrivals_max
    count = slots * arrivals
    bounds = np.empty((count, 4), dtype=np.float64)
    regional = getattr(args, "md_arrivals_per_region", None)
    if regional:
        if len(regional) != 2 or sum(regional) != arrivals:
            raise ValueError("md_arrivals_per_region must contain lower/upper counts")
        lower = np.tile(
            np.r_[np.ones(regional[0], dtype=bool), np.zeros(regional[1], dtype=bool)],
            slots,
        )
    else:
        lower = rng.random(count) < (1.0 / 6.0)
    bounds[lower] = [0.0, 175.0, 0.0, 175.0]
    bounds[~lower] = [200.0, 600.0, 200.0, 600.0]

    positions = np.empty((count, life, 2), dtype=np.float64)
    positions[:, 0, 0] = rng.uniform(bounds[:, 0], bounds[:, 1])
    positions[:, 0, 1] = rng.uniform(bounds[:, 2], bounds[:, 3])
    velocity = np.empty((count, life), dtype=np.float64)
    direction = np.empty((count, life), dtype=np.float64)
    direction0 = np.empty((count, life), dtype=np.float64)
    velocity[:, 0] = np.clip(
        rng.normal(args.mean_velocity, 0.3, count),
        0.7 * args.mean_velocity,
        1.3 * args.mean_velocity,
    )
    direction[:, 0] = rng.uniform(0.0, 2.0 * np.pi, count)
    direction0[:, 0] = direction[:, 0]

    alpha = 0.7
    noise_scale = math.sqrt(1.0 - alpha ** 2)
    for age in range(1, life):
        velocity[:, age] = (
            alpha * velocity[:, age - 1]
            + (1.0 - alpha) * args.mean_velocity
            + noise_scale * rng.normal(0.0, 0.01, count)
        )
        direction[:, age] = (
            alpha * direction[:, age - 1]
            + (1.0 - alpha) * direction0[:, age - 1]
            + noise_scale * rng.normal(0.0, 0.01, count)
        )
        direction0[:, age] = direction0[:, age - 1]
        positions[:, age] = positions[:, age - 1]
        positions[:, age, 0] += velocity[:, age] * np.cos(direction[:, age]) * args.Delta_t
        positions[:, age, 1] += velocity[:, age] * np.sin(direction[:, age]) * args.Delta_t

        x_low = positions[:, age, 0] < bounds[:, 0]
        x_high = positions[:, age, 0] > bounds[:, 1]
        y_low = positions[:, age, 1] < bounds[:, 2]
        y_high = positions[:, age, 1] > bounds[:, 3]
        positions[x_low, age, 0] = 2.0 * bounds[x_low, 0] - positions[x_low, age, 0]
        positions[x_high, age, 0] = 2.0 * bounds[x_high, 1] - positions[x_high, age, 0]
        positions[y_low, age, 1] = 2.0 * bounds[y_low, 2] - positions[y_low, age, 1]
        positions[y_high, age, 1] = 2.0 * bounds[y_high, 3] - positions[y_high, age, 1]
        vertical, horizontal = x_low | x_high, y_low | y_high
        corner = vertical & horizontal
        direction[corner, age] = (direction[corner, age] + np.pi) % (2.0 * np.pi)
        only_vertical = vertical & ~corner
        only_horizontal = horizontal & ~corner
        direction[only_vertical, age] = (np.pi - direction[only_vertical, age]) % (2.0 * np.pi)
        direction[only_horizontal, age] = (2.0 * np.pi - direction[only_horizontal, age]) % (2.0 * np.pi)
        bounced = vertical | horizontal
        direction0[bounced, age] = direction[bounced, age]

    tasks = np.empty((count, life, 3), dtype=np.float64)
    tasks[:, :, 0] = rng.uniform(args.D_min, args.D_max, (count, life))
    tasks[:, :, 1] = rng.uniform(args.C_min, args.C_max, (count, life))
    tasks[:, :, 2] = rng.uniform(args.delay_min, args.delay_max, (count, life))
    return {
        "positions": positions,
        "velocity": velocity,
        "direction": direction,
        "direction0": direction0,
        "bounds": bounds,
        "tasks": tasks,
        "birth": np.repeat(np.arange(slots), arrivals),
    }


def coverage_score(points, uavs, radius):
    covered = np.any(np.linalg.norm(uavs[:, None, :] - points[None, :, :], axis=2) <= radius, axis=0)
    return float(np.mean(covered))


def choose_41_layout(args, learned_layout):
    small_ids = np.flatnonzero(
        (learned_layout[:, 0] >= 0) & (learned_layout[:, 0] <= 175)
        & (learned_layout[:, 1] >= 0) & (learned_layout[:, 1] <= 175)
    )
    if len(small_ids) != 2:
        raise ValueError(f"Expected learned 3-large/2-small deployment, found small IDs {small_ids.tolist()}")
    center = np.array([87.5, 87.5])
    retain = small_ids[np.argmin(np.linalg.norm(learned_layout[small_ids] - center, axis=1))]
    move = int(small_ids[small_ids != retain][0])
    fixed = np.delete(learned_layout, move, axis=0)

    axis = np.arange(205.0, 600.0, 10.0)
    gx, gy = np.meshgrid(axis, axis, indexing="xy")
    candidates = np.column_stack((gx.ravel(), gy.ravel()))
    small_axis = np.arange(2.5, 175.0, 5.0)
    big_axis = np.arange(202.5, 600.0, 5.0)
    sx, sy = np.meshgrid(small_axis, small_axis, indexing="xy")
    bx, by = np.meshgrid(big_axis, big_axis, indexing="xy")
    small_points = np.column_stack((sx.ravel(), sy.ravel()))
    big_points = np.column_stack((bx.ravel(), by.ravel()))
    fixed_small = np.any(
        np.linalg.norm(fixed[:, None, :] - small_points[None, :, :], axis=2) <= args.Cover_R, axis=0
    )
    fixed_big = np.any(
        np.linalg.norm(fixed[:, None, :] - big_points[None, :, :], axis=2) <= args.Cover_R, axis=0
    )
    best = None
    for target in candidates:
        if np.min(np.linalg.norm(fixed - target, axis=1)) < args.Dis_min:
            continue
        small_cov = fixed_small | (np.linalg.norm(small_points - target, axis=1) <= args.Cover_R)
        big_cov = fixed_big | (np.linalg.norm(big_points - target, axis=1) <= args.Cover_R)
        score = np.mean(small_cov) / 6.0 + 5.0 * np.mean(big_cov) / 6.0
        if best is None or score > best[0]:
            best = (float(score), target.copy())
    layout = learned_layout.copy()
    layout[move] = best[1]
    return layout, move, int(retain), best[0]


def waypoint_path(start, waypoints, steps, max_step):
    result = np.empty((steps + 1, 2), dtype=np.float64)
    result[0] = start
    targets = [np.asarray(point, dtype=np.float64) for point in waypoints]
    target_id = 0
    for step in range(steps):
        remaining = max_step
        position = result[step].copy()
        while remaining > 1e-10 and target_id < len(targets):
            delta = targets[target_id] - position
            distance = np.linalg.norm(delta)
            if distance <= remaining:
                position = targets[target_id].copy()
                remaining -= distance
                target_id += 1
            else:
                position += delta * remaining / distance
                remaining = 0.0
        result[step + 1] = position
    return result


def choose_safe_counterfactual_path(learned_path, moved_uav, target, args):
    detours = [
        [],
        [[200.0, 550.0]],
        [[550.0, 200.0]],
        [[200.0, 600.0]],
        [[600.0, 200.0]],
        [[175.0, 600.0]],
        [[600.0, 175.0]],
    ]
    feasible = []
    for detour in detours:
        candidate = learned_path.copy()
        points = detour + [target.tolist()]
        candidate[:, moved_uav] = waypoint_path(
            learned_path[0, moved_uav], points, len(learned_path) - 1, args.v_max * args.Delta_t
        )
        pairwise = np.linalg.norm(
            candidate[:, :, None, :] - candidate[:, None, :, :], axis=3
        )
        pairwise[:, np.arange(args.n_UAVs), np.arange(args.n_UAVs)] = np.inf
        if np.min(pairwise) >= args.Dis_min:
            travel = float(np.sum(np.linalg.norm(np.diff(candidate[:, moved_uav], axis=0), axis=1)))
            feasible.append((travel, candidate, detour, float(np.min(pairwise))))
    if not feasible:
        raise RuntimeError("No safe deterministic path to the 4/1 target was found")
    return min(feasible, key=lambda item: item[0])


def flight_actions(path, step, args):
    delta = path[step + 1] - path[step]
    distance = np.linalg.norm(delta, axis=1)
    action = np.zeros((args.n_UAVs, 2), dtype=np.float64)
    moving = distance > 1e-10
    if getattr(args, "cartesian_flight", False):
        action[moving] = delta[moving] / (args.v_max * args.Delta_t)
        return action
    action[moving, 0] = np.mod(np.arctan2(delta[moving, 1], delta[moving, 0]), 2.0 * np.pi) / (2.0 * np.pi)
    action[:, 1] = np.clip(distance / (args.v_max * args.Delta_t), 0.0, 1.0)
    return action


def inject_state(env, trace, active, path, step, args):
    env.time_step = step + 1
    env.uav_positions[:, :2] = path[step]
    env.gu_positions[:] = [0.0, 0.0, args.H_GU]
    env.gu_velocities[:] = 0.0
    env.gu_directions[:] = 0.0
    env.gu_directions_0[:] = 0.0
    env.gu_tasks[:] = 0.0
    env.active_md_mask[:] = False
    env.md_remaining_lifetime[:] = 0
    env.md_session_ids[:] = -1
    for md_slot, candidate_id in active.items():
        age = step - int(trace["birth"][candidate_id])
        env.active_md_mask[md_slot] = True
        env.gu_positions[md_slot, :2] = trace["positions"][candidate_id, age]
        env.gu_velocities[md_slot] = trace["velocity"][candidate_id, age]
        env.gu_directions[md_slot] = trace["direction"][candidate_id, age]
        env.gu_directions_0[md_slot] = trace["direction0"][candidate_id, age]
        env.gu_tasks[md_slot] = trace["tasks"][candidate_id, age]
        env.md_remaining_lifetime[md_slot] = args.md_lifetime_max - age
        env.md_session_ids[md_slot] = candidate_id
        env.x_min_all_gus[md_slot], env.x_max_all_gus[md_slot], env.y_min_all_gus[md_slot], env.y_max_all_gus[md_slot] = trace["bounds"][candidate_id]
    env._update_distance_matrices()
    env._calculate_channel_gains()
    env.nearby_gus_of_uavs = env.get_nearby_users_sorted_all()
    env.complete_task[:] = 0.0
    env.self_complete_task[:] = 0.0


def admit_births(active, trace, step, uav_positions, args):
    admitted = 0
    free = [slot for slot in range(args.n_GUs) if slot not in active]
    ids = np.flatnonzero(trace["birth"] == step)
    for candidate_id in ids:
        position = trace["positions"][candidate_id, 0]
        if np.any(np.linalg.norm(uav_positions - position, axis=1) <= args.Cover_R):
            if not free:
                raise RuntimeError("External dynamic-MD capacity was exceeded")
            active[free.pop(0)] = int(candidate_id)
            admitted += 1
    return admitted


def advance_population(active, trace, step, next_uavs, args):
    expired = departed = 0
    for md_slot, candidate_id in list(active.items()):
        age = step - int(trace["birth"][candidate_id])
        if age + 1 >= args.md_lifetime_max:
            del active[md_slot]
            expired += 1
        elif not np.any(
            np.linalg.norm(next_uavs - trace["positions"][candidate_id, age + 1], axis=1) <= args.Cover_R
        ):
            del active[md_slot]
            departed += 1
    return expired, departed


def rule_action(env, path, step, args):
    action = np.zeros((args.n_UAVs, 2 + 3 * args.n_GUs), dtype=np.float64)
    action[:, :2] = flight_actions(path, step, args)
    offload = action[:, 2:2 + args.n_GUs]
    bandwidth = action[:, 2 + args.n_GUs:2 + 2 * args.n_GUs]
    computation = action[:, 2 + 2 * args.n_GUs:]
    proposed = [[] for _ in range(args.n_UAVs)]
    for gu_id in np.flatnonzero(env.active_md_mask):
        distances = env.uav_gu_distances_2d[:, gu_id]
        uav_id = int(np.argmin(distances))
        if distances[uav_id] <= args.Cover_R:
            proposed[uav_id].append((float(distances[uav_id]), int(gu_id)))
    for uav_id, users in enumerate(proposed):
        selected = [gu_id for _, gu_id in sorted(users)[:args.max_GUs_in_range]]
        if selected:
            offload[uav_id, selected] = 1.0
            bandwidth[uav_id, selected] = 1.0 / len(selected)
            computation[uav_id, selected] = 1.0 / len(selected)
    env.proposed_offload_actions = offload.copy()
    return action


def actor_action(env, path, step, actors, normers, rnn_states, masks, args):
    obs = normalize_obs(normers, env.get_local_obs())
    available = env.get_local_avail_actions()
    result = []
    with torch.no_grad():
        for agent_id, actor in enumerate(actors):
            action, _, next_rnn = actor(
                obs[agent_id:agent_id + 1],
                rnn_states[agent_id:agent_id + 1],
                masks[agent_id:agent_id + 1],
                available[agent_id:agent_id + 1],
                deterministic=True,
            )
            result.append(action.cpu().numpy()[0])
            rnn_states[agent_id] = next_rnn.cpu().numpy()[0]
    result = np.asarray(result)
    result[:, :2] = flight_actions(path, step, args)
    return result


def run_condition(args, trace, path, allocator, actors=None, normers=None):
    steps = len(path) - 1
    env = WrappedMECEnv(args=args).env
    env.reset()
    # Population transitions are supplied from the common trace below.
    env._advance_dynamic_md_population = lambda: None
    active = {}
    admitted = admit_births(active, trace, 0, path[0], args)
    expired = departed = 0
    active_sum = completed_sum = served_sum = 0
    training_return = 0.0
    rnn_states = np.zeros((args.n_UAVs, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((args.n_UAVs, 1), dtype=np.float32)

    for step in range(steps):
        inject_state(env, trace, active, path, step, args)
        active_count = len(active)
        active_sum += active_count
        if allocator == "rule":
            action = rule_action(env, path, step, args)
            rewards = env.calculate_reward(action)
            service = action[:, 2:2 + args.n_GUs]
        else:
            raw = actor_action(env, path, step, actors, normers, rnn_states, masks, args)
            rewards = env.calculate_local_reward(raw)
            service = env.proposed_offload_actions
        training_return += float(np.mean(rewards))
        completed_sum += int(np.sum(env.complete_task[env.active_md_mask]))
        served_sum += int(np.sum(np.any(service[:, env.active_md_mask] > 0, axis=0)))

        just_expired, just_departed = advance_population(active, trace, step, path[step + 1], args)
        expired += just_expired
        departed += just_departed
        if step + 1 < steps:
            admitted += admit_births(active, trace, step + 1, path[step + 1], args)

    env.close()
    denominator = max(active_sum, 1)
    return {
        "system_performance_true_all_GUs": float(env.system_performance_true_all_GUs[0]),
        "system_performance_equivalent_full_GUs": float(env.system_performance_equivalent_full_GUs[0]),
        "system_performance": float(env.system_performance[0]),
        "training_return": training_return,
        "mean_active_mds": active_sum / steps,
        "admission_ratio": admitted / (args.md_arrivals_max * float(steps)),
        "completion_ratio": completed_sum / denominator,
        "served_ratio": served_sum / denominator,
        "admitted": admitted,
        "expired": expired,
        "departed": departed,
    }


def paired_summary(rows, baseline, counterfactual, metric):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["trace_seed"], {})[row["condition"]] = row[metric]
    deltas = np.array([values[counterfactual] - values[baseline] for values in by_seed.values()])
    mean = float(np.mean(deltas))
    sem = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
    baseline_values = np.array([values[baseline] for values in by_seed.values()])
    return {
        "baseline": baseline,
        "counterfactual": counterfactual,
        "metric": metric,
        "n_paired_traces": len(deltas),
        "baseline_mean": float(np.mean(baseline_values)),
        "counterfactual_mean": float(np.mean(baseline_values + deltas)),
        "mean_delta_41_minus_32": mean,
        "relative_delta_percent": 100.0 * mean / float(np.mean(baseline_values)),
        "normal_95ci_delta": [mean - 1.96 * sem, mean + 1.96 * sem],
        "wins_41": int(np.sum(deltas > 0)),
        "ties": int(np.sum(deltas == 0)),
        "losses_41": int(np.sum(deltas < 0)),
    }


def main():
    cli = parse_cli()
    checkpoint_dir = cli.checkpoint_dir.resolve()
    output_dir = cli.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    args = load_args(checkpoint_dir)
    actors, normers = load_policy(args, checkpoint_dir)

    episode = np.load(cli.baseline_episode.resolve())
    learned_path = np.asarray(episode["uav_positions"], dtype=np.float64)
    steps = int(args.episode_length)
    if learned_path.shape != (steps + 1, args.n_UAVs, 2):
        raise ValueError(f"Unexpected baseline path shape {learned_path.shape}")
    learned_layout = learned_path[-100:].mean(axis=0)
    layout_41, moved_uav, retained_small_uav, target_score = choose_41_layout(args, learned_layout)
    travel_distance, counterfactual_path, detour, path_min_separation = choose_safe_counterfactual_path(
        learned_path, moved_uav, layout_41[moved_uav], args
    )
    static_32 = np.repeat(learned_layout[None, :, :], cli.steady_slots + 1, axis=0)
    static_41 = np.repeat(layout_41[None, :, :], cli.steady_slots + 1, axis=0)

    speed = np.linalg.norm(np.diff(counterfactual_path, axis=0), axis=2) / args.Delta_t
    pairwise = np.linalg.norm(
        counterfactual_path[:, :, None, :] - counterfactual_path[:, None, :, :], axis=3
    )
    pairwise[:, np.arange(args.n_UAVs), np.arange(args.n_UAVs)] = np.inf
    if np.max(speed) > args.v_max + 1e-8 or np.min(pairwise) < args.Dis_min:
        raise RuntimeError("Counterfactual trajectory violates speed or safety constraints")

    conditions = {
        "transition_32_rule": (learned_path, "rule"),
        "transition_41_rule": (counterfactual_path, "rule"),
        "transition_32_actor": (learned_path, "actor"),
        "transition_41_actor": (counterfactual_path, "actor"),
        "steady_32_rule": (static_32, "rule"),
        "steady_41_rule": (static_41, "rule"),
        "steady_32_actor": (static_32, "actor"),
        "steady_41_actor": (static_41, "actor"),
    }
    rows = []
    for trial in range(cli.trials):
        seed = cli.seed_base + trial
        trace = make_trace(args, seed, max(steps, cli.steady_slots))
        trial_results = {}
        for name, (path, allocator) in conditions.items():
            result = run_condition(args, trace, path, allocator, actors, normers)
            trial_results[name] = result
            rows.append({"trace_seed": seed, "condition": name, **result})
        print(
            f"trace={seed} transition rule delta="
            f"{trial_results['transition_41_rule']['system_performance_true_all_GUs'] - trial_results['transition_32_rule']['system_performance_true_all_GUs']:.3f} "
            f"actor delta={trial_results['transition_41_actor']['system_performance_true_all_GUs'] - trial_results['transition_32_actor']['system_performance_true_all_GUs']:.3f}",
            flush=True,
        )

    # Deterministic replay is the central validity check for paired random traces.
    check_trace = make_trace(args, cli.seed_base, steps)
    check_a = run_condition(args, check_trace, learned_path, "rule")
    check_b = run_condition(args, check_trace, learned_path, "rule")
    if check_a != check_b:
        raise AssertionError("Identical path and common trace did not replay exactly")

    fieldnames = list(rows[0])
    with (output_dir / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    comparisons = []
    for phase in ("transition", "steady"):
        for allocator in ("rule", "actor"):
            for metric in (
                "system_performance_true_all_GUs",
                "system_performance_equivalent_full_GUs",
                "mean_active_mds",
                "admission_ratio",
                "completion_ratio",
                "served_ratio",
            ):
                comparisons.append(
                    paired_summary(rows, f"{phase}_32_{allocator}", f"{phase}_41_{allocator}", metric)
                )

    summary = {
        "status": "completed",
        "protocol": {
            "common_random_trace": "same candidate births, 20-slot reflected Gauss-Markov paths, and tasks",
            "rule_allocator": "nearest covered UAV, at most 20 users/UAV, equal bandwidth and CPU",
            "actor_allocator": "saved deterministic MAPPO resource heads; flight heads replaced by prescribed path",
            "reward_engine": "production MEC.calculate_reward / calculate_local_reward",
            "transition_32": "latest MAPPO deterministic trajectory",
            "transition_41": "same four UAV trajectories; one redundant small-region UAV moves straight to optimized large-region target",
            "steady": "both deployments hover for all slots",
            "candidate_mds_per_slot": 3,
            "lifetime_slots": 20,
            "trials": cli.trials,
            "seed_base": cli.seed_base,
        },
        "layout": {
            "learned_3large_2small": learned_layout.tolist(),
            "counterfactual_4large_1small": layout_41.tolist(),
            "moved_uav_zero_based": moved_uav,
            "retained_small_uav_zero_based": retained_small_uav,
            "optimized_weighted_coverage_score": target_score,
            "counterfactual_max_speed": float(np.max(speed)),
            "counterfactual_min_uav_separation": float(np.min(pairwise)),
            "counterfactual_detour_waypoints": detour,
            "moved_uav_travel_distance": travel_distance,
        },
        "validations": {
            "identical_rule_replay_exact": True,
            "counterfactual_speed_valid": True,
            "counterfactual_safety_valid": True,
        },
        "comparisons": comparisons,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary["layout"], ensure_ascii=False, indent=2))
    for comparison in comparisons:
        if comparison["metric"] == "system_performance_true_all_GUs":
            print(json.dumps(comparison, ensure_ascii=False))
    print(output_dir)


if __name__ == "__main__":
    main()
