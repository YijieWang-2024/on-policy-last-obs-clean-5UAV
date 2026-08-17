"""Evaluate four final association-threshold checkpoints on common seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_attention import R_Actor_Attention
from onpolicy.envs.mec.env_maker import WrappedMECEnv
from onpolicy.envs.mec.mec import N_0, p_t
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.scripts.eval.render_dynamic_mappo_episode import (
    checkpoint_manifest_step,
    prepare_policy_inputs,
    snapshot_checkpoint,
)

RESULTS = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
RUNS = {
    "psi0p3_seed32_remote": HERE / "checkpoints" / "remote_psi0p3" / "run1",
    "psi0p5_seed2_local": RESULTS / (
        "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
        "vmax30_psi0p5_nofilter_seed2_60m_20260814"
    ) / "run1",
    "psi0p7_seed2_local": RESULTS / (
        "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
        "vmax30_psi0p7_nofilter_seed2_60m_20260814"
    ) / "run1",
    "psi0p9_seed32_local": RESULTS / (
        "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
        "vmax30_psi0p9_nofilter_seed32_60m_20260815"
    ) / "run1",
}

METRICS = (
    "total_reward",
    "raw_association_score_pass_rate",
    "raw_association_score_mean",
    "active_task_count",
    "successful_task_count",
    "all_task_completion_rate",
    "offloaded_task_count",
    "successful_offloaded_task_count",
    "offload_ratio",
    "task_offloading_success_rate",
    "offload_yield",
    "effective_md_association_rate",
    "not_offloaded_task_count",
    "successful_local_task_count",
    "local_failure_count",
    "local_completion_rate",
    "local_yield",
    "local_share",
    "md_admission_ratio",
    "complete_task_ratio_env",
    "system_performance_true_all_GUs",
    "system_gain_per_active_task",
    "delay_true_all_GUs",
    "energy_true_all_GUs",
    "energy_all_GUs_UAVs",
    "average_active_mds",
    "serving_uav_fraction",
    "bandwidth_allocation_occupancy",
    "cpu_allocation_occupancy",
    "bandwidth_allocation_when_serving",
    "cpu_allocation_when_serving",
    "bandwidth_allocated_utilization",
    "cpu_allocated_utilization",
    "bandwidth_useful_utilization",
    "cpu_useful_utilization",
    "bandwidth_failure_waste_ratio",
    "cpu_failure_waste_ratio",
    "cpu_demand_load_system",
    "cpu_demand_load_serving",
    "transmission_time_load_system",
    "transmission_time_load_serving",
    "final_small_uav_count",
    "final_large_uav_count",
    "deployment_1plus4_final",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scalar_info(info, key):
    value = info.get(key)
    if value is None:
        return float("nan")
    array = np.asarray(value, dtype=float)
    return float(np.mean(array)) if array.size else float("nan")


def vector_info(info, key, length):
    """Return a final-info vector without collapsing UAV-specific values."""
    value = info.get(key)
    if value is None:
        return np.full(length, np.nan, dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == length:
        return array.copy()
    if array.size == 1:
        return np.repeat(array, length)
    return np.full(length, np.nan, dtype=float)


def load_policy(run_dir: Path, checkpoint_dir: Path):
    with (run_dir / "args.json").open("r", encoding="utf-8") as handle:
        args = Namespace(**json.load(handle))
    args.n_rollout_threads = 1
    args.n_training_threads = 1
    args.model_dir = str(checkpoint_dir)
    env = WrappedMECEnv(args=args)
    device = torch.device("cpu")
    actor_class = R_Actor_Attention if args.use_atten_actor else R_Actor
    actors, normers = [], []
    for agent_id in range(int(args.n_UAVs)):
        actor = actor_class(args, env.observation_space[agent_id], env.action_space[agent_id], device)
        checkpoint = checkpoint_dir / f"actor_agent{agent_id}.pt"
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        actor.load_state_dict(state)
        actor.eval()
        actors.append(actor)
        normer = Normer(args=args, obs_space=env.observation_space[agent_id], states_space=env.share_observation_space[agent_id])
        normer.load(checkpoint_dir / f"normer{agent_id}.pkl")
        normers.append(normer)
    return args, env, actors, normers


def process_full_actions(env, raw_actions):
    """Match MEC.step: local slots -> full MD arrays -> legal actions."""
    transformed = env.env.transform_uav_actions(raw_actions)
    return np.asarray(env.env.process_actions(transformed), dtype=np.float32)


def full_action_components(args, processed):
    if bool(args.fix_uav_pos):
        n_gus = int(args.n_GUs)
        return processed[:, :n_gus], processed[:, n_gus:2 * n_gus], processed[:, 2 * n_gus:]
    if bool(args.ave_resource) or bool(args.ave_bandwidth) or bool(args.nearest_associate):
        raise ValueError("Expected four-head continuous association actions")
    n_gus = int(args.n_GUs)
    return (
        processed[:, 2:2 + n_gus],
        processed[:, 2 + n_gus:2 + 2 * n_gus],
        processed[:, 2 + 2 * n_gus:2 + 3 * n_gus],
    )


def evaluate_episode(
    args,
    env,
    actors,
    normers,
    seed,
    checkpoint_step,
    measurement_start_step=0,
    deterministic=True,
    resource_normers=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    raw_obs, raw_states, available_actions, _, attention_mask = env.reset()
    obs, states = prepare_policy_inputs(normers, raw_obs, raw_states)
    resource_obs = None
    if resource_normers is not None:
        resource_obs, _ = prepare_policy_inputs(resource_normers, raw_obs, raw_states)

    n_uavs = int(args.n_UAVs)
    steps = int(args.episode_length)
    measurement_start_step = max(0, min(int(measurement_start_step), steps))
    max_slots = int(args.max_GUs_in_range)
    rnn_states = np.zeros((n_uavs, int(args.recurrent_N), int(args.hidden_size)), dtype=np.float32)
    masks = np.ones((n_uavs, 1), dtype=np.float32)

    raw_pass, raw_den, raw_score = 0, 0, 0.0
    active_count, complete_count = 0, 0
    offload_count, offload_complete_count = 0, 0
    local_count, local_complete_count = 0, 0
    per_uav_offload_count = np.zeros(n_uavs, dtype=float)
    per_uav_offload_complete_count = np.zeros(n_uavs, dtype=float)
    per_uav_service_steps = np.zeros(n_uavs, dtype=float)
    per_uav_allocated_bw = np.zeros(n_uavs, dtype=float)
    per_uav_useful_bw = np.zeros(n_uavs, dtype=float)
    per_uav_allocated_cpu = np.zeros(n_uavs, dtype=float)
    per_uav_useful_cpu = np.zeros(n_uavs, dtype=float)
    per_uav_reward = np.zeros(n_uavs, dtype=float)
    resource_steps, serving_steps = 0, 0
    bw_sum = cpu_sum = bw_serving_sum = cpu_serving_sum = 0.0
    allocated_bw_total = allocated_cpu_total = 0.0
    useful_bw_total = useful_cpu_total = 0.0
    cpu_load_system = cpu_load_serving = 0.0
    tx_load_system = tx_load_serving = 0.0
    total_reward = 0.0
    tail_system_performance = 0.0
    tail_delay = 0.0
    tail_energy = 0.0
    tail_energy_all_uavs = 0.0
    tail_per_uav_system_performance = np.zeros(n_uavs, dtype=float)
    tail_candidates = 0.0
    tail_admitted = 0.0
    tail_counter_available = True
    tail_cumulative_available = False
    inner = getattr(env, "env", env)

    def cumulative_snapshot(name):
        value = getattr(inner, name, None)
        if value is None:
            return None
        return np.asarray(value, dtype=float).copy()

    final_info = {}
    trajectory = [np.asarray(env.uav_positions[:, :2], dtype=np.float32).copy()]

    with torch.no_grad():
        for step_index in range(steps):
            measure = step_index >= measurement_start_step
            pre_active_ids = np.flatnonzero(np.asarray(env.active_md_mask, dtype=bool))
            valid_slots = np.asarray(env.nearby_gus_of_uavs[:, :max_slots] != -1, dtype=bool)
            raw_actions = []
            for agent_id, actor in enumerate(actors):
                kwargs = {}
                if args.use_atten_actor:
                    kwargs["attention_active_mask"] = attention_mask[agent_id:agent_id + 1]
                if resource_normers is not None:
                    kwargs["resource_obs"] = resource_obs[agent_id:agent_id + 1]
                action, _, next_rnn = actor(
                    obs[agent_id:agent_id + 1],
                    rnn_states[agent_id:agent_id + 1],
                    masks[agent_id:agent_id + 1],
                    available_actions[agent_id:agent_id + 1],
                    deterministic=deterministic,
                    **kwargs,
                )
                raw_actions.append(action.cpu().numpy()[0])
                rnn_states[agent_id] = next_rnn.cpu().numpy()[0]
            raw_actions = np.asarray(raw_actions, dtype=np.float32)

            scores = np.clip(raw_actions[:, 2:2 + max_slots], 0.0, 1.0)
            threshold = float(args.association_threshold)
            if measure:
                raw_pass += int(np.sum((scores >= threshold) & valid_slots))
                raw_den += int(np.sum(valid_slots))
                raw_score += float(np.sum(scores[valid_slots]))

            processed = process_full_actions(env, raw_actions)
            association, bandwidth, computation = full_action_components(args, processed)
            selected = np.any(association > 1e-8, axis=0)
            offloaded_ids = pre_active_ids[selected[pre_active_ids]]
            selected_matrix = association > 1e-8
            active_selected_matrix = np.zeros_like(selected_matrix, dtype=bool)
            if pre_active_ids.size:
                active_selected_matrix[:, pre_active_ids] = selected_matrix[:, pre_active_ids]
            resource_selected_matrix = active_selected_matrix if measurement_start_step > 0 else selected_matrix
            if measure:
                per_uav_offload_count += np.sum(active_selected_matrix, axis=1)
                per_uav_service_steps += np.sum(active_selected_matrix, axis=1) > 0
                per_uav_allocated_bw += np.sum(bandwidth * active_selected_matrix, axis=1)
                per_uav_allocated_cpu += np.sum(computation * active_selected_matrix, axis=1)
                allocated_bw_total += float(np.sum(bandwidth * resource_selected_matrix))
                allocated_cpu_total += float(np.sum(computation * resource_selected_matrix))

                bw_alloc = np.sum(bandwidth * resource_selected_matrix, axis=1)
                cpu_alloc = np.sum(computation * resource_selected_matrix, axis=1)
                serving = np.sum(resource_selected_matrix, axis=1) > 0
                resource_steps += 1
                serving_steps += int(np.sum(serving))
                bw_sum += float(np.sum(bw_alloc))
                cpu_sum += float(np.sum(cpu_alloc))
                bw_serving_sum += float(np.sum(bw_alloc[serving]))
                cpu_serving_sum += float(np.sum(cpu_alloc[serving]))

            if measure and offloaded_ids.size:
                cycles = np.asarray(env.gu_tasks[offloaded_ids, 1], dtype=float)
                cpu_load_system += float(np.sum(cycles) / (float(env.F_m) * float(env.Delta_t) * n_uavs))
                cpu_load_serving += float(np.sum(cycles) / (float(env.F_m) * float(env.Delta_t) * max(int(np.sum(serving)), 1)))
                for gu_id in offloaded_ids:
                    uav_id = int(np.argmax(association[:, gu_id]))
                    allocated_bw = float(bandwidth[uav_id, gu_id]) * float(env.B)
                    if allocated_bw <= 0.0:
                        continue
                    rate = allocated_bw * np.log2(
                        1.0 + p_t * float(env.channel_gains[uav_id, gu_id])
                        / (N_0 * (allocated_bw + 1e-10))
                    )
                    tx_delay = float(env.gu_tasks[gu_id, 0]) / max(float(rate), 1e-10)
                    tx_load_system += tx_delay / (float(env.Delta_t) * n_uavs)
                    tx_load_serving += tx_delay / (float(env.Delta_t) * max(int(np.sum(serving)), 1))

            if measure:
                prev_system = cumulative_snapshot("system_performance_true_all_GUs")
                prev_individual = cumulative_snapshot("system_performance_individual")
                prev_delay = cumulative_snapshot("delay_true_all_GUs")
                prev_energy = cumulative_snapshot("energy_true_all_GUs")
                prev_energy_all = cumulative_snapshot("energy_all_GUs_UAVs")
                prev_candidates = getattr(inner, "dynamic_md_candidates", None)
                prev_admitted = getattr(inner, "dynamic_md_admitted", None)
            else:
                prev_system = prev_individual = prev_delay = prev_energy = prev_energy_all = None
                prev_candidates = prev_admitted = None

            raw_obs, raw_states, reward, dones, info, available_actions, _, attention_mask = env.step(raw_actions)
            if measure:
                total_reward += float(np.sum(reward))
                per_uav_reward += np.asarray(reward, dtype=float).reshape(-1)[:n_uavs]
                post_system = cumulative_snapshot("system_performance_true_all_GUs")
                post_individual = cumulative_snapshot("system_performance_individual")
                post_delay = cumulative_snapshot("delay_true_all_GUs")
                post_energy = cumulative_snapshot("energy_true_all_GUs")
                post_energy_all = cumulative_snapshot("energy_all_GUs_UAVs")
                snapshots = (
                    prev_system, prev_individual, prev_delay, prev_energy, prev_energy_all,
                    post_system, post_individual, post_delay, post_energy, post_energy_all,
                )
                if all(item is not None for item in snapshots):
                    tail_system_performance += float(np.mean(post_system - prev_system))
                    tail_per_uav_system_performance += post_individual - prev_individual
                    tail_delay += float(np.mean(post_delay - prev_delay))
                    tail_energy += float(np.mean(post_energy - prev_energy))
                    tail_energy_all_uavs += float(np.mean(post_energy_all - prev_energy_all))
                    tail_cumulative_available = True
                else:
                    tail_cumulative_available = False
                post_candidates = getattr(inner, "dynamic_md_candidates", None)
                post_admitted = getattr(inner, "dynamic_md_admitted", None)
                if prev_candidates is not None and post_candidates is not None:
                    tail_candidates += float(post_candidates - prev_candidates)
                else:
                    tail_counter_available = False
                if prev_admitted is not None and post_admitted is not None:
                    tail_admitted += float(post_admitted - prev_admitted)
                else:
                    tail_counter_available = False
            final_info = info
            if measure and pre_active_ids.size:
                completed = np.asarray(env.complete_task[pre_active_ids], dtype=float) > 0.5
                active_count += int(pre_active_ids.size)
                complete_count += int(np.sum(completed))
                offload_count += int(offloaded_ids.size)
                local_mask = ~selected[pre_active_ids]
                local_count += int(np.sum(local_mask))
                local_complete_count += int(np.sum(completed[local_mask]))
                if offloaded_ids.size:
                    successful_mask = completed[selected[pre_active_ids]]
                    successful_ids = offloaded_ids[successful_mask]
                    offload_complete_count += int(np.sum(successful_mask))
                    successful_matrix = np.zeros_like(selected_matrix, dtype=bool)
                    for gu_id in successful_ids:
                        uav_id = int(np.argmax(association[:, gu_id]))
                        successful_matrix[uav_id, gu_id] = True
                    per_uav_offload_complete_count += np.sum(successful_matrix, axis=1)
                    per_uav_useful_bw += np.sum(bandwidth * successful_matrix, axis=1)
                    per_uav_useful_cpu += np.sum(computation * successful_matrix, axis=1)
                    useful_bw_total += float(np.sum(bandwidth * successful_matrix))
                    useful_cpu_total += float(np.sum(computation * successful_matrix))
            trajectory.append(np.asarray(env.uav_positions[:, :2], dtype=np.float32).copy())
            obs, states = prepare_policy_inputs(normers, raw_obs, raw_states)
            if resource_normers is not None:
                resource_obs, _ = prepare_policy_inputs(resource_normers, raw_obs, raw_states)
            masks[:] = 0.0 if np.all(dones) else 1.0

    trajectory = np.stack(trajectory, axis=0)
    final_positions = trajectory[-1]
    small = (final_positions[:, 0] < 200.0) & (final_positions[:, 1] < 200.0)
    large = (final_positions[:, 0] >= 200.0) & (final_positions[:, 1] >= 200.0)
    final_small = int(np.sum(small))
    final_large = int(np.sum(large))

    tail_mode = measurement_start_step > 0
    metric_steps = resource_steps
    if tail_mode:
        if tail_counter_available and tail_candidates > 0:
            metric_md_admission_ratio = tail_admitted / tail_candidates
        else:
            metric_md_admission_ratio = float("nan")
        metric_complete_task_ratio_env = (
            complete_count / active_count if active_count else float("nan")
        )
        if tail_cumulative_available:
            metric_system_performance = tail_system_performance
            metric_delay = tail_delay / metric_steps if metric_steps else float("nan")
            metric_energy = tail_energy / metric_steps if metric_steps else float("nan")
            metric_energy_all_uavs = (
                tail_energy_all_uavs / metric_steps if metric_steps else float("nan")
            )
            metric_per_uav_system_performance = tail_per_uav_system_performance
        else:
            metric_system_performance = float("nan")
            metric_delay = metric_energy = metric_energy_all_uavs = float("nan")
            metric_per_uav_system_performance = np.full(n_uavs, np.nan, dtype=float)
        metric_average_active_mds = active_count / metric_steps if metric_steps else float("nan")
        metric_cumulative_individual_reward = per_uav_reward
    else:
        metric_md_admission_ratio = scalar_info(final_info, "md_admission_ratio")
        metric_complete_task_ratio_env = scalar_info(final_info, "complete_task_ratio")
        metric_system_performance = scalar_info(final_info, "system_performance_true_all_GUs")
        metric_delay = scalar_info(final_info, "delay_true_all_GUs")
        metric_energy = scalar_info(final_info, "energy_true_all_GUs")
        metric_energy_all_uavs = scalar_info(final_info, "energy_all_GUs_UAVs")
        metric_average_active_mds = scalar_info(final_info, "average_active_mds")
        metric_per_uav_system_performance = vector_info(
            final_info, "system_performance_individual", n_uavs
        )
        metric_cumulative_individual_reward = vector_info(
            final_info, "cumulative_individual_reward", n_uavs
        )

    return {
        "checkpoint_step": int(checkpoint_step),
        "evaluation_seed": int(seed),
        "total_reward": total_reward,
        "raw_association_score_pass_rate": raw_pass / raw_den if raw_den else float("nan"),
        "raw_association_score_mean": raw_score / raw_den if raw_den else float("nan"),
        "active_task_count": active_count,
        "successful_task_count": complete_count,
        "all_task_completion_rate": complete_count / active_count if active_count else float("nan"),
        "offloaded_task_count": offload_count,
        "successful_offloaded_task_count": offload_complete_count,
        "offload_ratio": offload_count / active_count if active_count else float("nan"),
        "task_offloading_success_rate": offload_complete_count / offload_count if offload_count else float("nan"),
        "offload_yield": offload_complete_count / active_count if active_count else float("nan"),
        "effective_md_association_rate": offload_count / active_count if active_count else float("nan"),
        "not_offloaded_task_count": local_count,
        "successful_local_task_count": local_complete_count,
        "local_failure_count": local_count - local_complete_count,
        "local_completion_rate": local_complete_count / local_count if local_count else float("nan"),
        "local_yield": local_complete_count / active_count if active_count else float("nan"),
        "local_share": local_count / active_count if active_count else float("nan"),
        "md_admission_ratio": metric_md_admission_ratio,
        "complete_task_ratio_env": metric_complete_task_ratio_env,
        "system_performance_true_all_GUs": metric_system_performance,
        "system_gain_per_active_task": (
            metric_system_performance / active_count
            if active_count else float("nan")
        ),
        "delay_true_all_GUs": metric_delay,
        "energy_true_all_GUs": metric_energy,
        "energy_all_GUs_UAVs": metric_energy_all_uavs,
        "average_active_mds": metric_average_active_mds,
        "serving_uav_fraction": serving_steps / (resource_steps * n_uavs),
        "bandwidth_allocation_occupancy": bw_sum / (resource_steps * n_uavs),
        "cpu_allocation_occupancy": cpu_sum / (resource_steps * n_uavs),
        "bandwidth_allocation_when_serving": bw_serving_sum / serving_steps if serving_steps else float("nan"),
        "cpu_allocation_when_serving": cpu_serving_sum / serving_steps if serving_steps else float("nan"),
        "bandwidth_allocated_utilization": allocated_bw_total / (resource_steps * n_uavs),
        "cpu_allocated_utilization": allocated_cpu_total / (resource_steps * n_uavs),
        "bandwidth_useful_utilization": useful_bw_total / (resource_steps * n_uavs),
        "cpu_useful_utilization": useful_cpu_total / (resource_steps * n_uavs),
        "bandwidth_failure_waste_ratio": (
            (allocated_bw_total - useful_bw_total) / allocated_bw_total
            if allocated_bw_total else float("nan")
        ),
        "cpu_failure_waste_ratio": (
            (allocated_cpu_total - useful_cpu_total) / allocated_cpu_total
            if allocated_cpu_total else float("nan")
        ),
        "cpu_demand_load_system": cpu_load_system / resource_steps,
        "cpu_demand_load_serving": cpu_load_serving / resource_steps,
        "transmission_time_load_system": tx_load_system / resource_steps,
        "transmission_time_load_serving": tx_load_serving / resource_steps,
        "final_small_uav_count": final_small,
        "final_large_uav_count": final_large,
        "deployment_1plus4_final": float(final_small == 1 and final_large == 4),
        "per_uav_offloaded_task_count": per_uav_offload_count.tolist(),
        "per_uav_successful_offloaded_task_count": per_uav_offload_complete_count.tolist(),
        "per_uav_service_success_rate": (
            np.divide(
                per_uav_offload_complete_count,
                per_uav_offload_count,
                out=np.full(n_uavs, np.nan),
                where=per_uav_offload_count > 0,
            ).tolist()
        ),
        "per_uav_service_steps": per_uav_service_steps.tolist(),
        "per_uav_service_step_fraction": (per_uav_service_steps / resource_steps).tolist(),
        "per_uav_allocated_bandwidth": per_uav_allocated_bw.tolist(),
        "per_uav_useful_bandwidth": per_uav_useful_bw.tolist(),
        "per_uav_allocated_cpu": per_uav_allocated_cpu.tolist(),
        "per_uav_useful_cpu": per_uav_useful_cpu.tolist(),
        "per_uav_episode_reward": per_uav_reward.tolist(),
        "per_uav_mean_reward": (per_uav_reward / resource_steps).tolist(),
        "per_uav_system_performance_individual": metric_per_uav_system_performance.tolist(),
        "per_uav_cumulative_individual_reward": metric_cumulative_individual_reward.tolist(),
    }


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    results = []
    for model, items in grouped.items():
        result = {"model": model, "episodes": len(items), "checkpoint_step": items[0]["checkpoint_step"]}
        for metric in METRICS:
            values = np.asarray([item[metric] for item in items], dtype=float)
            values = values[np.isfinite(values)]
            result[f"{metric}_n"] = int(values.size)
            result[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            result[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            result[f"{metric}_se"] = result[f"{metric}_std"] / np.sqrt(values.size) if values.size > 1 else 0.0
        results.append(result)
    return results


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(output_dir, aggregate_rows):
    import matplotlib.pyplot as plt

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    labels = {
        "psi0p3_seed32_remote": r"psi=0.3 (remote, s32)",
        "psi0p5_seed2_local": r"psi=0.5 (local, s2)",
        "psi0p7_seed2_local": r"psi=0.7 (local, s2)",
        "psi0p9_seed32_local": r"psi=0.9 (local, s32)",
    }
    order = [row["model"] for row in aggregate_rows]
    by_model = {row["model"]: row for row in aggregate_rows}
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    def error_bars(ax, metric, title, ylabel, scale=1.0):
        means = [by_model[m][f"{metric}_mean"] * scale for m in order]
        errors = [by_model[m][f"{metric}_std"] * scale for m in order]
        ax.bar(x, means, yerr=errors, capsize=4, color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in order], rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    error_bars(axes[0, 0], "system_performance_true_all_GUs", "Overall system gain", "system_performance_true_all_GUs")
    error_bars(axes[0, 1], "task_offloading_success_rate", "Offloaded-task success rate", "success rate (%)", 100.0)

    width = 0.36
    completion = [by_model[m]["all_task_completion_rate_mean"] * 100 for m in order]
    completion_err = [by_model[m]["all_task_completion_rate_std"] * 100 for m in order]
    offload = [by_model[m]["effective_md_association_rate_mean"] * 100 for m in order]
    offload_err = [by_model[m]["effective_md_association_rate_std"] * 100 for m in order]
    axes[1, 0].bar(x - width / 2, completion, width, yerr=completion_err, capsize=3, color="#0072B2", label="all-task completion")
    axes[1, 0].bar(x + width / 2, offload, width, yerr=offload_err, capsize=3, color="#E69F00", label="effective offload share")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([labels[m] for m in order], rotation=18, ha="right")
    axes[1, 0].set_title("Completion versus offloading load")
    axes[1, 0].set_ylabel("rate (%)")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8, loc="lower left")

    bw = [by_model[m]["bandwidth_allocation_occupancy_mean"] * 100 for m in order]
    cpu = [by_model[m]["cpu_allocation_occupancy_mean"] * 100 for m in order]
    axes[1, 1].bar(x - width / 2, bw, width, color="#56B4E9", label="bandwidth allocation occupancy")
    axes[1, 1].bar(x + width / 2, cpu, width, color="#F0E442", edgecolor="#555555", label="CPU allocation occupancy")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([labels[m] for m in order], rotation=18, ha="right")
    axes[1, 1].set_title("Resource allocation occupancy (code-level)")
    axes[1, 1].set_ylabel("normalized allocation (%)")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8, loc="lower left")

    fig.suptitle("Four final checkpoints: association-threshold reviewer metrics", fontsize=14)
    fig.savefig(output_dir / "psi_reviewer_metrics_summary.png", dpi=200)
    plt.close(fig)


def make_detailed_plot(output_dir, aggregate_rows):
    import matplotlib.pyplot as plt

    labels = {
        "psi0p3_seed32_remote": "psi=0.3\n(remote,s32)",
        "psi0p5_seed2_local": "psi=0.5\n(local,s2)",
        "psi0p7_seed2_local": "psi=0.7\n(local,s2)",
        "psi0p9_seed32_local": "psi=0.9\n(local,s32)",
    }
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    order = [row["model"] for row in aggregate_rows]
    by_model = {row["model"]: row for row in aggregate_rows}
    x = np.arange(len(order))
    width = 0.25
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    def grouped_rates(ax, specs, title):
        for index, (metric, label, color) in enumerate(specs):
            means = [by_model[m][f"{metric}_mean"] * 100.0 for m in order]
            errors = [by_model[m][f"{metric}_std"] * 100.0 for m in order]
            ax.bar(
                x + (index - (len(specs) - 1) / 2) * width,
                means,
                width,
                yerr=errors,
                capsize=3,
                color=color,
                label=label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in order])
        ax.set_ylim(0, 105)
        ax.set_ylabel("percent (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    gain = [by_model[m]["system_performance_true_all_GUs_mean"] for m in order]
    gain_err = [by_model[m]["system_performance_true_all_GUs_std"] for m in order]
    axes[0, 0].bar(x, gain, yerr=gain_err, capsize=4, color=colors, edgecolor="#333333")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([labels[m] for m in order])
    axes[0, 0].set_title("Episode system gain")
    axes[0, 0].set_ylabel("system_performance_true_all_GUs")
    axes[0, 0].grid(axis="y", alpha=0.25)

    grouped_rates(
        axes[0, 1],
        (
            ("offload_ratio", "OffloadRatio", "#0072B2"),
            ("task_offloading_success_rate", "OSR", "#D55E00"),
            ("offload_yield", "OffloadYield", "#009E73"),
        ),
        "Offloading decomposition",
    )
    grouped_rates(
        axes[0, 2],
        (
            ("all_task_completion_rate", "OverallCompletion", "#0072B2"),
            ("effective_md_association_rate", "Effective offload share", "#E69F00"),
        ),
        "Overall completion versus offloading load",
    )

    def resource_panel(ax, alloc_metric, useful_metric, waste_metric, title):
        alloc = [by_model[m][f"{alloc_metric}_mean"] * 100.0 for m in order]
        useful = [by_model[m][f"{useful_metric}_mean"] * 100.0 for m in order]
        alloc_err = [by_model[m][f"{alloc_metric}_std"] * 100.0 for m in order]
        useful_err = [by_model[m][f"{useful_metric}_std"] * 100.0 for m in order]
        ax.bar(x - width / 2, alloc, width, yerr=alloc_err, capsize=3, color="#56B4E9", label="allocated")
        ax.bar(x + width / 2, useful, width, yerr=useful_err, capsize=3, color="#009E73", label="useful (x*s)")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in order])
        ax.set_ylim(0, 105)
        ax.set_ylabel("capacity utilization (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    resource_panel(
        axes[1, 0],
        "bandwidth_allocated_utilization",
        "bandwidth_useful_utilization",
        "bandwidth_failure_waste_ratio",
        "Bandwidth: allocated vs useful",
    )
    resource_panel(
        axes[1, 1],
        "cpu_allocated_utilization",
        "cpu_useful_utilization",
        "cpu_failure_waste_ratio",
        "CPU: allocated vs useful",
    )

    bw_waste = [by_model[m]["bandwidth_failure_waste_ratio_mean"] * 100.0 for m in order]
    cpu_waste = [by_model[m]["cpu_failure_waste_ratio_mean"] * 100.0 for m in order]
    bw_waste_err = [by_model[m]["bandwidth_failure_waste_ratio_std"] * 100.0 for m in order]
    cpu_waste_err = [by_model[m]["cpu_failure_waste_ratio_std"] * 100.0 for m in order]
    axes[1, 2].bar(x - width / 2, bw_waste, width, yerr=bw_waste_err, capsize=3, color="#D55E00", label="bandwidth waste")
    axes[1, 2].bar(x + width / 2, cpu_waste, width, yerr=cpu_waste_err, capsize=3, color="#CC79A7", label="CPU waste")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels([labels[m] for m in order])
    axes[1, 2].set_ylabel("failed-offload resource fraction (%)")
    axes[1, 2].set_title("Resource wasted on failed offloads")
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].legend(fontsize=8, loc="upper left")

    fig.suptitle("Explicit reviewer metrics: active tasks, offloading yield, and useful resources", fontsize=14)
    fig.savefig(output_dir / "psi_reviewer_metrics_detailed.png", dpi=200)
    plt.close(fig)


def write_protocol(output_dir):
    protocol = """# psi_min reviewer-metric protocol

All four checkpoints are evaluated deterministically on the same episode seeds.

- Raw association pass rate: valid local association slots whose policy score is at least the checkpoint psi_min.
- Effective offload fraction: active MDs with a nonzero association after local-action transformation and environment post-processing.
- OffloadRatio: actual offloaded active task instances divided by all active task instances.
- Task offloading success rate: completed offloaded tasks divided by offloaded tasks. Completion is the environment complete_task flag after that step. The deadline filter is off, so the environment does not pre-cancel predicted-late offloads; late offloads remain failures.
- OffloadYield: completed offloaded task instances divided by all active task instances.
- Not-offloaded task count: active task instances that were not selected for offloading after action transformation.
- Successful local task count: not-offloaded task instances whose environment complete_task flag is true after the step.
- Local completion rate: successful local task count divided by not-offloaded task count.
- LocalYield: successful local task instances divided by all active task instances.
- LocalShare: not-offloaded active task instances divided by all active task instances.
- All-task completion rate: completed local and offloaded tasks divided by all active tasks. This is separate from offloading success rate.
- Resource allocation occupancy: normalized bandwidth/CPU allocation sums per UAV, averaged over UAVs and steps. Serving UAVs are normalized to one by the current code path, so this is an occupancy/scheduling statistic, not an independent physical-capacity utilization measurement.
- Allocated utilization: sum of x times normalized allocated bandwidth/CPU over all UAV-slots.
- Useful utilization: sum of x times s times normalized allocated bandwidth/CPU over all UAV-slots.
- Failure waste ratio: allocated resource assigned to x=1, s=0 divided by all allocated resource.
- Demand-equivalent loads: offloaded CPU cycles divided by total UAV CPU capacity in one slot, plus transmission time divided by system slot capacity.
- Overall system gain: environment metric system_performance_true_all_GUs.

The per-episode CSV preserves the exact numerator and denominator counts.
"""
    (output_dir / "metric_protocol.md").write_text(protocol, encoding="utf-8")


def main():
    cli = parse_args()
    if cli.episodes < 20:
        raise ValueError("Use at least 20 common test episodes")
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    cli.output_dir.mkdir(parents=True)
    (cli.output_dir / "snapshots").mkdir()
    write_protocol(cli.output_dir)
    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    rows = []
    metadata = {"episodes": cli.episodes, "test_seeds": seeds, "deterministic": True, "runs": {}}
    torch.set_num_threads(1)

    for model, run_dir in RUNS.items():
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        checkpoint_dir, files = snapshot_checkpoint(run_dir, cli.output_dir / "snapshots" / model)
        checkpoint_step = checkpoint_manifest_step(checkpoint_dir)
        if checkpoint_step is None:
            raise RuntimeError(f"Missing verified checkpoint manifest: {run_dir}")
        args, env, actors, normers = load_policy(run_dir, checkpoint_dir)
        metadata["runs"][model] = {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": checkpoint_step,
            "trained_psi": float(args.association_threshold),
            "checkpoint_files": files,
        }
        try:
            for index, seed in enumerate(seeds, 1):
                row = evaluate_episode(args, env, actors, normers, seed, checkpoint_step)
                row["model"] = model
                rows.append(row)
                if index == 1 or index % 5 == 0 or index == len(seeds):
                    print(json.dumps({"model": model, "episode": index, "episodes": len(seeds)}), flush=True)
        finally:
            env.close()

    aggregates = aggregate(rows)
    write_csv(cli.output_dir / "episode_metrics.csv", rows)
    write_csv(cli.output_dir / "aggregate_metrics.csv", aggregates)
    make_plot(cli.output_dir, aggregates)
    make_detailed_plot(cli.output_dir, aggregates)
    metadata["aggregate_metrics"] = aggregates
    (cli.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(cli.output_dir), "rows": len(rows), "plot": str(cli.output_dir / "psi_reviewer_metrics_summary.png")}, indent=2))


if __name__ == "__main__":
    main()
