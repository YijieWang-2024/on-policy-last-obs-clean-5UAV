"""Counterfactual evaluation of association-threshold effects.

The policy checkpoint is held fixed while the environment-side association
threshold is changed at evaluation time.  Every configuration uses the same
test seeds.  This separates a direct action-postprocessing effect from the
policy that was learned during training.

By default only the MEC environment threshold is changed.  The actor keeps
the threshold stored in its checkpoint; this avoids silently changing the
resource-head availability mask while testing the environment postprocessor.
Use ``--threshold-scope actor_and_env`` only for the secondary sensitivity
check that changes both layers.
"""

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
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.scripts.eval.render_dynamic_mappo_episode import (
    checkpoint_manifest_step,
    prepare_policy_inputs,
    snapshot_checkpoint,
)


RESULTS = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
RUNS = {
    "trained_psi0p3": RESULTS / (
        "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
        "vmax30_psi0p3_nofilter_seed2_60m_20260814"
    ) / "run1",
    "trained_psi0p5": RESULTS / (
        "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
        "vmax30_psi0p5_nofilter_seed2_60m_20260814"
    ) / "run1",
    "trained_psi0p7": RESULTS / (
        "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
        "vmax30_psi0p7_nofilter_seed2_60m_20260814"
    ) / "run1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=3001)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
    )
    parser.add_argument(
        "--threshold-scope",
        choices=("env_only", "actor_and_env"),
        default="env_only",
        help="change only MEC postprocessing, or both MEC and actor masks",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scalar_info(info: dict, key: str) -> float:
    value = info.get(key)
    if value is None:
        return float("nan")
    array = np.asarray(value, dtype=float)
    return float(np.mean(array)) if array.size else float("nan")


def load_policy(run_dir: Path, checkpoint_dir: Path):
    with (run_dir / "args.json").open("r", encoding="utf-8") as handle:
        args = Namespace(**json.load(handle))
    args.n_rollout_threads = 1
    args.n_training_threads = 1
    args.model_dir = str(checkpoint_dir)

    env = WrappedMECEnv(args=args)
    device = torch.device("cpu")
    actor_class = R_Actor_Attention if args.use_atten_actor else R_Actor
    actors = []
    normers = []
    for agent_id in range(int(args.n_UAVs)):
        actor = actor_class(
            args,
            env.observation_space[agent_id],
            env.action_space[agent_id],
            device,
        )
        checkpoint = checkpoint_dir / f"actor_agent{agent_id}.pt"
        try:
            state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint, map_location=device)
        actor.load_state_dict(state_dict)
        actor.eval()
        actors.append(actor)
        normer = Normer(
            args=args,
            obs_space=env.observation_space[agent_id],
            states_space=env.share_observation_space[agent_id],
        )
        normer.load(checkpoint_dir / f"normer{agent_id}.pkl")
        normers.append(normer)
    return args, env, actors, normers


def set_threshold(env, actors, threshold: float, scope: str) -> None:
    # WrappedMECEnv delegates attribute reads, but assignment must target the
    # actual MEC object explicitly.
    env.env.association_threshold = float(threshold)
    if scope == "actor_and_env":
        for actor in actors:
            actor.act.association_threshold = float(threshold)


def region_mask(positions: np.ndarray, region: str) -> np.ndarray:
    positions = np.asarray(positions)
    if region == "small":
        return (positions[..., 0] < 200.0) & (positions[..., 1] < 200.0)
    if region == "large":
        return (positions[..., 0] >= 200.0) & (positions[..., 1] >= 200.0)
    raise ValueError(region)


def evaluate_episode(
    args,
    env,
    actors,
    normers,
    seed: int,
    deterministic: bool,
    checkpoint_step: int,
    target_threshold: float,
    threshold_scope: str,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    set_threshold(env, actors, target_threshold, threshold_scope)
    obs, states, available_actions, _, attention_mask = env.reset()
    obs, states = prepare_policy_inputs(normers, obs, states)

    n_uavs = int(args.n_UAVs)
    steps = int(args.episode_length)
    max_slots = int(args.max_GUs_in_range)
    rnn_states = np.zeros(
        (n_uavs, int(args.recurrent_N), int(args.hidden_size)), dtype=np.float32
    )
    masks = np.ones((n_uavs, 1), dtype=np.float32)

    pass_count = np.zeros(n_uavs, dtype=np.float64)
    pass_denominator = np.zeros(n_uavs, dtype=np.float64)
    score_sum = np.zeros(n_uavs, dtype=np.float64)
    effective_count = np.zeros(n_uavs, dtype=np.float64)
    effective_denominator = np.zeros(n_uavs, dtype=np.float64)
    effective_small_count = np.zeros(n_uavs, dtype=np.float64)
    effective_large_count = np.zeros(n_uavs, dtype=np.float64)
    effective_small_denominator = 0.0
    effective_large_denominator = 0.0
    trajectory = [np.asarray(env.uav_positions[:, :2], dtype=np.float32).copy()]
    final_info = {}
    total_reward = 0.0

    with torch.no_grad():
        for _ in range(steps):
            valid_slots = np.asarray(
                env.nearby_gus_of_uavs[:, :max_slots] != -1,
                dtype=bool,
            )
            raw_actions = []
            for agent_id, actor in enumerate(actors):
                kwargs = (
                    {"attention_active_mask": attention_mask[agent_id:agent_id + 1]}
                    if args.use_atten_actor
                    else {}
                )
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
            association_scores = np.clip(
                raw_actions[:, 2:2 + max_slots], 0.0, 1.0
            )
            pass_count += np.sum(
                (association_scores >= target_threshold) & valid_slots,
                axis=1,
            )
            pass_denominator += np.sum(valid_slots, axis=1)
            score_sum += np.sum(association_scores * valid_slots, axis=1)

            obs, states, reward, dones, info, available_actions, _, attention_mask = env.step(
                raw_actions
            )
            total_reward += float(np.sum(reward))
            final_info = info

            active_ids = np.flatnonzero(env.active_md_mask)
            if active_ids.size:
                proposed = np.asarray(env.proposed_offload_actions)[:, active_ids]
                selected = proposed > 1e-8
                effective_count += np.sum(selected, axis=1)
                effective_denominator += float(active_ids.size)
                md_positions = np.asarray(env.gu_positions[active_ids, :2])
                small_md = region_mask(md_positions, "small")
                large_md = region_mask(md_positions, "large")
                effective_small_count += np.sum(selected[:, small_md], axis=1)
                effective_large_count += np.sum(selected[:, large_md], axis=1)
                effective_small_denominator += float(np.sum(small_md))
                effective_large_denominator += float(np.sum(large_md))

            trajectory.append(np.asarray(env.uav_positions[:, :2], dtype=np.float32).copy())
            obs, states = prepare_policy_inputs(normers, obs, states)
            masks[:] = 0.0 if np.all(dones) else 1.0

    trajectory = np.stack(trajectory, axis=0)
    small_position_fraction = np.mean(region_mask(trajectory, "small"), axis=0)
    large_position_fraction = np.mean(region_mask(trajectory, "large"), axis=0)
    final_positions = trajectory[-1]

    episode_row = {
        "checkpoint_step": int(checkpoint_step),
        "evaluation_seed": int(seed),
        "action_mode": "deterministic" if deterministic else "stochastic",
        "trained_threshold": float(args.association_threshold),
        "evaluation_threshold": float(target_threshold),
        "threshold_scope": threshold_scope,
        "total_reward": total_reward,
        "env_association_score_pass_rate": (
            float(np.sum(pass_count) / np.sum(pass_denominator))
            if np.sum(pass_denominator) else float("nan")
        ),
        "effective_md_association_rate": (
            float(
                np.sum(effective_count)
                / (np.sum(effective_denominator) / n_uavs)
            )
            if np.sum(effective_denominator) else float("nan")
        ),
        "md_admission_ratio": scalar_info(final_info, "md_admission_ratio"),
        "complete_task_ratio": scalar_info(final_info, "complete_task_ratio"),
        "system_performance_true_all_GUs": scalar_info(
            final_info, "system_performance_true_all_GUs"
        ),
        "average_active_mds": scalar_info(final_info, "average_active_mds"),
    }

    uav_rows = []
    for agent_id in range(n_uavs):
        uav_rows.append({
            "checkpoint_step": int(checkpoint_step),
            "evaluation_seed": int(seed),
            "action_mode": episode_row["action_mode"],
            "trained_threshold": float(args.association_threshold),
            "evaluation_threshold": float(target_threshold),
            "threshold_scope": threshold_scope,
            "uav_id": agent_id + 1,
            "association_score_pass_rate": (
                float(pass_count[agent_id] / pass_denominator[agent_id])
                if pass_denominator[agent_id] else float("nan")
            ),
            "association_score_mean": (
                float(score_sum[agent_id] / pass_denominator[agent_id])
                if pass_denominator[agent_id] else float("nan")
            ),
            "effective_association_rate": (
                float(effective_count[agent_id] / effective_denominator[agent_id])
                if effective_denominator[agent_id] else float("nan")
            ),
            "effective_small_count": float(effective_small_count[agent_id]),
            "effective_large_count": float(effective_large_count[agent_id]),
            "effective_small_rate": (
                float(effective_small_count[agent_id] / effective_small_denominator)
                if effective_small_denominator else float("nan")
            ),
            "effective_large_rate": (
                float(effective_large_count[agent_id] / effective_large_denominator)
                if effective_large_denominator else float("nan")
            ),
            "small_position_fraction": float(small_position_fraction[agent_id]),
            "large_position_fraction": float(large_position_fraction[agent_id]),
            "other_position_fraction": float(
                1.0 - small_position_fraction[agent_id] - large_position_fraction[agent_id]
            ),
            "final_x": float(final_positions[agent_id, 0]),
            "final_y": float(final_positions[agent_id, 1]),
        })

    return episode_row, uav_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key, items in grouped.items():
        output = {key: value for key, value in zip(keys, group_key)}
        output["episodes"] = len(items)
        for metric in metrics:
            values = np.asarray([item[metric] for item in items], dtype=float)
            output[f"{metric}_mean"] = float(np.nanmean(values))
            output[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        result.append(output)
    return result


def make_plot(output_dir: Path, aggregate: list[dict], uav_aggregate: list[dict]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "trained_psi0p3": "#0072B2",
        "trained_psi0p5": "#D55E00",
        "trained_psi0p7": "#6A3D9A",
    }
    labels = {
        "trained_psi0p3": "checkpoint trained at psi=0.3",
        "trained_psi0p5": "checkpoint trained at psi=0.5",
        "trained_psi0p7": "checkpoint trained at psi=0.7",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)

    for model in RUNS:
        items = [
            row for row in aggregate
            if row["trained_model"] == model and row["action_mode"] == "deterministic"
        ]
        items.sort(key=lambda row: float(row["evaluation_threshold"]))
        x = [float(row["evaluation_threshold"]) for row in items]
        y = [float(row["system_performance_true_all_GUs_mean"]) for row in items]
        e = [float(row["system_performance_true_all_GUs_std"]) for row in items]
        axes[0].errorbar(
            x, y, yerr=e, marker="o", capsize=3,
            color=colors[model], label=labels[model], linewidth=2,
        )
    axes[0].set_title("Fixed checkpoint: system performance")
    axes[0].set_xlabel("evaluation-time environment psi")
    axes[0].set_ylabel("system_performance_true_all_GUs")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")

    # The second panel directly targets the observed UAV2 collapse.
    for model in RUNS:
        items = [
            row for row in uav_aggregate
            if row["trained_model"] == model
            and row["action_mode"] == "deterministic"
            and int(row["uav_id"]) == 2
        ]
        items.sort(key=lambda row: float(row["evaluation_threshold"]))
        x = [float(row["evaluation_threshold"]) for row in items]
        y = [float(row["small_position_fraction_mean"]) for row in items]
        e = [float(row["small_position_fraction_std"]) for row in items]
        axes[1].errorbar(
            x, y, yerr=e, marker="o", capsize=3,
            color=colors[model], label=labels[model], linewidth=2,
        )
    axes[1].set_title("UAV2 time in small region")
    axes[1].set_xlabel("evaluation-time environment psi")
    axes[1].set_ylabel("fraction in [0,200] x [0,200]")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)

    # For the p=0.3 checkpoint, show all UAVs under each counterfactual.
    selected = [
        row for row in uav_aggregate
        if row["trained_model"] == "trained_psi0p3"
        and row["action_mode"] == "deterministic"
    ]
    thresholds = sorted({float(row["evaluation_threshold"]) for row in selected})
    width = 0.22
    uav_ids = list(range(1, 6))
    for index, threshold in enumerate(thresholds):
        values = []
        errors = []
        for uav_id in uav_ids:
            match = next(
                row for row in selected
                if float(row["evaluation_threshold"]) == threshold
                and int(row["uav_id"]) == uav_id
            )
            values.append(float(match["small_position_fraction_mean"]))
            errors.append(float(match["small_position_fraction_std"]))
        positions = np.arange(len(uav_ids)) + (index - 1) * width
        axes[2].bar(
            positions, values, width=width, yerr=errors, capsize=2,
            label=f"eval psi={threshold:g}",
            color=("#0072B2", "#D55E00", "#009E73")[index],
        )
    axes[2].set_title("Trained psi=0.3: all UAVs in small region")
    axes[2].set_xlabel("UAV id")
    axes[2].set_ylabel("fraction in small region")
    axes[2].set_xticks(np.arange(len(uav_ids)))
    axes[2].set_xticklabels([str(value) for value in uav_ids])
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8, loc="best")

    fig.suptitle("Counterfactual psi evaluation with fixed checkpoints")
    fig.savefig(output_dir / "counterfactual_psi_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    if cli.episodes < 10 or cli.episodes > 30:
        raise ValueError("--episodes must be between 10 and 30")
    thresholds = [float(value) for value in cli.thresholds]
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("thresholds must be in [0, 1]")
    if cli.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {cli.output_dir}")
    cli.output_dir.mkdir(parents=True)
    (cli.output_dir / "snapshots").mkdir()

    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    episode_rows = []
    uav_rows = []
    metadata = {
        "episodes_per_configuration": cli.episodes,
        "test_seeds": seeds,
        "evaluation_thresholds": thresholds,
        "threshold_scope": cli.threshold_scope,
        "runs": {},
        "region_definition": {
            "small": "x < 200 and y < 200",
            "large": "x >= 200 and y >= 200",
        },
    }

    torch.set_num_threads(1)
    for model, run_dir in RUNS.items():
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        snapshot_dir = cli.output_dir / "snapshots" / model
        checkpoint_dir, checkpoint_files = snapshot_checkpoint(run_dir, snapshot_dir)
        checkpoint_step = checkpoint_manifest_step(checkpoint_dir)
        if checkpoint_step is None:
            raise RuntimeError(f"No verified checkpoint manifest for {run_dir}")
        args, env, actors, normers = load_policy(run_dir, checkpoint_dir)
        metadata["runs"][model] = {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": checkpoint_step,
            "trained_association_threshold": float(args.association_threshold),
            "offload_deadline_filter": bool(args.offload_deadline_filter),
            "checkpoint_files": checkpoint_files,
        }
        try:
            for target_threshold in thresholds:
                for deterministic in (True, False):
                    for seed in seeds:
                        episode_row, episode_uav_rows = evaluate_episode(
                            args,
                            env,
                            actors,
                            normers,
                            seed,
                            deterministic,
                            checkpoint_step,
                            target_threshold,
                            cli.threshold_scope,
                        )
                        episode_row["trained_model"] = model
                        for row in episode_uav_rows:
                            row["trained_model"] = model
                        episode_rows.append(episode_row)
                        uav_rows.extend(episode_uav_rows)
                    print(json.dumps({
                        "trained_model": model,
                        "evaluation_threshold": target_threshold,
                        "action_mode": "deterministic" if deterministic else "stochastic",
                        "episodes": cli.episodes,
                    }, ensure_ascii=False))
        finally:
            env.close()

    episode_metrics = cli.output_dir / "episode_metrics.csv"
    uav_metrics = cli.output_dir / "uav_metrics.csv"
    write_csv(episode_metrics, episode_rows)
    write_csv(uav_metrics, uav_rows)
    aggregate = aggregate_rows(
        episode_rows,
        ("trained_model", "action_mode", "evaluation_threshold"),
        (
            "total_reward",
            "env_association_score_pass_rate",
            "effective_md_association_rate",
            "md_admission_ratio",
            "complete_task_ratio",
            "system_performance_true_all_GUs",
            "average_active_mds",
        ),
    )
    uav_aggregate = aggregate_rows(
        uav_rows,
        ("trained_model", "action_mode", "evaluation_threshold", "uav_id"),
        (
            "association_score_pass_rate",
            "association_score_mean",
            "effective_association_rate",
            "effective_small_rate",
            "effective_large_rate",
            "small_position_fraction",
            "large_position_fraction",
            "other_position_fraction",
            "final_x",
            "final_y",
        ),
    )
    write_csv(cli.output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(cli.output_dir / "uav_aggregate_metrics.csv", uav_aggregate)
    make_plot(cli.output_dir, aggregate, uav_aggregate)
    metadata["aggregate_metrics"] = aggregate
    metadata["uav_aggregate_metrics"] = uav_aggregate
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(cli.output_dir),
        "episode_rows": len(episode_rows),
        "uav_rows": len(uav_rows),
        "plot": str(cli.output_dir / "counterfactual_psi_summary.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
