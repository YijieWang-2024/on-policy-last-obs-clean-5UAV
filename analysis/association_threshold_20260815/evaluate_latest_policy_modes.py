"""Evaluate the latest coherent checkpoints with deterministic and sampled actions.

This is a metric-only companion to render_dynamic_mappo_episode.py.  It snapshots
each live run once, then evaluates the same checkpoint with the same test seeds in
both action modes.  In addition to environment metrics, it records:

* env_association_score_pass_rate: clipped association scores >= psi on valid
  local-MD action slots, before environment post-processing;
* effective_md_association_rate: fraction of active MDs with a nonzero proposed
  offload after environment post-processing.
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
    "local_psi0p5_deadline_off": RESULTS / (
        "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
        "vmax30_psi0p5_nofilter_seed2_60m_20260814"
    ) / "run1",
    "local_psi0p7_deadline_off": RESULTS / (
        "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
        "vmax30_psi0p7_nofilter_seed2_60m_20260814"
    ) / "run1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="number of common test seeds per model and action mode",
    )
    parser.add_argument("--seed-start", type=int, default=2001)
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


def evaluate_episode(args, env, actors, normers, seed: int, deterministic: bool, checkpoint_step: int):
    # The evaluation seed is applied independently for every model and mode.
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    obs, states, available_actions, _, attention_mask = env.reset()
    obs, states = prepare_policy_inputs(normers, obs, states)

    n_uavs = int(args.n_UAVs)
    steps = int(args.episode_length)
    max_slots = int(args.max_GUs_in_range)
    rnn_states = np.zeros(
        (n_uavs, int(args.recurrent_N), int(args.hidden_size)), dtype=np.float32
    )
    masks = np.ones((n_uavs, 1), dtype=np.float32)
    threshold = float(getattr(args, "association_threshold", 0.5))

    pass_count = 0
    pass_denominator = 0
    score_sum = 0.0
    effective_association_count = 0
    effective_association_denominator = 0
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
            pass_count += int(np.sum((association_scores >= threshold) & valid_slots))
            pass_denominator += int(np.sum(valid_slots))
            score_sum += float(np.sum(association_scores[valid_slots]))

            obs, states, reward, dones, info, available_actions, _, attention_mask = env.step(
                raw_actions
            )
            total_reward += float(np.sum(reward))
            final_info = info
            active_ids = np.flatnonzero(env.active_md_mask)
            if active_ids.size:
                proposed = np.asarray(env.proposed_offload_actions)[:, active_ids]
                effective_association_count += int(
                    np.sum(np.max(proposed, axis=0) > 1e-8)
                )
                effective_association_denominator += int(active_ids.size)

            obs, states = prepare_policy_inputs(normers, obs, states)
            masks[:] = 0.0 if np.all(dones) else 1.0

    row = {
        "checkpoint_step": int(checkpoint_step),
        "evaluation_seed": int(seed),
        "action_mode": "deterministic" if deterministic else "stochastic",
        "total_reward": total_reward,
        "env_association_score_pass_rate": (
            pass_count / pass_denominator if pass_denominator else float("nan")
        ),
        "env_association_score_mean": (
            score_sum / pass_denominator if pass_denominator else float("nan")
        ),
        "effective_md_association_rate": (
            effective_association_count / effective_association_denominator
            if effective_association_denominator
            else float("nan")
        ),
        "md_admission_ratio": scalar_info(final_info, "md_admission_ratio"),
        "complete_task_ratio": scalar_info(final_info, "complete_task_ratio"),
        "system_performance_true_all_GUs": scalar_info(
            final_info, "system_performance_true_all_GUs"
        ),
        "average_active_mds": scalar_info(final_info, "average_active_mds"),
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["action_mode"])].append(row)
    result = []
    metric_names = [
        "total_reward",
        "env_association_score_pass_rate",
        "env_association_score_mean",
        "effective_md_association_rate",
        "md_admission_ratio",
        "complete_task_ratio",
        "system_performance_true_all_GUs",
        "average_active_mds",
    ]
    for (model, action_mode), items in grouped.items():
        output = {
            "model": model,
            "action_mode": action_mode,
            "episodes": len(items),
            "checkpoint_step": items[0]["checkpoint_step"],
        }
        for metric in metric_names:
            values = np.asarray([item[metric] for item in items], dtype=float)
            output[f"{metric}_mean"] = float(np.nanmean(values))
            output[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
            output[f"{metric}_se"] = float(
                np.nanstd(values, ddof=1) / np.sqrt(len(values))
            )
        result.append(output)
    return result


def main() -> None:
    cli = parse_args()
    if cli.episodes < 20 or cli.episodes > 30:
        raise ValueError("--episodes must be between 20 and 30")
    if cli.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {cli.output_dir}")
    cli.output_dir.mkdir(parents=True)
    (cli.output_dir / "snapshots").mkdir()

    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    all_rows = []
    metadata = {
        "episodes_per_model_and_mode": cli.episodes,
        "test_seeds": seeds,
        "modes": ["deterministic", "stochastic"],
        "runs": {},
        "metric_definitions": {
            "env_association_score_pass_rate": "clipped raw association score >= psi over valid local-MD action slots",
            "effective_md_association_rate": "active MDs with nonzero proposed offload after environment post-processing",
        },
    }

    torch.set_num_threads(1)
    for model, run_dir in RUNS.items():
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        snapshot_dir = cli.output_dir / "snapshots" / model
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir, checkpoint_files = snapshot_checkpoint(run_dir, snapshot_dir)
        checkpoint_step = checkpoint_manifest_step(checkpoint_dir)
        if checkpoint_step is None:
            raise RuntimeError(f"No verified checkpoint manifest for {run_dir}")
        args, env, actors, normers = load_policy(run_dir, checkpoint_dir)
        metadata["runs"][model] = {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": checkpoint_step,
            "association_threshold": float(args.association_threshold),
            "offload_deadline_filter": bool(args.offload_deadline_filter),
            "checkpoint_files": checkpoint_files,
        }

        try:
            for deterministic in (True, False):
                for seed in seeds:
                    row = evaluate_episode(
                        args,
                        env,
                        actors,
                        normers,
                        seed,
                        deterministic,
                        checkpoint_step,
                    )
                    row["model"] = model
                    all_rows.append(row)
                    print(json.dumps({
                        "model": model,
                        "mode": row["action_mode"],
                        "seed": seed,
                        "system_performance": row["system_performance_true_all_GUs"],
                    }, ensure_ascii=False))
        finally:
            env.close()

    write_csv(cli.output_dir / "episode_metrics.csv", all_rows)
    aggregate = summarize(all_rows)
    write_csv(cli.output_dir / "aggregate_metrics.csv", aggregate)
    metadata["aggregate"] = aggregate
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(cli.output_dir), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
