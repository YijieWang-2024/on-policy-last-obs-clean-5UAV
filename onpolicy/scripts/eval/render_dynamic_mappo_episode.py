#!/usr/bin/env python
"""Render one deterministic dynamic-MD episode from a separated MAPPO checkpoint."""

import argparse
import csv
import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_attention import R_Actor_Attention
from onpolicy.envs.mec.env_maker import WrappedMECEnv
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.utils.checkpoint_manifest import (
    expected_checkpoint_names,
    read_checkpoint_manifest,
    sha256,
)


COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument("--training-step", type=int)
    parser.add_argument(
        "--confirm-checkpoint-frozen",
        action="store_true",
        help=(
            "required only for a legacy checkpoint without a manifest; confirms "
            "that no trainer is writing run-dir/models"
        ),
    )
    parser.add_argument(
        "--mask-actor-messages",
        action="store_true",
        help=(
            "Counterfactually replace every radius-gated actor message block, "
            "including presence masks, with zeros before policy inference."
        ),
    )
    return parser.parse_args()


def snapshot_checkpoint(run_dir, output_dir, allow_frozen_legacy=False):
    source = run_dir / "models"
    required = [source / name for name in expected_checkpoint_names(5)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files: " + ", ".join(missing))

    destination = output_dir / "checkpoint_snapshot"
    temporary = output_dir / "checkpoint_snapshot.tmp"
    for _ in range(3):
        manifest_before = read_checkpoint_manifest(
            source, 5, args_path=run_dir / "args.json"
        )
        if manifest_before is None and not allow_frozen_legacy:
            raise ValueError(
                "legacy checkpoint has no manifest; stop checkpoint writes and "
                "explicitly confirm that the directory is frozen"
            )
        source_files = sorted(path for path in source.iterdir() if path.is_file())
        before = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_files}
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        for path in source_files:
            shutil.copy2(path, temporary / path.name)
        after = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_files}
        manifest_after = read_checkpoint_manifest(
            source, 5, args_path=run_dir / "args.json"
        )
        if before == after and manifest_before == manifest_after:
            shutil.copy2(run_dir / "args.json", temporary / "args.json")
            temporary.replace(destination)
            return destination, {
                path.name: {
                    "bytes": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                    "sha256": sha256(path),
                }
                for path in sorted(destination.iterdir())
                if path.is_file()
            }
    raise RuntimeError("Checkpoint changed during all three snapshot attempts")


def checkpoint_manifest_step(checkpoint_dir):
    manifest = read_checkpoint_manifest(
        checkpoint_dir, 5, args_path=checkpoint_dir / "args.json"
    )
    return None if manifest is None else int(manifest["total_num_steps"])


def frozen_normalize(normers, obs, states):
    obs = np.asarray(obs, dtype=np.float32).copy()
    states = np.asarray(states, dtype=np.float32).copy()
    for agent_id, normer in enumerate(normers):
        if normer.ob_rms is None:
            continue
        start = normer.not_norm
        preserved_obs = [
            (slice_start, slice_end, obs[agent_id, slice_start:slice_end].copy())
            for slice_start, slice_end in getattr(
                normer, "obs_preserve_slices", ()
            )
        ]
        obs[agent_id, start:] = np.clip(
            (obs[agent_id, start:] - normer.ob_rms.mean[start:])
            / np.sqrt(normer.ob_rms.var[start:] + normer.epsilon),
            -normer.clipob,
            normer.clipob,
        )
        states[agent_id, start:] = np.clip(
            (states[agent_id, start:] - normer.state_rms.mean[start:])
            / np.sqrt(normer.state_rms.var[start:] + normer.epsilon),
            -normer.clipob,
            normer.clipob,
        )
        for slice_start, slice_end, raw_values in preserved_obs:
            obs[agent_id, slice_start:slice_end] = raw_values
    return obs, states


def mask_actor_message_observations(normers, obs):
    """Return a copy with every actor-only communication block zeroed.

    The slice layout is reconstructed by each saved Normer from args, so this
    does not hard-code timestep/agent-ID offsets. Refuse a silent no-op when a
    checkpoint was trained without a message-capable observation contract.
    """
    obs = np.asarray(obs, dtype=np.float32).copy()
    layouts = [
        tuple(getattr(normer, "obs_preserve_slices", ()))
        for normer in normers
    ]
    if not layouts or any(layout != layouts[0] for layout in layouts):
        raise ValueError("actor message slice layouts differ across normers")
    if not layouts[0]:
        raise ValueError(
            "--mask-actor-messages requires a message-capable checkpoint"
        )
    for agent_id, layout in enumerate(layouts):
        for slice_start, slice_end in layout:
            obs[agent_id, slice_start:slice_end] = 0.0
    return obs


def prepare_policy_inputs(normers, obs, states, mask_actor_messages=False):
    if mask_actor_messages:
        obs = mask_actor_message_observations(normers, obs)
    return frozen_normalize(normers, obs, states)


def serving_uavs(service_matrix, threshold=1e-8):
    maxima = np.max(service_matrix, axis=0)
    result = np.full(service_matrix.shape[1], -1, dtype=np.int16)
    served = maxima > threshold
    result[served] = np.argmax(service_matrix[:, served], axis=0)
    return result


def draw_md_regions(ax):
    ax.add_patch(Rectangle((0, 0), 175, 175, fill=False, edgecolor="0.45", linestyle="--", linewidth=1.2))
    ax.add_patch(Rectangle((200, 200), 400, 400, fill=False, edgecolor="0.45", linestyle="--", linewidth=1.2))


def render_frame(path, method_label, slot, cover_radius, uav_positions, gu_positions, active_mask,
                 service_matrix, uav_trail):
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set(xlim=(0, 600), ylim=(0, 600), xlabel="X position (m)", ylabel="Y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28, linestyle="--")
    draw_md_regions(ax)

    for i, (x, y) in enumerate(uav_positions):
        trail = uav_trail[:, i]
        ax.plot(trail[:, 0], trail[:, 1], color=COLORS[i], linewidth=1.2, alpha=0.65)
        ax.add_patch(Circle((x, y), cover_radius, color=COLORS[i], alpha=0.10))
        ax.scatter(x, y, marker="X", color=COLORS[i], s=180, zorder=5)
        ax.text(x + 5, y + 5, f"UAV{i + 1}", color=COLORS[i], fontsize=10, weight="bold")

    assignments = serving_uavs(service_matrix)
    active_ids = np.flatnonzero(active_mask)
    for md_id in active_ids:
        x, y = gu_positions[md_id]
        agent_id = assignments[md_id]
        color = "black" if agent_id < 0 else COLORS[agent_id]
        ax.scatter(x, y, marker="o", color=color, s=34, zorder=4)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[i], markersize=7,
                   label=f"MDs served by UAV {i + 1}")
        for i in range(5)
    ]
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=7,
                              label="MDs computing locally"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.88, fontsize=8)
    served_count = int(np.sum(assignments[active_ids] >= 0))
    ax.text(8, 590, f"Active MDs: {len(active_ids)}   Served: {served_count}   Local: {len(active_ids) - served_count}",
            va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    ax.set_title(f"{method_label} Test Episode - Slot {slot} Decision")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_trajectory(path, method_label, uav_positions):
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set(xlim=(0, 600), ylim=(0, 600), xlabel="X position (m)", ylabel="Y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28, linestyle="--")
    draw_md_regions(ax)
    for i in range(uav_positions.shape[1]):
        path_i = uav_positions[:, i]
        ax.plot(path_i[:, 0], path_i[:, 1], color=COLORS[i], linewidth=1.8, label=f"UAV {i + 1}")
        ax.scatter(*path_i[0], marker="o", color=COLORS[i], s=35)
        ax.scatter(*path_i[-1], marker="X", color=COLORS[i], s=150)
    ax.legend(loc="upper right")
    ax.set_title(f"{method_label} Test Episode - UAV Trajectories (400 Slots)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    cli = parse_cli()
    run_dir = cli.run_dir.resolve()
    output_dir = cli.output_dir.resolve()
    method_label = "DC-PPO" if "dcppo" in str(run_dir).lower() else "MAPPO"
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()

    checkpoint_dir, checkpoint_files = snapshot_checkpoint(
        run_dir,
        output_dir,
        allow_frozen_legacy=cli.confirm_checkpoint_frozen,
    )
    manifest_step = checkpoint_manifest_step(checkpoint_dir)
    if (
        cli.training_step is not None
        and manifest_step is not None
        and int(cli.training_step) != manifest_step
    ):
        raise ValueError(
            f"declared training step {cli.training_step} does not match "
            f"checkpoint manifest step {manifest_step}"
        )
    with (run_dir / "args.json").open("r", encoding="utf-8") as handle:
        args = Namespace(**json.load(handle))
    args.n_rollout_threads = 1
    args.n_training_threads = 1
    args.model_dir = str(checkpoint_dir)

    seed = cli.seed if cli.seed is not None else int(args.seed) * 50000
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = WrappedMECEnv(args=args)
    env.seed(seed)
    device = torch.device("cpu")
    actor_class = R_Actor_Attention if args.use_atten_actor else R_Actor
    actors = []
    normers = []
    for agent_id in range(args.n_UAVs):
        actor = actor_class(args, env.observation_space[agent_id], env.action_space[agent_id], device)
        try:
            state_dict = torch.load(checkpoint_dir / f"actor_agent{agent_id}.pt", map_location=device,
                                    weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_dir / f"actor_agent{agent_id}.pt", map_location=device)
        actor.load_state_dict(state_dict)
        actor.eval()
        actors.append(actor)

        normer = Normer(args=args, obs_space=env.observation_space[agent_id],
                        states_space=env.share_observation_space[agent_id])
        normer.load(checkpoint_dir / f"normer{agent_id}.pkl")
        normers.append(normer)

    obs, states, available_actions, _, attention_mask = env.reset()
    obs, states = prepare_policy_inputs(
        normers,
        obs,
        states,
        mask_actor_messages=cli.mask_actor_messages,
    )
    steps = int(args.episode_length)
    n_uavs, n_mds = int(args.n_UAVs), int(args.n_GUs)
    rnn_states = np.zeros((n_uavs, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((n_uavs, 1), dtype=np.float32)

    uav_positions = np.zeros((steps + 1, n_uavs, 2), dtype=np.float64)
    gu_positions = np.zeros((steps + 1, n_mds, 2), dtype=np.float64)
    active_masks = np.zeros((steps + 1, n_mds), dtype=bool)
    session_ids = np.full((steps + 1, n_mds), -1, dtype=np.int64)
    remaining_lifetimes = np.zeros((steps + 1, n_mds), dtype=np.int32)
    service_actions = np.zeros((steps, n_uavs, n_mds), dtype=np.float64)
    rewards = np.zeros((steps, n_uavs), dtype=np.float64)
    task_completed = np.zeros((steps, n_mds), dtype=bool)

    uav_positions[0] = env.uav_positions[:, :2]
    gu_positions[0] = env.gu_positions[:, :2]
    active_masks[0] = env.active_md_mask
    session_ids[0] = env.md_session_ids
    remaining_lifetimes[0] = env.md_remaining_lifetime
    service_rows = []
    final_info = {}

    with torch.no_grad():
        for step in range(steps):
            slot = step + 1
            decision_uavs = env.uav_positions[:, :2].copy()
            decision_mds = env.gu_positions[:, :2].copy()
            decision_active = env.active_md_mask.copy()
            decision_sessions = env.md_session_ids.copy()

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

            obs, states, reward, dones, info, available_actions, _, attention_mask = env.step(
                np.asarray(raw_actions)
            )
            service = env.proposed_offload_actions.copy()
            assignments = serving_uavs(service)
            service_actions[step] = service
            rewards[step] = np.asarray(reward).reshape(n_uavs)
            task_completed[step] = env.complete_task.astype(bool)
            final_info = info

            coverage = np.linalg.norm(
                decision_uavs[:, None, :] - decision_mds[None, :, :], axis=2
            ) <= args.Cover_R
            for md_id in np.flatnonzero(decision_active):
                agent_id = int(assignments[md_id])
                service_rows.append({
                    "slot": slot,
                    "md_slot": int(md_id),
                    "session_id": int(decision_sessions[md_id]),
                    "x": float(decision_mds[md_id, 0]),
                    "y": float(decision_mds[md_id, 1]),
                    "covered": bool(np.any(coverage[:, md_id])),
                    "served_by": "local" if agent_id < 0 else f"UAV{agent_id + 1}",
                    "service_weight": 0.0 if agent_id < 0 else float(service[agent_id, md_id]),
                    "task_completed": bool(task_completed[step, md_id]),
                })

            uav_positions[slot] = env.uav_positions[:, :2]
            gu_positions[slot] = env.gu_positions[:, :2]
            active_masks[slot] = env.active_md_mask
            session_ids[slot] = env.md_session_ids
            remaining_lifetimes[slot] = env.md_remaining_lifetime

            if slot % cli.frame_interval == 0:
                render_frame(
                    frames_dir / f"slot_{slot:03d}.png",
                    method_label,
                    slot,
                    args.Cover_R,
                    decision_uavs,
                    decision_mds,
                    decision_active,
                    service,
                    uav_positions[:slot],
                )

            obs, states = prepare_policy_inputs(
                normers,
                obs,
                states,
                mask_actor_messages=cli.mask_actor_messages,
            )
            masks[:] = 0.0 if np.all(dones) else 1.0

    env.close()
    render_trajectory(output_dir / "uav_trajectory_overview.png", method_label, uav_positions)

    with (output_dir / "uav_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestep", "uav", "x", "y"])
        writer.writeheader()
        for timestep, positions in enumerate(uav_positions):
            for agent_id, (x, y) in enumerate(positions):
                writer.writerow({"timestep": timestep, "uav": agent_id + 1, "x": x, "y": y})

    with (output_dir / "service_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "slot", "md_slot", "session_id", "x", "y", "covered",
            "served_by", "service_weight", "task_completed",
        ])
        writer.writeheader()
        writer.writerows(service_rows)

    np.savez_compressed(
        output_dir / "episode_data.npz",
        uav_positions=uav_positions,
        gu_positions=gu_positions,
        active_masks=active_masks,
        md_session_ids=session_ids,
        md_remaining_lifetimes=remaining_lifetimes,
        service_actions=service_actions,
        rewards=rewards,
        task_completed=task_completed,
    )

    summary = {
        "status": "completed",
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_files": checkpoint_files,
        "training_step_at_snapshot": (
            manifest_step if manifest_step is not None else cli.training_step
        ),
        "checkpoint_manifest_status": (
            "verified" if manifest_step is not None else "legacy_unavailable"
        ),
        "evaluation_seed": seed,
        "actor_mode": "deterministic",
        "actor_message_evaluation": (
            "masked_to_zero" if cli.mask_actor_messages else "as_observed"
        ),
        "normalization": "frozen saved statistics",
        "episode_length": steps,
        "frame_interval": cli.frame_interval,
        "frame_count": len(list(frames_dir.glob("*.png"))),
        "total_reward_per_uav": rewards.sum(axis=0).tolist(),
        "final_environment_info": {key: jsonable(value) for key, value in final_info.items()},
    }
    with (output_dir / "episode_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: summary[key] for key in (
        "status", "training_step_at_snapshot", "evaluation_seed", "frame_count", "total_reward_per_uav"
    )}, ensure_ascii=False, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
