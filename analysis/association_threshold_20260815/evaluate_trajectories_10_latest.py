"""Evaluate all seven association-threshold runs on ten common episodes.

The three local runs are snapshotted once before inference.  The three remote
runs and the deadline-filter baseline are read from the frozen checkpoint
copies under ``latest_remote_checkpoints_20260815``.  Every model uses the
same deterministic test seeds (1001--1010), frozen saved normalization
statistics, and a 400-slot episode.

For every episode this writes metrics, UAV trajectories, and a compact NPZ.
For each model it also saves the three highest-performing episode trajectories
as GIFs, plus a ten-episode grid and a seven-model best-episode grid.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from onpolicy.scripts.eval.render_dynamic_mappo_episode import (  # noqa: E402
    checkpoint_manifest_step,
    draw_md_regions,
    prepare_policy_inputs,
    render_trajectory,
    snapshot_checkpoint,
)
from evaluate_latest_policy_modes import load_policy, scalar_info  # noqa: E402


RESULTS = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
CHECKPOINTS = HERE / "latest_remote_checkpoints_20260815"
REMOTE_REFRESH = CHECKPOINTS / "remote_refresh_20260815"
OUTPUT = HERE / "trajectory_eval_10_latest_20260815_v4"
SEEDS = list(range(1001, 1011))
N_UAVS = 5

RUNS = {
    "remote_psi0p3_deadline_on": {
        "label": "remote psi=0.3 | deadline ON",
        "run_dir": REMOTE_REFRESH / "psi0p3",
        "checkpoint_dir": REMOTE_REFRESH / "psi0p3" / "models" / "models",
        "snapshot": False,
    },
    "remote_psi0p7_deadline_on": {
        "label": "remote psi=0.7 | deadline ON",
        "run_dir": REMOTE_REFRESH / "psi0p7",
        "checkpoint_dir": REMOTE_REFRESH / "psi0p7" / "models",
        "snapshot": False,
    },
    "remote_psi0p9_deadline_on": {
        "label": "remote psi=0.9 | deadline ON",
        "run_dir": REMOTE_REFRESH / "psi0p9",
        "checkpoint_dir": REMOTE_REFRESH / "psi0p9" / "models",
        "snapshot": False,
    },
    "local_psi0p3_deadline_off": {
        "label": "local psi=0.3 | deadline OFF",
        "run_dir": RESULTS / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p3_nofilter_seed2_60m_20260814" / "run1",
        "snapshot": True,
    },
    "local_psi0p5_deadline_off": {
        "label": "local psi=0.5 | deadline OFF",
        "run_dir": RESULTS / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed2_60m_20260814" / "run1",
        "snapshot": True,
    },
    "local_psi0p7_deadline_off": {
        "label": "local psi=0.7 | deadline OFF",
        "run_dir": RESULTS / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed2_60m_20260814" / "run1",
        "snapshot": True,
    },
    "baseline_psi0p5_deadline_on": {
        "label": "baseline psi=0.5 | deadline ON",
        "run_dir": CHECKPOINTS / "baseline_psi0p5" / "checkpoint_snapshot",
        "checkpoint_dir": CHECKPOINTS / "baseline_psi0p5" / "checkpoint_snapshot",
        "snapshot": False,
    },
}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _region_membership(positions: np.ndarray, bounds: np.ndarray | None, row: int) -> int:
    if bounds is None or bounds.shape != (2, 4):
        return 0
    x0, x1, y0, y1 = bounds[row]
    inside = (
        (positions[:, 0] >= x0)
        & (positions[:, 0] <= x1)
        & (positions[:, 1] >= y0)
        & (positions[:, 1] <= y1)
    )
    return int(np.sum(inside))


def _pairwise_mean(positions: np.ndarray) -> float:
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distances.append(float(np.linalg.norm(positions[i] - positions[j])))
    return float(np.mean(distances)) if distances else float("nan")


def evaluate_episode(args, env, actors, normers, seed: int, checkpoint_step: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    obs, states, available_actions, _, attention_mask = env.reset()
    obs, states = prepare_policy_inputs(normers, obs, states)

    n_uavs = int(args.n_UAVs)
    n_mds = int(args.n_GUs)
    steps = int(args.episode_length)
    max_slots = int(args.max_GUs_in_range)
    rnn_states = np.zeros(
        (n_uavs, int(args.recurrent_N), int(args.hidden_size)), dtype=np.float32
    )
    masks = np.ones((n_uavs, 1), dtype=np.float32)
    threshold = float(getattr(args, "association_threshold", 0.5))

    uav_positions = np.zeros((steps + 1, n_uavs, 2), dtype=np.float64)
    gu_positions = np.zeros((steps + 1, n_mds, 2), dtype=np.float64)
    active_masks = np.zeros((steps + 1, n_mds), dtype=bool)
    rewards = np.zeros((steps, n_uavs), dtype=np.float64)
    uav_positions[0] = env.uav_positions[:, :2]
    gu_positions[0] = env.gu_positions[:, :2]
    active_masks[0] = env.active_md_mask
    region_bounds = np.asarray(getattr(env, "episode_hotspot_bounds", ()), dtype=float)
    if region_bounds.shape != (2, 4):
        region_bounds = None

    pass_count = 0
    pass_denominator = 0
    score_sum = 0.0
    effective_count = 0
    effective_denominator = 0
    final_info = {}
    total_reward = 0.0

    with torch.no_grad():
        for step in range(steps):
            valid_slots = np.asarray(
                env.nearby_gus_of_uavs[:, :max_slots] != -1, dtype=bool
            )
            raw_actions = []
            for agent_id, actor in enumerate(actors):
                kwargs = (
                    {"attention_active_mask": attention_mask[agent_id : agent_id + 1]}
                    if args.use_atten_actor
                    else {}
                )
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

            raw_actions = np.asarray(raw_actions, dtype=np.float32)
            association_scores = np.clip(raw_actions[:, 2 : 2 + max_slots], 0.0, 1.0)
            pass_count += int(np.sum((association_scores >= threshold) & valid_slots))
            pass_denominator += int(np.sum(valid_slots))
            score_sum += float(np.sum(association_scores[valid_slots]))

            obs, states, reward, dones, info, available_actions, _, attention_mask = env.step(raw_actions)
            rewards[step] = np.asarray(reward).reshape(n_uavs)
            total_reward += float(np.sum(reward))
            final_info = info
            active_ids = np.flatnonzero(env.active_md_mask)
            if active_ids.size:
                proposed = np.asarray(env.proposed_offload_actions)[:, active_ids]
                effective_count += int(np.sum(np.max(proposed, axis=0) > 1e-8))
                effective_denominator += int(active_ids.size)

            uav_positions[step + 1] = env.uav_positions[:, :2]
            gu_positions[step + 1] = env.gu_positions[:, :2]
            active_masks[step + 1] = env.active_md_mask
            obs, states = prepare_policy_inputs(normers, obs, states)
            masks[:] = 0.0 if np.all(dones) else 1.0

    final_positions = uav_positions[-1]
    row = {
        "checkpoint_step": int(checkpoint_step),
        "evaluation_seed": int(seed),
        "action_mode": "deterministic",
        "total_reward": total_reward,
        "env_association_score_pass_rate": pass_count / pass_denominator if pass_denominator else float("nan"),
        "env_association_score_mean": score_sum / pass_denominator if pass_denominator else float("nan"),
        "effective_md_association_rate": effective_count / effective_denominator if effective_denominator else float("nan"),
        "md_admission_ratio": scalar_info(final_info, "md_admission_ratio"),
        "complete_task_ratio": scalar_info(final_info, "complete_task_ratio"),
        "system_performance_true_all_GUs": scalar_info(final_info, "system_performance_true_all_GUs"),
        "average_active_mds": scalar_info(final_info, "average_active_mds"),
        "final_small_region_uavs": _region_membership(final_positions, region_bounds, 0),
        "final_large_region_uavs": _region_membership(final_positions, region_bounds, 1),
        "final_uav_pairwise_mean_distance_m": _pairwise_mean(final_positions),
        "final_uav_positions": final_positions.tolist(),
    }
    arrays = {
        "uav_positions": uav_positions,
        "gu_positions": gu_positions,
        "active_masks": active_masks,
        "rewards": rewards,
        "region_bounds": np.asarray(region_bounds if region_bounds is not None else [], dtype=np.float64),
    }
    return row, arrays, region_bounds


def write_episode_artifacts(output_dir: Path, label: str, row: dict, arrays: dict, region_bounds) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "episode_data.npz", **arrays)
    uav_positions = arrays["uav_positions"]
    render_trajectory(
        output_dir / "uav_trajectory_overview.png",
        label,
        uav_positions,
        map_size=600.0,
        region_bounds=region_bounds,
    )
    with (output_dir / "uav_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestep", "uav", "x", "y"])
        writer.writeheader()
        for timestep, positions in enumerate(uav_positions):
            for agent_id, (x, y) in enumerate(positions):
                writer.writerow({"timestep": timestep, "uav": agent_id + 1, "x": float(x), "y": float(y)})
    summary = {key: jsonable(value) for key, value in row.items()}
    summary["label"] = label
    with (output_dir / "episode_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def make_gif(episode_dir: Path, label: str, output: Path) -> None:
    data = np.load(episode_dir / "episode_data.npz")
    positions = np.asarray(data["uav_positions"], dtype=float)
    bounds = np.asarray(data["region_bounds"], dtype=float)
    if bounds.shape != (2, 4):
        bounds = None
    indices = np.unique(np.linspace(0, len(positions) - 1, min(81, len(positions))).astype(int))
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.set(xlim=(0, 600), ylim=(0, 600), xlabel="X position (m)", ylabel="Y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linestyle="--")
    draw_md_regions(ax, bounds)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    lines = [ax.plot([], [], color=colors[i], linewidth=1.8, label=f"UAV {i + 1}")[0] for i in range(N_UAVS)]
    points = [ax.plot([], [], marker="o", color=colors[i], markersize=5)[0] for i in range(N_UAVS)]
    ax.legend(loc="upper right", fontsize=7)
    title = ax.set_title("")

    def update(frame_number):
        index = int(indices[frame_number])
        for i, (line, point) in enumerate(zip(lines, points)):
            line.set_data(positions[: index + 1, i, 0], positions[: index + 1, i, 1])
            point.set_data([positions[index, i, 0]], [positions[index, i, 1]])
        title.set_text(f"{label}\nslot {index}/{len(positions) - 1}")
        return lines + points + [title]

    movie = animation.FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)
    movie.save(output, writer=PillowWriter(fps=10))
    plt.close(fig)


def make_model_grid(model_dir: Path, rows: list[dict], output: Path) -> None:
    rows = sorted(rows, key=lambda item: item["evaluation_seed"])
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.2))
    for axis, row in zip(axes.flat, rows):
        image = plt.imread(model_dir / f"episode_seed{row['evaluation_seed']}" / "uav_trajectory_overview.png")
        axis.imshow(image)
        axis.set_title(f"seed {row['evaluation_seed']}\nperf={row['system_performance_true_all_GUs']:,.0f}", fontsize=8)
        axis.axis("off")
    fig.suptitle(model_dir.name.replace("_", " "), y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_best_grid(best_rows: list[dict], output: Path, root: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2))
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, best_rows):
        image = plt.imread(root / row["model"] / f"episode_seed{row['evaluation_seed']}" / "uav_trajectory_overview.png")
        axis.imshow(image)
        axis.set_title(f"{row['label']}\nseed {row['evaluation_seed']} | perf={row['system_performance_true_all_GUs']:,.0f}", fontsize=8)
        axis.axis("off")
    fig.suptitle("Best deterministic episode trajectory per model", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists() and (OUTPUT / "metadata.json").exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "snapshots").mkdir(exist_ok=True)
    (OUTPUT / "episodes").mkdir(exist_ok=True)
    torch.set_num_threads(1)

    all_rows = []
    metadata = {
        "test_seeds": SEEDS,
        "episodes_per_model": len(SEEDS),
        "action_mode": "deterministic",
        "episode_length": 400,
        "snapshot_policy": "live local runs snapshotted once; remote copies reused as frozen checkpoints",
        "runs": {},
    }

    for model, spec in RUNS.items():
        run_dir = Path(spec["run_dir"])
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        if spec.get("snapshot"):
            snapshot_root = OUTPUT / "snapshots" / model
            existing_snapshot = snapshot_root / "checkpoint_snapshot"
            if existing_snapshot.is_dir():
                checkpoint_dir = existing_snapshot
                checkpoint_files = {
                    path.name: {"bytes": path.stat().st_size, "mtime": path.stat().st_mtime}
                    for path in sorted(checkpoint_dir.iterdir()) if path.is_file()
                }
            else:
                checkpoint_dir, checkpoint_files = snapshot_checkpoint(run_dir, snapshot_root)
        else:
            checkpoint_dir = Path(spec["checkpoint_dir"])
            checkpoint_files = {
                path.name: {"bytes": path.stat().st_size, "mtime": path.stat().st_mtime}
                for path in sorted(checkpoint_dir.iterdir()) if path.is_file()
            }
            # A downloaded remote models directory does not carry the run-level
            # args.json; the manifest verifier needs the matching copy beside
            # the checkpoint files.
            if not (checkpoint_dir / "args.json").is_file():
                shutil.copy2(run_dir / "args.json", checkpoint_dir / "args.json")
        checkpoint_step = checkpoint_manifest_step(checkpoint_dir)
        if checkpoint_step is None:
            raise RuntimeError(f"No verified checkpoint manifest for {model}: {checkpoint_dir}")

        model_dir = OUTPUT / "episodes" / model
        summary_paths = [model_dir / f"episode_seed{seed}" / "episode_summary.json" for seed in SEEDS]
        if all(path.is_file() for path in summary_paths):
            # Resume support: a previous run may have completed a model before
            # a later model's path or checkpoint failed.  Reuse the existing
            # episode artifacts instead of rerunning those ten episodes.
            model_rows = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
            for row in model_rows:
                row["model"] = model
                row["label"] = spec["label"]
            with (run_dir / "args.json").open("r", encoding="utf-8") as handle:
                args_data = Namespace(**json.load(handle))
            metadata["runs"][model] = {
                "label": spec["label"],
                "run_dir": str(run_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_step": int(checkpoint_step),
                "association_threshold": float(getattr(args_data, "association_threshold", 0.5)),
                "offload_deadline_filter": bool(getattr(args_data, "offload_deadline_filter", True)),
                "actor_message_mode": getattr(args_data, "actor_message_mode", "unknown"),
                "checkpoint_files": checkpoint_files,
                "resumed_from_existing_artifacts": True,
            }
            all_rows.extend(model_rows)
            print(json.dumps({"model": model, "status": "reused_existing_10_episode_artifacts"}, ensure_ascii=False))
            continue

        args, env, actors, normers = load_policy(run_dir, checkpoint_dir)
        model_rows = []
        metadata["runs"][model] = {
            "label": spec["label"],
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": int(checkpoint_step),
            "association_threshold": float(getattr(args, "association_threshold", 0.5)),
            "offload_deadline_filter": bool(getattr(args, "offload_deadline_filter", True)),
            "actor_message_mode": getattr(args, "actor_message_mode", "unknown"),
            "checkpoint_files": checkpoint_files,
        }
        try:
            for seed in SEEDS:
                row, arrays, region_bounds = evaluate_episode(args, env, actors, normers, seed, checkpoint_step)
                row["model"] = model
                row["label"] = spec["label"]
                episode_dir = model_dir / f"episode_seed{seed}"
                write_episode_artifacts(episode_dir, spec["label"], row, arrays, region_bounds)
                model_rows.append(row)
                all_rows.append(row)
                print(json.dumps({
                    "model": model,
                    "seed": seed,
                    "checkpoint_step": checkpoint_step,
                    "system_performance": row["system_performance_true_all_GUs"],
                    "final_small_region_uavs": row["final_small_region_uavs"],
                    "final_large_region_uavs": row["final_large_region_uavs"],
                }, ensure_ascii=False))
        finally:
            env.close()

        for rank, row in enumerate(sorted(model_rows, key=lambda item: item["system_performance_true_all_GUs"], reverse=True)[:3], start=1):
            make_gif(
                model_dir / f"episode_seed{row['evaluation_seed']}",
                spec["label"],
                model_dir / f"best{rank}_seed{row['evaluation_seed']}.gif",
            )
        make_model_grid(model_dir, model_rows, model_dir / "trajectory_grid_10_episodes.png")

    metric_names = [
        "total_reward",
        "env_association_score_pass_rate",
        "env_association_score_mean",
        "effective_md_association_rate",
        "md_admission_ratio",
        "complete_task_ratio",
        "system_performance_true_all_GUs",
        "average_active_mds",
        "final_small_region_uavs",
        "final_large_region_uavs",
        "final_uav_pairwise_mean_distance_m",
    ]
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["model"]].append(row)
    aggregate = []
    best_rows = []
    for model, rows in grouped.items():
        best_rows.append(max(rows, key=lambda item: item["system_performance_true_all_GUs"]))
        output = {
            "model": model,
            "label": rows[0]["label"],
            "episodes": len(rows),
            "checkpoint_step": rows[0]["checkpoint_step"],
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            output[f"{metric}_mean"] = float(np.nanmean(values))
            output[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        aggregate.append(output)

    with (OUTPUT / "episode_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(all_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUTPUT / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    make_best_grid(best_rows, OUTPUT / "trajectory_grid_best_of_7.png", OUTPUT / "episodes")
    metadata["aggregate_metrics"] = aggregate
    metadata["best_episode_per_model"] = [
        {key: jsonable(value) for key, value in row.items() if key not in {"final_uav_positions"}}
        for row in best_rows
    ]
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
