"""Diagnose why UAV 5 changes its deployment under different psi thresholds.

This is a read-only diagnostic over frozen checkpoints.  It uses the same
deterministic evaluation seeds as the three-episode trajectory evaluation and
records both the actor's raw association scores and the environment's final
post-threshold association decisions.  The movement action is independent of
the association slice in the raw action, so the comparison helps distinguish
an action-layout bug from a learned reward/credit effect.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "association_threshold_20260815" / "evaluate_trajectories_10_latest.py"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BASE_SCRIPT.parent))

spec = importlib.util.spec_from_file_location("association_eval_diagnostic_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import evaluator: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


RESULTS = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
EVAL_ROOT = HERE / "trajectory_eval_3_latest_20260816"
REMOTE_ROOT = HERE / "remote_snapshots"
OUTPUT = HERE / "uav5_diagnostic_20260816"
SEEDS = [1001, 1002, 1003]

RUNS = {
    "remote_psi0p1_seed42": {
        "label": "remote psi=0.1 | seed=42",
        "run_dir": REMOTE_ROOT / "remote_psi0p1" / "checkpoint_snapshot",
        "checkpoint_dir": REMOTE_ROOT / "remote_psi0p1" / "checkpoint_snapshot" / "models",
    },
    "remote_psi0p3_seed42": {
        "label": "remote psi=0.3 | seed=42",
        "run_dir": REMOTE_ROOT / "remote_psi0p3" / "checkpoint_snapshot",
        "checkpoint_dir": REMOTE_ROOT / "remote_psi0p3" / "checkpoint_snapshot" / "models",
    },
    "remote_psi0p5_seed42": {
        "label": "remote psi=0.5 | seed=42",
        "run_dir": REMOTE_ROOT / "remote_psi0p5" / "checkpoint_snapshot",
        "checkpoint_dir": REMOTE_ROOT / "remote_psi0p5" / "checkpoint_snapshot" / "models",
    },
    "local_psi0p7_seed42": {
        "label": "local psi=0.7 | seed=42",
        "run_dir": EVAL_ROOT / "snapshots" / "local_psi0p7_seed42_deadline_off" / "checkpoint_snapshot",
        "checkpoint_dir": EVAL_ROOT / "snapshots" / "local_psi0p7_seed42_deadline_off" / "checkpoint_snapshot",
    },
    "local_psi0p9_seed42": {
        "label": "local psi=0.9 | seed=42",
        "run_dir": EVAL_ROOT / "snapshots" / "local_psi0p9_seed42_deadline_off" / "checkpoint_snapshot",
        "checkpoint_dir": EVAL_ROOT / "snapshots" / "local_psi0p9_seed42_deadline_off" / "checkpoint_snapshot",
    },
    "local_psi0p1_seed2": {
        "label": "local psi=0.1 | seed=2",
        "run_dir": EVAL_ROOT / "snapshots" / "local_psi0p1_seed2_deadline_off" / "checkpoint_snapshot",
        "checkpoint_dir": EVAL_ROOT / "snapshots" / "local_psi0p1_seed2_deadline_off" / "checkpoint_snapshot",
    },
}


def _inside(xy: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    return (
        (xy[:, 0] >= x0)
        & (xy[:, 0] <= x1)
        & (xy[:, 1] >= y0)
        & (xy[:, 1] <= y1)
    )


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate_one(args: Namespace, env, actors, normers, seed: int, checkpoint_step: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    obs, states, available_actions, _, attention_mask = env.reset()
    obs, states = base.prepare_policy_inputs(normers, obs, states)

    n_uavs = int(args.n_UAVs)
    n_mds = int(args.n_GUs)
    steps = int(args.episode_length)
    max_slots = int(args.max_GUs_in_range)
    threshold = float(getattr(args, "association_threshold", 0.5))
    rnn_states = np.zeros(
        (n_uavs, int(args.recurrent_N), int(args.hidden_size)), dtype=np.float32
    )
    masks = np.ones((n_uavs, 1), dtype=np.float32)

    positions = np.zeros((steps + 1, n_uavs, 2), dtype=np.float64)
    positions[0] = env.uav_positions[:, :2]
    raw_score_pass = np.zeros(n_uavs, dtype=np.float64)
    raw_score_denominator = np.zeros(n_uavs, dtype=np.float64)
    raw_score_sum = np.zeros(n_uavs, dtype=np.float64)
    raw_score_count = np.zeros(n_uavs, dtype=np.float64)
    flight_sum = np.zeros((n_uavs, 2), dtype=np.float64)
    flight_norm_sum = np.zeros(n_uavs, dtype=np.float64)
    flight_toward_small = np.zeros(n_uavs, dtype=np.float64)
    effective_selected = np.zeros(n_uavs, dtype=np.float64)
    effective_small = np.zeros(n_uavs, dtype=np.float64)
    effective_large = np.zeros(n_uavs, dtype=np.float64)
    effective_steps = np.zeros(n_uavs, dtype=np.float64)
    selected_small_gu_ids: list[list[int]] = []
    selected_large_gu_ids: list[list[int]] = []
    uav5_first_steps = []

    small_bounds = (0.0, 200.0, 0.0, 200.0)
    large_bounds = (200.0, 600.0, 200.0, 600.0)

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
            flight = raw_actions[:, :2]
            scores = np.clip(raw_actions[:, 2 : 2 + max_slots], 0.0, 1.0)
            valid_scores = scores[valid_slots]
            for uav in range(n_uavs):
                vals = scores[uav][valid_slots[uav]]
                raw_score_denominator[uav] += vals.size
                raw_score_sum[uav] += float(vals.sum()) if vals.size else 0.0
                raw_score_count[uav] += vals.size
                raw_score_pass[uav] += float(np.sum(vals >= threshold)) if vals.size else 0.0
                flight_sum[uav] += flight[uav]
                flight_norm_sum[uav] += float(np.linalg.norm(flight[uav]))
                before = env.uav_positions[uav, :2].copy()
                small_center = np.array([100.0, 100.0])
                before_d = float(np.linalg.norm(before - small_center))
                # The actual position delta is recorded after env.step below.
                uav5_first_steps.append({
                    "timestep": step,
                    "uav": uav + 1,
                    "x": float(before[0]),
                    "y": float(before[1]),
                    "flight_x": float(flight[uav, 0]),
                    "flight_y": float(flight[uav, 1]),
                    "flight_norm": float(np.linalg.norm(flight[uav])),
                    "valid_slots": int(valid_slots[uav].sum()),
                    "raw_pass": int(np.sum(scores[uav][valid_slots[uav]] >= threshold)),
                    "raw_score_mean": float(np.mean(vals)) if vals.size else float("nan"),
                    "distance_to_small_before": before_d,
                })

            gu_positions_before = np.asarray(env.gu_positions[:, :2], dtype=float).copy()
            active_before = np.asarray(env.active_md_mask, dtype=bool).copy()
            obs, states, reward, dones, info, available_actions, _, attention_mask = env.step(raw_actions)

            actual_delta = env.uav_positions[:, :2] - positions[step]
            for uav in range(n_uavs):
                after = env.uav_positions[uav, :2]
                flight_toward_small[uav] += float(
                    np.linalg.norm(positions[step, uav] - np.array([100.0, 100.0]))
                    - np.linalg.norm(after - np.array([100.0, 100.0]))
                )

            proposed = np.asarray(env.proposed_offload_actions, dtype=float)
            selected = proposed > 1e-8
            selected_ids = np.flatnonzero(np.any(selected, axis=0) & active_before)
            small_ids = selected_ids[_inside(gu_positions_before[selected_ids], small_bounds)] if selected_ids.size else np.array([], dtype=int)
            large_ids = selected_ids[_inside(gu_positions_before[selected_ids], large_bounds)] if selected_ids.size else np.array([], dtype=int)
            selected_small_gu_ids.append(small_ids.tolist())
            selected_large_gu_ids.append(large_ids.tolist())
            for uav in range(n_uavs):
                ids = np.flatnonzero(selected[uav] & active_before)
                effective_selected[uav] += ids.size
                effective_steps[uav] += 1.0
                effective_small[uav] += int(np.sum(_inside(gu_positions_before[ids], small_bounds))) if ids.size else 0
                effective_large[uav] += int(np.sum(_inside(gu_positions_before[ids], large_bounds))) if ids.size else 0

            positions[step + 1] = env.uav_positions[:, :2]
            obs, states = base.prepare_policy_inputs(normers, obs, states)
            masks[:] = 0.0 if np.all(dones) else 1.0

    uav5 = positions[:, 4]
    path = float(np.linalg.norm(np.diff(uav5, axis=0), axis=1).sum())
    return {
        "evaluation_seed": seed,
        "checkpoint_step": checkpoint_step,
        "association_threshold": threshold,
        "uav5_start": positions[0, 4].tolist(),
        "uav5_end": positions[-1, 4].tolist(),
        "uav5_path_length_m": path,
        "uav5_min_distance_to_small_center_m": float(np.linalg.norm(uav5 - np.array([100.0, 100.0]), axis=1).min()),
        "uav5_small_time_fraction": float(np.mean(_inside(uav5, small_bounds))),
        "raw_association_pass_rate_by_uav": (raw_score_pass / np.maximum(raw_score_denominator, 1.0)).tolist(),
        "raw_association_score_mean_by_uav": (raw_score_sum / np.maximum(raw_score_count, 1.0)).tolist(),
        "raw_flight_mean_by_uav": (flight_sum / steps).tolist(),
        "raw_flight_norm_mean_by_uav": (flight_norm_sum / steps).tolist(),
        "movement_toward_small_center_m_by_uav": flight_toward_small.tolist(),
        "effective_selected_gus_by_uav": effective_selected.tolist(),
        "effective_small_selected_gus_by_uav": effective_small.tolist(),
        "effective_large_selected_gus_by_uav": effective_large.tolist(),
        "effective_selected_per_step_by_uav": (effective_selected / np.maximum(effective_steps, 1.0)).tolist(),
        "uav5_first_25_steps": [item for item in uav5_first_steps if item["uav"] == 5][:25],
    }


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metadata = {"seeds": SEEDS, "runs": {}}
    for name, run in RUNS.items():
        run_dir = Path(run["run_dir"])
        checkpoint_dir = Path(run["checkpoint_dir"])
        args, env, actors, normers = base.load_policy(run_dir, checkpoint_dir)
        checkpoint_step = base.checkpoint_manifest_step(checkpoint_dir)
        metadata["runs"][name] = {
            "label": run["label"],
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": checkpoint_step,
            "association_threshold": float(getattr(args, "association_threshold", 0.5)),
            "offload_deadline_filter": bool(getattr(args, "offload_deadline_filter", True)),
            "cartesian_flight": bool(getattr(args, "cartesian_flight", False)),
            "continuous_associate": bool(getattr(args, "continuous_associate", False)),
        }
        try:
            for seed in SEEDS:
                row = evaluate_one(args, env, actors, normers, seed, checkpoint_step)
                row["model"] = name
                row["label"] = run["label"]
                all_rows.append(row)
                print(json.dumps({
                    "model": name,
                    "seed": seed,
                    "uav5_end": row["uav5_end"],
                    "uav5_small_time_fraction": row["uav5_small_time_fraction"],
                    "raw_pass_rate": row["raw_association_pass_rate_by_uav"][4],
                    "effective_selected_uav5": row["effective_selected_gus_by_uav"][4],
                }, ensure_ascii=False))
        finally:
            env.close()

    fields = [
        "model", "label", "evaluation_seed", "checkpoint_step", "association_threshold",
        "uav5_start", "uav5_end", "uav5_path_length_m", "uav5_min_distance_to_small_center_m",
        "uav5_small_time_fraction", "raw_association_pass_rate_by_uav",
        "raw_association_score_mean_by_uav", "raw_flight_mean_by_uav",
        "raw_flight_norm_mean_by_uav", "movement_toward_small_center_m_by_uav",
        "effective_selected_gus_by_uav", "effective_small_selected_gus_by_uav",
        "effective_large_selected_gus_by_uav", "effective_selected_per_step_by_uav",
        "uav5_first_25_steps",
    ]
    with (OUTPUT / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "rows": all_rows}, handle, ensure_ascii=False, indent=2)
    with (OUTPUT / "diagnostic_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict)) else row[key] for key in fields})

    print(json.dumps({"output": str(OUTPUT), "rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
