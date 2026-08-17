"""25-seed UAV-wise comparison for homogeneous, heterogeneous, and hybrid policies.

The hybrid diagnostic deliberately uses the homogeneous checkpoint for the
flight branch and the heterogeneous checkpoint for the resource/association
branches.  It is evaluation-only; it does not alter the training code.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE_SCRIPT = PROJECT / "analysis" / "psi_reviewer_final_metrics_20260816" / "evaluate_final_checkpoints.py"
HELPER_SCRIPT = PROJECT / "analysis" / "psi0p5_per_uav_30det_20260817" / "evaluate_psi0p5_per_uav.py"
HYBRID_SCRIPT = PROJECT / "analysis" / "psi_hybrid_checkpoint_20260817" / "evaluate_hybrid.py"
RESULTS = PROJECT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
DEFAULT_LOCAL_RUN = RESULTS / "dcppoR520_fixed600_200_het1_13_07_13_07_psi05_nf_cur_s2_60m_20260817" / "run2"
DEFAULT_REMOTE_ROOT = HERE / "remote_final_snapshot_59980800"
DEFAULT_OUTPUT = HERE / "eval25_per_uav_20260817_current"
PER_UAV_FIELDS = (
    "service_md_count",
    "successful_service_md_count",
    "service_success_rate",
    "service_steps",
    "service_step_fraction",
    "allocated_bandwidth",
    "useful_bandwidth",
    "allocated_cpu",
    "useful_cpu",
    "episode_reward",
    "mean_reward",
    "system_performance_individual",
    "cumulative_individual_reward",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=7001)
    parser.add_argument("--local-run-dir", type=Path, default=DEFAULT_LOCAL_RUN)
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_uav_rows(base, helper, rows: list[dict], model: str, n_uavs: int) -> list[dict]:
    converted = helper.make_per_uav_rows(rows, n_uavs)
    for row in converted:
        row["model"] = model
    return converted


def aggregate_per_uav(rows: list[dict]) -> list[dict]:
    output = []
    for model in sorted({row["model"] for row in rows}):
        for uav_id in range(5):
            selected = [
                row for row in rows
                if row["model"] == model and int(row["uav_id"]) == uav_id
            ]
            item = {
                "model": model,
                "uav_id": uav_id,
                "episodes": len(selected),
                "checkpoint_step": selected[0]["checkpoint_step"],
            }
            for field in PER_UAV_FIELDS:
                values = np.asarray([float(row[field]) for row in selected], dtype=float)
                values = values[np.isfinite(values)]
                item[f"{field}_mean"] = float(values.mean()) if values.size else float("nan")
                item[f"{field}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            output.append(item)
    return output


def args_dict(run_dir: Path) -> dict:
    return json.loads((run_dir / "args.json").read_text(encoding="utf-8"))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def resource_audit(flight_dir: Path, resource_dir: Path) -> dict:
    flight = args_dict(flight_dir)
    resource = args_dict(resource_dir)
    allowed = {
        "association_threshold",
        "experiment_name",
        "device",
        "cuda",
        "cuda_deterministic",
        "model_dir",
        "checkpoint_base_steps",
        "confirm_legacy_checkpoint_frozen",
        "eval_episodes",
        "eval_interval",
        "log_interval",
        "save_interval",
        "run_dir",
        "save_dir",
        "log_dir",
        # This is the intentional branch swap in this diagnostic.
        "uav_resource_mode",
        "uav_resource_scale_factors",
    }
    differences = {}
    for key in sorted(set(flight) | set(resource)):
        if key in allowed:
            continue
        left = json_value(flight.get(key))
        right = json_value(resource.get(key))
        if json.dumps(left, sort_keys=True, ensure_ascii=False) != json.dumps(
            right, sort_keys=True, ensure_ascii=False
        ):
            differences[key] = {"flight": left, "resource": right}
    return {
        "allowed_differences": sorted(allowed),
        "intentional_resource_difference": {
            "flight": {"uav_resource_mode": flight.get("uav_resource_mode"), "uav_resource_scale_factors": flight.get("uav_resource_scale_factors")},
            "resource": {"uav_resource_mode": resource.get("uav_resource_mode"), "uav_resource_scale_factors": resource.get("uav_resource_scale_factors")},
        },
        "unexpected_differences": differences,
        "compatible_except_allowed_fields": not differences,
    }


def checkpoint_record(base, run_dir: Path, checkpoint_dir: Path, label: str) -> dict:
    step = base.checkpoint_manifest_step(checkpoint_dir)
    if step is None:
        raise RuntimeError(f"No manifest-verified checkpoint for {label}: {checkpoint_dir}")
    return {"label": label, "run_dir": str(run_dir), "checkpoint_dir": str(checkpoint_dir), "checkpoint_step": int(step)}


def evaluate_native(base, helper, run_dir, checkpoint_dir, model, seeds, output):
    args, env, actors, normers = base.load_policy(run_dir, checkpoint_dir)
    rows = []
    try:
        for index, seed in enumerate(seeds, 1):
            row = base.evaluate_episode(args, env, actors, normers, seed, checkpoint_record(base, run_dir, checkpoint_dir, model)["checkpoint_step"], deterministic=True)
            row["model"] = model
            rows.append(row)
            if index == 1 or index % 5 == 0 or index == len(seeds):
                print(json.dumps({"model": model, "episode": index, "episodes": len(seeds)}), flush=True)
    finally:
        env.close()
    return rows, args


def main():
    cli = parse_args()
    if cli.episodes < 1:
        raise ValueError("episodes must be positive")
    if cli.output_dir.exists():
        raise FileExistsError(f"Output already exists: {cli.output_dir}")
    for path in (cli.local_run_dir, cli.remote_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    cli.output_dir.mkdir(parents=True)
    (cli.output_dir / "snapshots").mkdir()

    base = load_module(BASE_SCRIPT, "eval_base_current_25")
    helper = load_module(HELPER_SCRIPT, "per_uav_helper_current_25")
    hybrid = load_module(HYBRID_SCRIPT, "hybrid_eval_current_25")
    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    torch.set_num_threads(1)

    local_checkpoint, local_files = base.snapshot_checkpoint(
        cli.local_run_dir, cli.output_dir / "snapshots" / "heterogeneous_local"
    )
    remote_checkpoint, remote_files = base.snapshot_checkpoint(
        cli.remote_root, cli.output_dir / "snapshots" / "homogeneous_remote"
    )
    local_step = base.checkpoint_manifest_step(local_checkpoint)
    remote_step = base.checkpoint_manifest_step(remote_checkpoint)
    if local_step is None or remote_step is None:
        raise RuntimeError("Snapshot manifest verification failed")

    hetero_model = "heterogeneous_local_latest"
    homo_model = "homogeneous_remote_final"
    hybrid_model = "hybrid_Fhomo_Rhetero"
    native_local, local_args = evaluate_native(
        base, helper, cli.local_run_dir, local_checkpoint, hetero_model, seeds, cli.output_dir
    )
    native_remote, remote_args = evaluate_native(
        base, helper, cli.remote_root, remote_checkpoint, homo_model, seeds, cli.output_dir
    )

    # Explicitly verify the intended hybrid contract before evaluating it.
    audit = resource_audit(cli.remote_root, cli.local_run_dir)
    if not audit["compatible_except_allowed_fields"]:
        raise RuntimeError("Unexpected flight/resource differences: " + json.dumps(audit["unexpected_differences"], ensure_ascii=False))
    flight_args = hybrid.read_args(cli.remote_root)
    resource_args = hybrid.read_args(cli.local_run_dir)
    hybrid.validate_args(flight_args, resource_args)
    runtime_psi = float(resource_args.association_threshold)
    resource_args.n_rollout_threads = 1
    resource_args.n_training_threads = 1
    resource_args.model_dir = str(local_checkpoint)
    flight_args.model_dir = str(remote_checkpoint)
    device = torch.device("cpu")
    env_hybrid = hybrid.WrappedMECEnv(args=resource_args)
    flight_actors = hybrid.load_actors(flight_args, env_hybrid, remote_checkpoint, device)
    resource_actors = hybrid.load_actors(resource_args, env_hybrid, local_checkpoint, device)
    flight_normers = hybrid.load_normers(flight_args, env_hybrid, remote_checkpoint)
    resource_normers = hybrid.load_normers(resource_args, env_hybrid, local_checkpoint)
    hybrid.validate_normers(flight_normers, resource_normers)
    hybrid_actors = [
        hybrid.HybridRActor(flight_actor, resource_actor, runtime_association_threshold=runtime_psi).eval()
        for flight_actor, resource_actor in zip(flight_actors, resource_actors)
    ]
    hybrid_rows = []
    try:
        for index, seed in enumerate(seeds, 1):
            row = base.evaluate_episode(
                resource_args,
                env_hybrid,
                hybrid_actors,
                flight_normers,
                seed,
                int(local_step),
                deterministic=True,
                resource_normers=resource_normers,
            )
            row["model"] = hybrid_model
            hybrid_rows.append(row)
            if index == 1 or index % 5 == 0 or index == len(seeds):
                print(json.dumps({"model": hybrid_model, "episode": index, "episodes": len(seeds)}), flush=True)
    finally:
        env_hybrid.close()

    all_rows = native_local + native_remote + hybrid_rows
    per_rows = []
    per_rows.extend(per_uav_rows(base, helper, native_local, hetero_model, int(local_args.n_UAVs)))
    per_rows.extend(per_uav_rows(base, helper, native_remote, homo_model, int(remote_args.n_UAVs)))
    per_rows.extend(per_uav_rows(base, helper, hybrid_rows, hybrid_model, int(resource_args.n_UAVs)))
    per_aggregate = aggregate_per_uav(per_rows)
    overall = base.aggregate(all_rows)
    scalar_rows = all_rows
    write_csv(cli.output_dir / "episode_metrics.csv", scalar_rows)
    write_csv(cli.output_dir / "overall_aggregate.csv", overall)
    write_csv(cli.output_dir / "per_uav_episode_metrics.csv", per_rows)
    write_csv(cli.output_dir / "per_uav_aggregate.csv", per_aggregate)
    (cli.output_dir / "configuration_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "episodes_per_model": len(seeds),
        "test_seeds": seeds,
        "deterministic": True,
        "metric_semantics": {
            "service_md_count": "active-MD association assignments credited to each UAV over the 400 measured steps",
            "successful_service_md_count": "those assigned MDs whose post-step complete_task flag is true",
            "service_success_rate": "successful_service_md_count / service_md_count",
            "service_step_fraction": "steps with at least one active assigned MD / 400",
            "episode_reward": "sum of that UAV's local env.step reward over 400 steps",
            "mean_reward": "episode_reward / 400",
        },
        "runs": {
            hetero_model: checkpoint_record(base, cli.local_run_dir, local_checkpoint, hetero_model) | {
                "uav_resource_mode": getattr(local_args, "uav_resource_mode", None),
                "uav_resource_scale_factors": getattr(local_args, "uav_resource_scale_factors", None),
            },
            homo_model: checkpoint_record(base, cli.remote_root, remote_checkpoint, homo_model) | {
                "uav_resource_mode": getattr(remote_args, "uav_resource_mode", None),
                "uav_resource_scale_factors": getattr(remote_args, "uav_resource_scale_factors", None),
            },
            hybrid_model: {
                "flight_checkpoint_step": int(remote_step),
                "resource_checkpoint_step": int(local_step),
                "flight_source": str(cli.remote_root),
                "resource_source": str(cli.local_run_dir),
                "runtime_association_threshold": runtime_psi,
                "flight_branch": "homogeneous remote final: flight_base + flight head",
                "resource_association_branch": "heterogeneous local latest: resource base + association/bandwidth/CPU heads",
            },
        },
        "overall_aggregate": overall,
        "per_uav_aggregate": per_aggregate,
    }
    (cli.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (cli.output_dir / "protocol.md").write_text(
        f"""# 25-seed deterministic homogeneous/heterogeneous/hybrid evaluation\n\n"
        f"Common seeds: {seeds[0]}--{seeds[-1]}.\n\n"
        f"- heterogeneous native: local latest verified checkpoint at {int(local_step):,} steps; resource factors [1.0, 1.3, 0.7, 1.3, 0.7].\n"
        f"- homogeneous native: remote final verified checkpoint at {int(remote_step):,} steps; equal resources.\n"
        f"- hybrid: homogeneous remote flight branch + heterogeneous local resource/association branches; runtime psi={runtime_psi:g}.\n"
        "- All three use the resource/heterogeneous environment configuration for the hybrid rollout, and the same deterministic seeds.\n"
        "- Per-UAV service counts are association/load assignments, not unique-MD counts; global task metrics remain in overall_aggregate.csv.\n"
        """,
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(cli.output_dir),
        "episodes_per_model": len(seeds),
        "checkpoint_steps": {hetero_model: int(local_step), homo_model: int(remote_step), hybrid_model: {"flight": int(remote_step), "resource": int(local_step)}},
        "rows": {"episode": len(all_rows), "per_uav_episode": len(per_rows)},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
