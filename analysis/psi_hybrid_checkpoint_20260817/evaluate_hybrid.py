"""Evaluate a hybrid actor assembled from two compatible checkpoints.

The flight checkpoint supplies ``flight_base`` and the flight head.  The
resource checkpoint supplies ``base``, association/bandwidth/CPU heads, its
normer, and the runtime association threshold.  This is an evaluation-only
diagnostic; ordinary training remains on ``R_Actor``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_EVALUATOR = PROJECT_ROOT / "analysis" / "psi_reviewer_final_metrics_20260816" / "evaluate_final_checkpoints.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "analysis" / "psi_curriculum_seed2_30det_latest_20260817" / "summarize_pooled.py"
sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.algorithms.r_mappo.algorithm.hybrid_actor import HybridRActor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.mec.env_maker import WrappedMECEnv
from onpolicy.envs.mec.vec_normalize import Normer


# These are run metadata, not policy/environment semantics.  All other keys
# must match so a hybrid cannot silently combine incompatible experiments.
ALLOWED_CONFIG_DIFFERENCES = {
    "association_threshold",
    "seed",
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
}


def load_base():
    spec = importlib.util.spec_from_file_location("reviewer_metrics_base_hybrid", BASE_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-run-dir", type=Path, required=True)
    parser.add_argument("--resource-run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=8001)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-psi",
        type=float,
        default=None,
        help="Optional explicit runtime threshold. Default: resource checkpoint psi.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample continuous actions instead of using deterministic modes.",
    )
    return parser.parse_args()


def read_args(run_dir: Path) -> SimpleNamespace:
    with (run_dir / "args.json").open("r", encoding="utf-8") as handle:
        return SimpleNamespace(**json.load(handle))


def read_args_dict(run_dir: Path) -> dict:
    with (run_dir / "args.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def audit_configs(flight_run_dir: Path, resource_run_dir: Path) -> dict:
    flight = read_args_dict(flight_run_dir)
    resource = read_args_dict(resource_run_dir)
    differences = {}
    for key in sorted(set(flight) | set(resource)):
        if key in ALLOWED_CONFIG_DIFFERENCES:
            continue
        left = json_value(flight.get(key))
        right = json_value(resource.get(key))
        if json.dumps(left, sort_keys=True, ensure_ascii=False) != json.dumps(
            right, sort_keys=True, ensure_ascii=False
        ):
            differences[key] = {"flight": left, "resource": right}
    return {
        "allowed_differences": sorted(ALLOWED_CONFIG_DIFFERENCES),
        "unexpected_differences": differences,
        "compatible_except_allowed_fields": not differences,
        "flight_association_threshold": flight.get("association_threshold"),
        "resource_association_threshold": resource.get("association_threshold"),
    }


def validate_args(flight_args, resource_args):
    for name, args in (("flight", flight_args), ("resource", resource_args)):
        if bool(getattr(args, "use_atten_actor", False)):
            raise ValueError(f"{name} actor attention path is not supported by HybridRActor")
        if not bool(getattr(args, "spatial_flight_actor", False)):
            raise ValueError(f"{name} checkpoint must use spatial_flight_actor")
        if not bool(getattr(args, "cartesian_flight", False)):
            raise ValueError(f"{name} checkpoint must use cartesian_flight")
        if bool(getattr(args, "fix_uav_pos", False)):
            raise ValueError("HybridRActor currently requires moving UAVs (fix_uav_pos=false)")
        if not bool(getattr(args, "continuous_associate", False)):
            raise ValueError("HybridRActor requires continuous association")
        if bool(getattr(args, "ave_resource", False)) or bool(
            getattr(args, "ave_bandwidth", False)
        ) or bool(getattr(args, "nearest_associate", False)):
            raise ValueError("HybridRActor requires four-head continuous association/resource actions")
        if bool(getattr(args, "use_recurrent_policy", False)) or bool(
            getattr(args, "use_naive_recurrent_policy", False)
        ):
            raise ValueError("HybridRActor currently supports feed-forward policies only")
    if int(flight_args.n_UAVs) != int(resource_args.n_UAVs):
        raise ValueError("Flight/resource n_UAVs differ")


def load_actors(args, env, checkpoint_dir: Path, device):
    actors = []
    for agent_id in range(int(args.n_UAVs)):
        actor = R_Actor(
            args,
            env.observation_space[agent_id],
            env.action_space[agent_id],
            device,
        )
        checkpoint = checkpoint_dir / f"actor_agent{agent_id}.pt"
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        actor.load_state_dict(state)
        actor.eval()
        actors.append(actor)
    return actors


def load_normers(args, env, checkpoint_dir: Path):
    normers = []
    for agent_id in range(int(args.n_UAVs)):
        normer = Normer(
            args=args,
            obs_space=env.observation_space[agent_id],
            states_space=env.share_observation_space[agent_id],
        )
        normer.load(checkpoint_dir / f"normer{agent_id}.pkl")
        normers.append(normer)
    return normers


def validate_normers(flight_normers, resource_normers):
    if len(flight_normers) != len(resource_normers):
        raise ValueError("Flight/resource normer counts differ")
    for agent_id, (flight_normer, resource_normer) in enumerate(
        zip(flight_normers, resource_normers)
    ):
        flight_shape = tuple(np.asarray(flight_normer.ob_rms.mean).shape) if flight_normer.ob_rms else None
        resource_shape = tuple(np.asarray(resource_normer.ob_rms.mean).shape) if resource_normer.ob_rms else None
        if flight_shape != resource_shape:
            raise ValueError(
                f"Normer observation shapes differ for agent {agent_id}: "
                f"{flight_shape} vs {resource_shape}"
            )
        if tuple(getattr(flight_normer, "obs_preserve_slices", ())) != tuple(
            getattr(resource_normer, "obs_preserve_slices", ())
        ):
            raise ValueError(
                f"Normer preserved-observation layouts differ for agent {agent_id}"
            )


def write_protocol(output_dir: Path, args, flight_run_dir: Path, resource_run_dir: Path, seeds, runtime_psi):
    text = f"""# Hybrid checkpoint evaluation

- flight checkpoint: `{flight_run_dir}`
- resource checkpoint: `{resource_run_dir}`
- flight branch: `flight_base` + `act.action_outs[0]`
- resource branch: `base` + `act.action_outs[1:4]`
- flight normer: loaded from the flight checkpoint
- resource normer: loaded from the resource checkpoint
- runtime association threshold: {runtime_psi:g}
- deterministic: true unless `--stochastic` is supplied
- test seeds: {seeds[0]}--{seeds[-1]}

The environment and all post-processing use the resource checkpoint's args.
The threshold is a runtime rule, not a state-dict parameter.  This is an
evaluation-only branch; it is not a newly trained policy and should be
reported as a cross-checkpoint branch-swap diagnostic.
"""
    (output_dir / "hybrid_protocol.md").write_text(text, encoding="utf-8")


def main():
    cli = parse_args()
    if cli.episodes < 1:
        raise ValueError("episodes must be positive")
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    if not cli.flight_run_dir.is_dir() or not cli.resource_run_dir.is_dir():
        raise FileNotFoundError("Both run directories must exist")

    flight_args = read_args(cli.flight_run_dir)
    resource_args = read_args(cli.resource_run_dir)
    validate_args(flight_args, resource_args)
    audit = audit_configs(cli.flight_run_dir, cli.resource_run_dir)
    if not audit["compatible_except_allowed_fields"]:
        raise RuntimeError(
            "Incompatible flight/resource configs; see printed differences: "
            + json.dumps(audit["unexpected_differences"], ensure_ascii=False)
        )

    runtime_psi = (
        float(resource_args.association_threshold)
        if cli.runtime_psi is None
        else float(cli.runtime_psi)
    )
    if not 0.0 <= runtime_psi <= 1.0:
        raise ValueError("runtime psi must be in [0, 1]")
    resource_args.association_threshold = runtime_psi
    resource_args.n_rollout_threads = 1
    resource_args.n_training_threads = 1
    resource_args.model_dir = str(cli.resource_run_dir / "models")

    cli.output_dir.mkdir(parents=True)
    (cli.output_dir / "snapshots").mkdir()
    (cli.output_dir / "configuration_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))

    base = load_base()
    flight_checkpoint, flight_files = base.snapshot_checkpoint(
        cli.flight_run_dir, cli.output_dir / "snapshots" / "flight"
    )
    resource_checkpoint, resource_files = base.snapshot_checkpoint(
        cli.resource_run_dir, cli.output_dir / "snapshots" / "resource"
    )
    flight_step = base.checkpoint_manifest_step(flight_checkpoint)
    resource_step = base.checkpoint_manifest_step(resource_checkpoint)
    if flight_step is None or resource_step is None:
        raise RuntimeError("Both checkpoints require verified checkpoint manifests")

    device = torch.device("cpu")
    env = WrappedMECEnv(args=resource_args)
    flight_actors = load_actors(flight_args, env, flight_checkpoint, device)
    resource_actors = load_actors(resource_args, env, resource_checkpoint, device)
    flight_normers = load_normers(flight_args, env, flight_checkpoint)
    resource_normers = load_normers(resource_args, env, resource_checkpoint)
    validate_normers(flight_normers, resource_normers)
    hybrid_actors = [
        HybridRActor(
            flight_actor,
            resource_actor,
            runtime_association_threshold=runtime_psi,
        ).eval()
        for flight_actor, resource_actor in zip(flight_actors, resource_actors)
    ]

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "deterministic": not cli.stochastic,
        "test_seeds": seeds,
        "runtime_association_threshold": runtime_psi,
        "flight_run_dir": str(cli.flight_run_dir),
        "resource_run_dir": str(cli.resource_run_dir),
        "flight_checkpoint_step": flight_step,
        "resource_checkpoint_step": resource_step,
        "flight_checkpoint_files": flight_files,
        "resource_checkpoint_files": resource_files,
        "configuration_audit": audit,
    }

    torch.set_num_threads(1)
    rows = []
    try:
        for index, seed in enumerate(seeds, 1):
            row = base.evaluate_episode(
                resource_args,
                env,
                hybrid_actors,
                flight_normers,
                seed,
                resource_step,
                deterministic=not cli.stochastic,
                resource_normers=resource_normers,
            )
            row["model"] = (
                f"hybrid_Fpsi{float(flight_args.association_threshold):g}_"
                f"Rpsi{float(resource_args.association_threshold):g}_"
                f"rtpsi{runtime_psi:g}"
            )
            rows.append(row)
            if index == 1 or index % 5 == 0 or index == len(seeds):
                print(json.dumps({"episode": index, "episodes": len(seeds)}), flush=True)
    finally:
        env.close()

    base.write_csv(cli.output_dir / "episode_metrics.csv", rows)
    base.write_csv(cli.output_dir / "aggregate_metrics.csv", base.aggregate(rows))
    write_protocol(
        cli.output_dir,
        resource_args,
        cli.flight_run_dir,
        cli.resource_run_dir,
        seeds,
        runtime_psi,
    )
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if SUMMARY_SCRIPT.is_file():
        subprocess.run(
            [
                sys.executable,
                str(SUMMARY_SCRIPT),
                "--input-dir",
                str(cli.output_dir),
            ],
            check=True,
        )
    print(json.dumps({"output_dir": str(cli.output_dir), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
