"""Evaluate a fixed flight checkpoint with several resource checkpoints.

This is the requested association-threshold diagnostic:

* every policy uses the same deterministic seeds;
* every hybrid uses the psi=0.5 checkpoint for ``flight_base`` and the flight
  action head;
* the resource/association base and heads, normer, and runtime threshold come
  from that row's resource checkpoint;
* all non-metadata configuration fields must match.

The script does not train or alter ordinary MAPPO evaluation.  It writes the
full per-episode metrics, episode-wise aggregates, pooled task-count metrics,
and a configuration audit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
HYBRID_EVALUATOR = HERE / "evaluate_hybrid.py"
SUMMARY_SCRIPT = (
    PROJECT_ROOT
    / "analysis"
    / "psi_curriculum_seed2_30det_latest_20260817"
    / "summarize_pooled.py"
)
sys.path.insert(0, str(PROJECT_ROOT))


def load_helper():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hybrid_helper_for_psi_sweep", HYBRID_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {HYBRID_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-run-dir", type=Path, required=True)
    parser.add_argument(
        "--resource-run",
        action="append",
        required=True,
        metavar="PSI=PATH",
        help="Repeat for each resource branch, e.g. 0.3=D:/.../run1",
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=8001)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def parse_resource_runs(values):
    result = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--resource-run must be PSI=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        try:
            requested_psi = float(label)
        except ValueError as exc:
            raise ValueError(f"Invalid resource psi {label!r}") from exc
        if not 0.0 <= requested_psi <= 1.0:
            raise ValueError(f"Resource psi must be in [0,1], got {requested_psi}")
        if requested_psi in seen:
            raise ValueError(f"Duplicate resource psi {requested_psi}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        result.append((requested_psi, path))
        seen.add(requested_psi)
    return sorted(result, key=lambda item: item[0])


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def main():
    cli = parse_args()
    if cli.episodes < 1:
        raise ValueError("episodes must be positive")
    if not cli.flight_run_dir.is_dir():
        raise FileNotFoundError(cli.flight_run_dir)
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)

    helper = load_helper()
    base = helper.load_base()
    resource_runs = parse_resource_runs(cli.resource_run)
    if len(resource_runs) != 5:
        raise ValueError(f"Expected five resource branches, got {len(resource_runs)}")

    flight_args = helper.read_args(cli.flight_run_dir)
    resource_args_by_psi = {}
    audits = {}
    for requested_psi, resource_run in resource_runs:
        resource_args = helper.read_args(resource_run)
        helper.validate_args(flight_args, resource_args)
        audit = helper.audit_configs(cli.flight_run_dir, resource_run)
        if not audit["compatible_except_allowed_fields"]:
            raise RuntimeError(
                f"Incompatible flight/resource configs for psi={requested_psi}: "
                + json.dumps(audit["unexpected_differences"], ensure_ascii=False)
            )
        actual_psi = float(resource_args.association_threshold)
        if abs(actual_psi - requested_psi) > 1e-12:
            raise ValueError(
                f"Resource path {resource_run} declares psi={actual_psi}, "
                f"but was labelled {requested_psi}"
            )
        resource_args_by_psi[requested_psi] = resource_args
        audits[str(requested_psi)] = {
            "resource_run_dir": str(resource_run),
            **audit,
        }

    runtime_psi_flight = float(flight_args.association_threshold)
    if abs(runtime_psi_flight - 0.5) > 1e-12:
        raise ValueError(
            f"The fixed flight checkpoint must be psi=0.5, got {runtime_psi_flight}"
        )

    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    cli.output_dir.mkdir(parents=True)
    (cli.output_dir / "snapshots").mkdir()
    (cli.output_dir / "configuration_audit.json").write_text(
        json.dumps(
            {
                "flight_run_dir": str(cli.flight_run_dir.resolve()),
                "audits_by_resource_psi": audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    flight_checkpoint, flight_files = base.snapshot_checkpoint(
        cli.flight_run_dir, cli.output_dir / "snapshots" / "flight_psi0p5"
    )
    flight_step = base.checkpoint_manifest_step(flight_checkpoint)
    if flight_step is None:
        raise RuntimeError("Could not verify the flight checkpoint manifest")

    device = torch.device("cpu")
    # Every resource configuration has the same observation/action shapes;
    # use the first resource environment to instantiate the fixed flight net.
    first_resource_args = resource_args_by_psi[resource_runs[0][0]]
    first_resource_args.n_rollout_threads = 1
    first_resource_args.n_training_threads = 1
    flight_env = helper.WrappedMECEnv(args=first_resource_args)
    flight_actors = helper.load_actors(
        flight_args, flight_env, flight_checkpoint, device
    )
    flight_normers = helper.load_normers(
        flight_args, flight_env, flight_checkpoint
    )

    rows = []
    resource_metadata = []
    deterministic = not cli.stochastic
    try:
        for requested_psi, resource_run in resource_runs:
            resource_args = resource_args_by_psi[requested_psi]
            resource_args.n_rollout_threads = 1
            resource_args.n_training_threads = 1
            resource_args.model_dir = str(resource_run / "models")

            snapshot_dir, resource_files = base.snapshot_checkpoint(
                resource_run,
                cli.output_dir / "snapshots" / f"resource_psi{requested_psi:g}",
            )
            resource_step = base.checkpoint_manifest_step(snapshot_dir)
            if resource_step is None:
                raise RuntimeError(
                    f"Could not verify resource checkpoint for psi={requested_psi}"
                )

            env = helper.WrappedMECEnv(args=resource_args)
            try:
                resource_actors = helper.load_actors(
                    resource_args, env, snapshot_dir, device
                )
                resource_normers = helper.load_normers(
                    resource_args, env, snapshot_dir
                )
                helper.validate_normers(flight_normers, resource_normers)
                hybrid_actors = [
                    helper.HybridRActor(
                        flight_actor,
                        resource_actor,
                        runtime_association_threshold=requested_psi,
                    ).eval()
                    for flight_actor, resource_actor in zip(
                        flight_actors, resource_actors
                    )
                ]

                for index, seed in enumerate(seeds, 1):
                    row = base.evaluate_episode(
                        resource_args,
                        env,
                        hybrid_actors,
                        flight_normers,
                        seed,
                        resource_step,
                        deterministic=deterministic,
                        resource_normers=resource_normers,
                    )
                    row["model"] = (
                        f"hybrid_Fpsi0p5_Rpsi{requested_psi:g}_"
                        f"rtpsi{requested_psi:g}"
                    )
                    row["psi"] = requested_psi
                    row["eval_seed"] = seed
                    rows.append(row)
                    if index == 1 or index % 5 == 0 or index == len(seeds):
                        print(
                            json.dumps(
                                {
                                    "psi": requested_psi,
                                    "episode": index,
                                    "episodes": len(seeds),
                                    "seed": seed,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
            finally:
                env.close()

            resource_metadata.append(
                {
                    "psi": requested_psi,
                    "run_dir": str(resource_run),
                    "checkpoint_step": resource_step,
                    "checkpoint_files": resource_files,
                }
            )
    finally:
        flight_env.close()

    base.write_csv(cli.output_dir / "episode_metrics.csv", rows)
    base.write_csv(cli.output_dir / "aggregate_metrics.csv", base.aggregate(rows))

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "deterministic": deterministic,
        "test_seeds": seeds,
        "flight_run_dir": str(cli.flight_run_dir.resolve()),
        "flight_checkpoint_step": flight_step,
        "flight_checkpoint_files": flight_files,
        "flight_runtime_psi": runtime_psi_flight,
        "resource_branches": resource_metadata,
        "protocol": (
            "All five policies use flight psi=0.5 base/head/normer; each row "
            "uses its own resource base/heads/normer and runtime psi."
        ),
    }
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cli.output_dir / "protocol.md").write_text(
        f"""# Fixed-flight psi sweep\n\n"
        f"- flight checkpoint: `{cli.flight_run_dir.resolve()}`\n"
        f"- flight branch: psi=0.5 checkpoint `flight_base` plus flight head\n"
        f"- resource branches: `{', '.join(f'{psi:g}' for psi, _ in resource_runs)}`\n"
        f"- deterministic seeds: `{seeds[0]}--{seeds[-1]}`\n"
        "- every resource branch has its own resource base, association/resource heads, normer, and runtime psi\n"
        "- configuration audit rejects any non-metadata difference\n"
        "- this is an evaluation-only branch-swap diagnostic; it does not train a hybrid policy\n"
        """,
        encoding="utf-8",
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
    print(
        json.dumps(
            {
                "output_dir": str(cli.output_dir),
                "flight_checkpoint_step": flight_step,
                "resource_branches": len(resource_runs),
                "episodes_per_branch": len(seeds),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
