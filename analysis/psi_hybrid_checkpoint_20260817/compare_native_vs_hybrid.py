"""Compare a native checkpoint with a same-checkpoint hybrid composition.

The two policies are evaluated on identical deterministic episode seeds.  The
native policy uses one actor and one observation normalizer.  The hybrid
policy uses the same checkpoint twice: its flight base/head are taken from the
flight copy and its resource/association base/heads are taken from the
resource copy, with separate normalizer objects.  This is an evaluation-only
equivalence check, not a training path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
# HERE is the analysis subdirectory; its second parent is the repository root.
PROJECT_ROOT = HERE.parents[1]
HYBRID_EVALUATOR = HERE / "evaluate_hybrid.py"
SUMMARY_SCRIPT = (
    PROJECT_ROOT
    / "analysis"
    / "psi_curriculum_seed2_30det_latest_20260817"
    / "summarize_pooled.py"
)
sys.path.insert(0, str(PROJECT_ROOT))


def load_hybrid_evaluator():
    spec = importlib.util.spec_from_file_location(
        "hybrid_evaluator_for_native_comparison", HYBRID_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {HYBRID_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=8001)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def compare_rows(native_rows, hybrid_rows, metrics):
    native_by_seed = {int(row["eval_seed"]): row for row in native_rows}
    hybrid_by_seed = {int(row["eval_seed"]): row for row in hybrid_rows}
    seeds = sorted(set(native_by_seed) & set(hybrid_by_seed))
    by_metric = {}
    all_equal = True
    for metric in metrics:
        differences = []
        pairs = []
        for seed in seeds:
            left = finite_float(native_by_seed[seed].get(metric))
            right = finite_float(hybrid_by_seed[seed].get(metric))
            if left is None or right is None:
                continue
            difference = abs(left - right)
            differences.append(difference)
            pairs.append(
                {
                    "seed": seed,
                    "native": left,
                    "hybrid": right,
                    "absolute_difference": difference,
                }
            )
        max_difference = max(differences) if differences else None
        metric_equal = bool(max_difference is not None and max_difference <= 1e-9)
        if not metric_equal:
            all_equal = False
        by_metric[metric] = {
            "max_absolute_difference": max_difference,
            "allclose_at_1e-9": metric_equal,
            "pairs": pairs,
        }
    return {
        "seeds_compared": seeds,
        "metrics": by_metric,
        "all_metrics_equal_at_1e-9": all_equal,
    }


def main():
    cli = parse_args()
    if cli.episodes < 1:
        raise ValueError("episodes must be positive")
    if not cli.run_dir.is_dir():
        raise FileNotFoundError(cli.run_dir)
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)

    helper = load_hybrid_evaluator()
    base = helper.load_base()
    run_dir = cli.run_dir.resolve()
    run_args = helper.read_args(run_dir)
    helper.validate_args(run_args, run_args)
    runtime_psi = float(run_args.association_threshold)
    if not 0.0 <= runtime_psi <= 1.0:
        raise ValueError("checkpoint association_threshold must be in [0, 1]")

    run_args.n_rollout_threads = 1
    run_args.n_training_threads = 1
    run_args.model_dir = str(run_dir / "models")
    seeds = list(range(cli.seed_start, cli.seed_start + cli.episodes))
    cli.output_dir.mkdir(parents=True)
    snapshot_dir, checkpoint_files = base.snapshot_checkpoint(
        run_dir, cli.output_dir / "snapshots" / "psi0p5"
    )
    checkpoint_step = base.checkpoint_manifest_step(snapshot_dir)
    if checkpoint_step is None:
        raise RuntimeError("The checkpoint manifest did not provide a verified step")

    device = torch.device("cpu")
    env_native = helper.WrappedMECEnv(args=run_args)
    env_hybrid = helper.WrappedMECEnv(args=run_args)
    try:
        # Load three independent actor lists.  The first is the native path;
        # the other two are deliberately separate reads used as the flight
        # and resource branches of the hybrid path.
        native_actors = helper.load_actors(run_args, env_native, snapshot_dir, device)
        flight_actors = helper.load_actors(run_args, env_hybrid, snapshot_dir, device)
        resource_actors = helper.load_actors(run_args, env_hybrid, snapshot_dir, device)
        native_normers = helper.load_normers(run_args, env_native, snapshot_dir)
        flight_normers = helper.load_normers(run_args, env_hybrid, snapshot_dir)
        resource_normers = helper.load_normers(run_args, env_hybrid, snapshot_dir)
        helper.validate_normers(flight_normers, resource_normers)
        hybrid_actors = [
            helper.HybridRActor(
                flight_actor,
                resource_actor,
                runtime_association_threshold=runtime_psi,
            ).eval()
            for flight_actor, resource_actor in zip(flight_actors, resource_actors)
        ]

        native_rows = []
        hybrid_rows = []
        deterministic = not cli.stochastic
        for index, seed in enumerate(seeds, 1):
            native_row = base.evaluate_episode(
                run_args,
                env_native,
                native_actors,
                native_normers,
                seed,
                checkpoint_step,
                deterministic=deterministic,
            )
            native_row["model"] = "native_psi0p5"
            native_row["eval_seed"] = seed
            native_rows.append(native_row)

            hybrid_row = base.evaluate_episode(
                run_args,
                env_hybrid,
                hybrid_actors,
                flight_normers,
                seed,
                checkpoint_step,
                deterministic=deterministic,
                resource_normers=resource_normers,
            )
            hybrid_row["model"] = "hybrid_Fpsi0p5_Rpsi0p5_rtpsi0p5"
            hybrid_row["eval_seed"] = seed
            hybrid_rows.append(hybrid_row)
            print(
                json.dumps(
                    {"episode": index, "episodes": len(seeds), "seed": seed},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        env_native.close()
        env_hybrid.close()

    rows = native_rows + hybrid_rows
    base.write_csv(cli.output_dir / "episode_metrics.csv", rows)
    base.write_csv(cli.output_dir / "aggregate_metrics.csv", base.aggregate(rows))
    comparison = compare_rows(native_rows, hybrid_rows, base.METRICS)
    (cli.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "deterministic": deterministic,
        "test_seeds": seeds,
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        "checkpoint_files": checkpoint_files,
        "association_threshold": runtime_psi,
        "native_normer_source": str(snapshot_dir),
        "flight_normer_source": str(snapshot_dir),
        "resource_normer_source": str(snapshot_dir),
        "comparison": "same checkpoint, same environment configuration, same seeds",
    }
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cli.output_dir / "protocol.md").write_text(
        f"""# Native vs same-checkpoint hybrid evaluation\n\n"
        f"- source run: `{run_dir}`\n"
        f"- checkpoint step: `{checkpoint_step}`\n"
        f"- deterministic: `{deterministic}`\n"
        f"- seeds: `{seeds[0]}--{seeds[-1]}`\n"
        "- native: one actor plus one normer\n"
        "- hybrid: the same actor checkpoint used as flight and resource source;\n"
        "  separate flight/resource normer instances; runtime psi comes from the checkpoint\n"
        "- metrics: the full METRICS set from evaluate_final_checkpoints.py\n"
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
                "checkpoint_step": checkpoint_step,
                "episodes_per_model": len(seeds),
                "all_metrics_equal_at_1e-9": comparison[
                    "all_metrics_equal_at_1e-9"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
