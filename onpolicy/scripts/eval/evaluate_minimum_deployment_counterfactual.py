#!/usr/bin/env python
"""Paired learned-path versus direct-deployment counterfactual."""

import argparse
import csv
import json
import sys
from argparse import Namespace
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.scripts.eval.compare_mappo_32_vs_41 import (
    load_policy,
    make_trace,
    paired_summary,
    run_condition,
    waypoint_path,
)


METRICS = (
    "system_performance_true_all_GUs",
    "system_performance_equivalent_full_GUs",
    "system_performance",
    "training_return",
    "mean_active_mds",
    "admission_ratio",
    "completion_ratio",
    "served_ratio",
    "departed",
)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--baseline-episode", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=620000)
    return parser.parse_args()


def direct_path(starts, targets, steps, max_step):
    return np.stack([
        waypoint_path(start, [target], steps, max_step)
        for start, target in zip(starts, targets)
    ], axis=1)


def path_stats(path, args):
    increments = np.linalg.norm(np.diff(path, axis=0), axis=2)
    pairwise = np.linalg.norm(path[:, :, None] - path[:, None, :], axis=3)
    pairwise[:, np.arange(args.n_UAVs), np.arange(args.n_UAVs)] = np.inf
    return {
        "total_length": float(increments.sum()),
        "per_uav_length": increments.sum(axis=0).tolist(),
        "max_speed": float(np.max(increments) / args.Delta_t),
        "min_separation": float(np.min(pairwise)),
        "last_100_length": float(increments[-100:].sum()),
    }


def shortest_safe_assignment(starts, targets, steps, args):
    candidates = []
    for order in permutations(range(args.n_UAVs)):
        assigned = targets[list(order)]
        path = direct_path(starts, assigned, steps, args.v_max * args.Delta_t)
        stats = path_stats(path, args)
        if stats["min_separation"] >= args.Dis_min:
            candidates.append((stats["total_length"], order, assigned, path, stats))
    if not candidates:
        raise RuntimeError("No collision-safe direct assignment exists")
    return min(candidates, key=lambda item: item[0])


def main():
    cli = parse_cli()
    checkpoint_dir = cli.checkpoint_dir.resolve()
    output_dir = cli.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    with (checkpoint_dir / "args.json").open("r", encoding="utf-8") as handle:
        args = Namespace(**json.load(handle))
    args.n_rollout_threads = args.n_training_threads = 1
    args.model_dir = str(checkpoint_dir)
    if not (
        args.dynamic_md and args.cartesian_flight and args.md_arrivals_per_region == [1, 5]
        and args.md_lifetime_min == args.md_lifetime_max == 10
    ):
        raise ValueError("This diagnostic expects the strict 1+5 Cartesian environment")
    torch.set_num_threads(1)
    actors, normers = load_policy(args, checkpoint_dir)

    episode = np.load(cli.baseline_episode.resolve())
    learned = np.asarray(episode["uav_positions"], dtype=np.float64)
    steps = int(args.episode_length)
    if learned.shape != (steps + 1, args.n_UAVs, 2):
        raise ValueError(f"Unexpected learned path shape: {learned.shape}")
    starts = learned[0]
    targets = learned[-100:].mean(axis=0)
    direct_same = direct_path(starts, targets, steps, args.v_max * args.Delta_t)
    _, order, matched_targets, direct_matched, matched_stats = shortest_safe_assignment(
        starts, targets, steps, args
    )

    conditions = {
        "learned_path": learned,
        "direct_same_identity": direct_same,
        "direct_minimum_assignment": direct_matched,
    }
    path_summaries = {name: path_stats(path, args) for name, path in conditions.items()}
    for name, stats in path_summaries.items():
        if stats["max_speed"] > args.v_max + 1e-3 or stats["min_separation"] < args.Dis_min:
            raise RuntimeError(f"Invalid prescribed path: {name}")
    output_dir.mkdir(parents=True)

    rows = []
    for trial in range(cli.trials):
        seed = cli.seed_base + trial
        trace = make_trace(args, seed, steps)
        for name, path in conditions.items():
            result = run_condition(args, trace, path, "actor", actors, normers)
            rows.append({"trace_seed": seed, "condition": name, **result})
        print(f"paired trace {trial + 1}/{cli.trials}: {seed}", flush=True)

    replay_trace = make_trace(args, cli.seed_base, steps)
    if run_condition(args, replay_trace, learned, "actor", actors, normers) != run_condition(
        args, replay_trace, learned, "actor", actors, normers
    ):
        raise AssertionError("Identical paired replay is not deterministic")

    with (output_dir / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparisons = []
    for baseline, counterfactual in (
        ("learned_path", "direct_same_identity"),
        ("direct_same_identity", "direct_minimum_assignment"),
        ("learned_path", "direct_minimum_assignment"),
    ):
        for metric in METRICS:
            comparisons.append(paired_summary(rows, baseline, counterfactual, metric))

    summary = {
        "status": "completed",
        "protocol": {
            "checkpoint_dir": str(checkpoint_dir),
            "baseline_episode": str(cli.baseline_episode.resolve()),
            "common_random_trace": "same strict 1+5 births, reflected mobility, and tasks",
            "resource_allocator": "saved deterministic actor resource heads",
            "changed_variable": "only the prescribed UAV flight path",
            "trials": cli.trials,
            "seed_base": cli.seed_base,
        },
        "targets": targets.tolist(),
        "minimum_assignment_target_order": list(order),
        "minimum_assignment_targets": matched_targets.tolist(),
        "paths": path_summaries,
        "comparisons": comparisons,
        "validations": {
            "identical_replay_exact": True,
            "all_paths_speed_and_safety_valid": True,
            "minimum_assignment_stats": matched_stats,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(output_dir)


if __name__ == "__main__":
    main()
