#!/usr/bin/env python
"""Compare fixed-reset evaluations with seed-paired statistics."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from onpolicy.scripts.eval.evaluate_dynamic_mappo_fixed import (
    METRIC_DIRECTIONS,
    SUMMARY_METRICS,
)


SEED_SETS = {
    "development": list(range(61001, 61021)),
    "validation": list(range(71001, 71041)),
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="LABEL=EVAL_DIR",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seed-set", choices=sorted(SEED_SETS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_evaluations(values):
    evaluations = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=EVAL_DIR, got: {value}")
        label, raw_path = value.split("=", 1)
        if not label or label in evaluations:
            raise ValueError(f"empty or duplicate evaluation label: {label!r}")
        evaluations[label] = Path(raw_path).resolve()
    return evaluations


def read_rows(eval_dir):
    with (eval_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("status") != "completed":
        raise ValueError(f"incomplete evaluation: {eval_dir}")
    if summary.get("reset_mode") != "fixed":
        raise ValueError(f"evaluation is not fixed-reset: {eval_dir}")
    if summary.get("actor_mode") != "deterministic":
        raise ValueError(f"evaluation is not deterministic: {eval_dir}")
    if summary.get("checkpoint_write_state") != "caller-confirmed frozen before snapshot":
        raise ValueError(f"checkpoint was not confirmed frozen: {eval_dir}")
    if summary.get("normalization") != "frozen saved statistics":
        raise ValueError(f"normalization was not frozen: {eval_dir}")

    rows = {}
    with (eval_dir / "episodes.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            if seed in rows:
                raise ValueError(f"duplicate seed {seed} in {eval_dir}")
            if row["fixed_reset"].lower() != "true":
                raise ValueError(f"seed {seed} was not fixed-reset in {eval_dir}")
            for metric in SUMMARY_METRICS:
                if metric not in row or not np.isfinite(float(row[metric])):
                    raise ValueError(
                        f"seed {seed} has missing/non-finite {metric} in {eval_dir}"
                    )
            rows[seed] = row
    return summary, rows


def paired_sign_flip_pvalue(differences, n_resamples=100_000, seed=20260803):
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(np.mean(differences)))
    if not differences.size or observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    batch_size = 10_000
    while completed < n_resamples:
        count = min(batch_size, n_resamples - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, differences.size))
        permuted = np.mean(signs * differences, axis=1)
        exceedances += int(np.sum(np.abs(permuted) >= observed - 1e-15))
        completed += count
    return float((exceedances + 1) / (n_resamples + 1))


def paired_stats(baseline, candidate, direction=1):
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired samples must be one-dimensional and shape-matched")
    finite = np.isfinite(baseline) & np.isfinite(candidate)
    baseline = baseline[finite]
    candidate = candidate[finite]
    if not baseline.size:
        raise ValueError("paired samples contain no finite values")
    differences = candidate - baseline
    improvements = direction * differences
    mean_difference = float(np.mean(differences))
    std_difference = (
        float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    )
    critical = (
        float(student_t.ppf(0.975, differences.size - 1))
        if differences.size > 1
        else 0.0
    )
    ci_half_width = critical * std_difference / np.sqrt(differences.size)
    baseline_mean = float(np.mean(baseline))
    return {
        "count": int(differences.size),
        "baseline_mean": baseline_mean,
        "candidate_mean": float(np.mean(candidate)),
        "mean_difference": mean_difference,
        "std_difference": std_difference,
        "ci95_half_width": float(ci_half_width),
        "ci95_low": float(mean_difference - ci_half_width),
        "ci95_high": float(mean_difference + ci_half_width),
        "direction": int(direction),
        "direction_adjusted_mean_improvement": float(np.mean(improvements)),
        "direction_adjusted_ci95_low": float(
            min(
                direction * (mean_difference - ci_half_width),
                direction * (mean_difference + ci_half_width),
            )
        ),
        "direction_adjusted_ci95_high": float(
            max(
                direction * (mean_difference - ci_half_width),
                direction * (mean_difference + ci_half_width),
            )
        ),
        "median_difference": float(np.median(differences)),
        "difference_iqr_low": float(np.quantile(differences, 0.25)),
        "difference_iqr_high": float(np.quantile(differences, 0.75)),
        "relative_mean_delta_percent": float(
            100.0 * mean_difference / (abs(baseline_mean) + 1e-12)
        ),
        "win_fraction": float(np.mean(improvements > 0.0)),
        "paired_sign_flip_pvalue_two_sided": paired_sign_flip_pvalue(differences),
    }


def main():
    cli = parse_cli()
    output_dir = cli.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    evaluations = parse_evaluations(cli.evaluation)
    if len(evaluations) < 2:
        raise ValueError("at least one baseline and one candidate are required")
    if cli.baseline not in evaluations:
        raise ValueError("baseline label is missing from --evaluation")

    expected_seeds = SEED_SETS[cli.seed_set]
    loaded = {}
    declared_steps = set()
    matched_config_hashes = set()
    for label, eval_dir in evaluations.items():
        summary, rows = read_rows(eval_dir)
        if sorted(rows) != expected_seeds:
            raise ValueError(
                f"{label} seeds do not exactly match {cli.seed_set}: "
                f"expected {expected_seeds}, got {sorted(rows)}"
            )
        loaded[label] = {"summary": summary, "rows": rows}
        declared_step = summary.get("declared_training_step")
        if declared_step is None:
            raise ValueError(f"{label} has no declared training step")
        declared_steps.add(int(declared_step))
        matched_config_hash = summary.get("matched_config_sha256")
        if not matched_config_hash or "matched_config" not in summary:
            raise ValueError(f"{label} has no matched configuration fingerprint")
        matched_config_hashes.add(matched_config_hash)
    if len(declared_steps) != 1:
        raise ValueError(
            f"evaluations are from different declared training steps: {declared_steps}"
        )
    if len(matched_config_hashes) != 1:
        configs = {
            label: loaded[label]["summary"].get("matched_config")
            for label in evaluations
        }
        raise ValueError(f"matched experiment configurations differ: {configs}")

    output_dir.mkdir(parents=True)
    baseline_rows = loaded[cli.baseline]["rows"]
    comparisons = {}
    detail_rows = []
    for label in evaluations:
        if label == cli.baseline:
            continue
        comparisons[label] = {}
        candidate_rows = loaded[label]["rows"]
        for seed in expected_seeds:
            for agent_id in range(5):
                for axis in ("x", "y"):
                    key = f"uav{agent_id}_initial_{axis}"
                    if key not in baseline_rows[seed] or key not in candidate_rows[seed]:
                        raise ValueError(f"missing fixed-start field {key}")
                    if not np.isclose(
                        float(baseline_rows[seed][key]),
                        float(candidate_rows[seed][key]),
                        atol=1e-8,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            f"fixed start mismatch for seed {seed}, {key}: "
                            f"{baseline_rows[seed][key]} vs {candidate_rows[seed][key]}"
                        )
        for metric in SUMMARY_METRICS:
            baseline_values = [float(baseline_rows[s][metric]) for s in expected_seeds]
            candidate_values = [float(candidate_rows[s][metric]) for s in expected_seeds]
            comparisons[label][metric] = paired_stats(
                baseline_values,
                candidate_values,
                direction=METRIC_DIRECTIONS[metric],
            )
            for seed, baseline_value, candidate_value in zip(
                expected_seeds, baseline_values, candidate_values
            ):
                detail_rows.append({
                    "candidate": label,
                    "baseline": cli.baseline,
                    "seed": seed,
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "difference": candidate_value - baseline_value,
                })

    with (output_dir / "paired_differences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    result = {
        "status": "completed",
        "seed_set": cli.seed_set,
        "seeds": expected_seeds,
        "baseline": cli.baseline,
        "declared_training_step": declared_steps.pop(),
        "primary_metric": "system_performance_true_all_GUs",
        "matched_config_sha256": matched_config_hashes.pop(),
        "difference_definition": "candidate - baseline",
        "direction_definition": "+1 means larger is better; -1 means smaller is better",
        "pairing_caveat": (
            "Pairs share the same initial seed but are not identical exogenous MD traces, "
            "because policy-dependent admissions change later RNG consumption."
        ),
        "sign_flip_role": (
            "robustness diagnostic only; the pre-registered adjacent True60 effect "
            "contrasts and their confidence intervals are primary"
        ),
        "evaluation_dirs": {k: str(v) for k, v in evaluations.items()},
        "method_aggregates": {
            label: loaded[label]["summary"].get("aggregate")
            for label in evaluations
        },
        "comparisons": comparisons,
    }
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
