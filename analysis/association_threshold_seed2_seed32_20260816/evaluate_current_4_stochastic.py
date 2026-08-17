"""Evaluate the four association-threshold runs with stochastic actor actions.

This reuses the previous ten-episode evaluator, but wraps every actor so the
policy is sampled with ``deterministic=False`` even though the base evaluator
was originally written for deterministic inference.  The four checkpoints,
test seeds, environment and normalization snapshots are otherwise unchanged.
"""

from __future__ import annotations

import importlib.util
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = (
    PROJECT_ROOT
    / "analysis"
    / "association_threshold_20260815"
    / "evaluate_trajectories_10_latest.py"
)


def load_base():
    spec = importlib.util.spec_from_file_location(
        "association_eval_base_stochastic", BASE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StochasticActor:
    """Force sampled actions while preserving the base actor interface."""

    def __init__(self, actor):
        self.actor = actor

    def __call__(self, *args, **kwargs):
        kwargs["deterministic"] = False
        return self.actor(*args, **kwargs)


def make_stochastic_best_grid(output: Path) -> None:
    """Create a correctly labelled four-model best-episode grid."""
    rows = []
    with (output / "episode_metrics.csv").open("r", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    best_rows = {}
    for row in rows:
        model = row["model"]
        if model not in best_rows or float(row["system_performance_true_all_GUs"]) > float(
            best_rows[model]["system_performance_true_all_GUs"]
        ):
            best_rows[model] = row

    ordered = [best_rows[key] for key in sorted(best_rows)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    for axis, row in zip(axes.flat, ordered):
        image = plt.imread(
            output
            / "episodes"
            / row["model"]
            / f"episode_seed{row['evaluation_seed']}"
            / "uav_trajectory_overview.png"
        )
        axis.imshow(image)
        axis.set_title(
            f"{row['label']}\nseed {row['evaluation_seed']} | "
            f"perf={float(row['system_performance_true_all_GUs']):,.0f}",
            fontsize=8,
        )
        axis.axis("off")
    fig.suptitle("Best stochastic episode trajectory per model", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "trajectory_grid_best_of_4.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    module = load_base()
    results = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
    remote_run = HERE / "remote_checkpoint" / "run1"
    module.HERE = HERE
    module.PROJECT_ROOT = PROJECT_ROOT
    module.RESULTS = results
    module.OUTPUT = HERE / "trajectory_eval_10_latest_stochastic_20260816"
    module.SEEDS = list(range(1001, 1011))
    module.RUNS = {
        "local_psi0p5_seed2_deadline_off": {
            "label": "local psi=0.5 | seed 2 | deadline OFF | stochastic",
            "run_dir": results
            / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed2_60m_20260814"
            / "run1",
            "snapshot": True,
        },
        "local_psi0p7_seed2_deadline_off": {
            "label": "local psi=0.7 | seed 2 | deadline OFF | stochastic",
            "run_dir": results
            / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed2_60m_20260814"
            / "run1",
            "snapshot": True,
        },
        "local_psi0p9_seed32_deadline_off": {
            "label": "local psi=0.9 | seed 32 | deadline OFF | stochastic",
            "run_dir": results
            / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p9_nofilter_seed32_60m_20260815"
            / "run1",
            "snapshot": True,
        },
        "remote_psi0p3_seed32_deadline_off": {
            "label": "remote psi=0.3 | seed 32 | deadline OFF | stochastic",
            "run_dir": remote_run,
            "checkpoint_dir": remote_run / "models",
            "snapshot": False,
        },
    }

    base_evaluate_episode = module.evaluate_episode

    def evaluate_episode_stochastic(
        args, env, actors, normers, seed: int, checkpoint_step: int
    ):
        wrapped_actors = [StochasticActor(actor) for actor in actors]
        row, arrays, region_bounds = base_evaluate_episode(
            args, env, wrapped_actors, normers, seed, checkpoint_step
        )
        row["action_mode"] = "stochastic"
        return row, arrays, region_bounds

    module.evaluate_episode = evaluate_episode_stochastic
    module.main()

    output = module.OUTPUT
    make_stochastic_best_grid(output)
    old_grid = output / "trajectory_grid_best_of_7.png"
    new_grid = output / "trajectory_grid_best_of_4.png"
    if old_grid.exists():
        shutil.copy2(old_grid, new_grid)

    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["action_mode"] = "stochastic"
        metadata["description"] = (
            "Ten common evaluation seeds (1001-1010), sampled actor actions, "
            "same four checkpoints and frozen observation-normalization files."
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
