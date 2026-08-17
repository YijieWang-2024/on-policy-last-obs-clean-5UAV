"""Evaluate the four explicitly named runs on ten common deterministic episodes."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = (
    PROJECT_ROOT
    / "analysis"
    / "association_threshold_20260815"
    / "evaluate_trajectories_10_latest.py"
)


def load_base():
    spec = importlib.util.spec_from_file_location("association_eval_base_4", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_base()
    results = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
    remote_run = HERE / "remote_checkpoint" / "run1"
    module.HERE = HERE
    module.PROJECT_ROOT = PROJECT_ROOT
    module.RESULTS = results
    module.OUTPUT = HERE / "trajectory_eval_10_latest_20260816"
    module.SEEDS = list(range(1001, 1011))
    module.RUNS = {
        "local_psi0p5_seed2_deadline_off": {
            "label": "local psi=0.5 | seed 2 | deadline OFF",
            "run_dir": results
            / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed2_60m_20260814"
            / "run1",
            "snapshot": True,
        },
        "local_psi0p7_seed2_deadline_off": {
            "label": "local psi=0.7 | seed 2 | deadline OFF",
            "run_dir": results
            / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed2_60m_20260814"
            / "run1",
            "snapshot": True,
        },
        "local_psi0p9_seed32_deadline_off": {
            "label": "local psi=0.9 | seed 32 | deadline OFF",
            "run_dir": results
            / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p9_nofilter_seed32_60m_20260815"
            / "run1",
            "snapshot": True,
        },
        "remote_psi0p3_seed32_deadline_off": {
            "label": "remote psi=0.3 | seed 32 | deadline OFF",
            "run_dir": remote_run,
            "checkpoint_dir": remote_run / "models",
            "snapshot": False,
        },
    }
    module.main()
    old_grid = module.OUTPUT / "trajectory_grid_best_of_7.png"
    new_grid = module.OUTPUT / "trajectory_grid_best_of_4.png"
    if old_grid.exists():
        shutil.copy2(old_grid, new_grid)


if __name__ == "__main__":
    main()
