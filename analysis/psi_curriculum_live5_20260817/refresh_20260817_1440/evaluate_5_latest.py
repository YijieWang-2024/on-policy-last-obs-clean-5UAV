"""Evaluate five current 5-UAV runs on three common deterministic episodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "association_threshold_20260815" / "evaluate_trajectories_10_latest.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("association_eval_refresh_20260817", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
    module.HERE = HERE
    module.PROJECT_ROOT = PROJECT_ROOT
    module.RESULTS = results
    module.SEEDS = [1001, 1002, 1003]
    module.OUTPUT = HERE / "trajectory_eval_3_latest"
    module.RUNS = {
        "local_psi0p1_seed2": {
            "label": "local | curriculum ON | deadline OFF | psi=0.1 | seed=2",
            "run_dir": results / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
            "snapshot": True,
        },
        "local_psi0p3_seed2": {
            "label": "local | curriculum ON | deadline OFF | psi=0.3 | seed=2",
            "run_dir": results / "dcppoR520_fixed600_200_noactor_psi0p3_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
            "snapshot": True,
        },
    }
    for psi in ("0p5", "0p7", "0p9"):
        checkpoint = HERE / "remote_checkpoints" / f"psi{psi}" / "checkpoint_snapshot"
        module.RUNS[f"remote_psi{psi}_seed2"] = {
            "label": f"remote | curriculum ON | deadline OFF | psi={psi.replace('p', '.')} | seed=2",
            "run_dir": checkpoint,
            "checkpoint_dir": checkpoint,
            "snapshot": False,
        }
    module.main()
    legacy = module.OUTPUT / "trajectory_grid_best_of_7.png"
    corrected = module.OUTPUT / "trajectory_grid_best_of_5.png"
    if legacy.exists() and not corrected.exists():
        legacy.replace(corrected)


if __name__ == "__main__":
    main()
