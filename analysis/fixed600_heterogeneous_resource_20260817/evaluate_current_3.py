"""Evaluate the latest local heterogeneous checkpoint on three episodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = (
    PROJECT_ROOT
    / "analysis"
    / "association_threshold_20260815"
    / "evaluate_trajectories_10_latest.py"
)
TARGET_EXPERIMENT = (
    "dcppoR520_fixed600_200_het1_13_07_13_07_psi05_nf_cur_s2_60m_20260817"
)


def load_base():
    spec = importlib.util.spec_from_file_location(
        "heterogeneous_resource_eval_base", BASE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_base()
    results = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
    module.HERE = HERE
    module.PROJECT_ROOT = PROJECT_ROOT
    module.RESULTS = results
    module.OUTPUT = HERE / "trajectory_eval_3_latest_20260817_refresh3"
    module.SEEDS = [1001, 1002, 1003]
    module.N_UAVS = 5
    module.RUNS = {
        "local_heterogeneous_resource": {
            "label": (
                "local heterogeneous resources [1.0,1.3,0.7,1.3,0.7] "
                "| psi=0.5 | curriculum | deadline OFF"
            ),
            "run_dir": results / TARGET_EXPERIMENT / "run2",
            "snapshot": True,
        }
    }
    module.main()


if __name__ == "__main__":
    main()
