"""Evaluate the two live psi=0.1 curriculum checkpoints on three episodes."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "association_threshold_20260815" / "evaluate_trajectories_10_latest.py"


def load_base():
    spec = importlib.util.spec_from_file_location("association_eval_curriculum_two", BASE_SCRIPT)
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
    output_name = os.environ.get("EVAL_OUTPUT_NAME", "trajectory_eval_3_20260816")
    module.OUTPUT = HERE / output_name
    module.SEEDS = [1001, 1002, 1003]
    module.RUNS = {
        "curriculum_psi0p1_seed2": {
            "label": "curriculum ON | psi=0.1 | seed=2 | p0.7->0 by 25M",
            "run_dir": results / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed2_60m_20260816" / "run1",
            "snapshot": True,
        },
        "curriculum_psi0p1_seed32": {
            "label": "curriculum ON | psi=0.1 | seed=32 | p0.7->0 by 25M",
            "run_dir": results / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed32_60m_20260816" / "run1",
            "snapshot": True,
        },
    }
    module.main()


if __name__ == "__main__":
    main()
