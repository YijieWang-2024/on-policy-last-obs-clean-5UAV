"""Evaluate the six live curriculum runs on three common deterministic episodes.

The three local runs are snapshotted once by the base evaluator.  The remote
runs use the non-destructive frozen copy under ``_remote_curr``.  Re-running
the script with a new ``EVAL_OUTPUT_NAME`` creates a fresh result directory.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "association_threshold_20260815" / "evaluate_trajectories_10_latest.py"
REMOTE_SNAPSHOT = PROJECT_ROOT / "_remote_curr" / "snapshot_20260816_231933"


def load_base():
    spec = importlib.util.spec_from_file_location("association_eval_curriculum_six", BASE_SCRIPT)
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
    module.OUTPUT = HERE / os.environ.get("EVAL_OUTPUT_NAME", "trajectory_eval_3_latest_20260816_v2")
    module.SEEDS = [1001, 1002, 1003]

    remote_specs = {}
    for psi in ("0p5", "0p7", "0p9"):
        name = f"dcppoR520_fixed600_200_noactor_psi{psi}_nofilter_curr_p07_10m25m_seed2_60m_20260816"
        remote_specs[f"remote_psi{psi}_seed2"] = {
            "label": f"remote | curriculum ON | deadline OFF | psi={psi.replace('p', '.')} | seed=2",
            "run_dir": REMOTE_SNAPSHOT / name / "run1",
            "checkpoint_dir": REMOTE_SNAPSHOT / name / "run1" / "models",
            "snapshot": False,
        }

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
        "local_psi0p1_seed32": {
            "label": "local | curriculum ON | deadline OFF | psi=0.1 | seed=32",
            "run_dir": results / "dcppoR520_fixed600_200_noactor_psi0p1_nofilter_curr_p07_10m25m_seed32_60m_20260816" / "run1",
            "snapshot": True,
        },
        **remote_specs,
    }
    module.main()


if __name__ == "__main__":
    main()
