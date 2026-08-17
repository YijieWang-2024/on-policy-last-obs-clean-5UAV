"""Evaluate the five requested runs on three common deterministic episodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "association_threshold_20260815" / "evaluate_trajectories_10_latest.py"


def load_base():
    spec = importlib.util.spec_from_file_location("association_eval_live5_20260817", BASE_SCRIPT)
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
    module.OUTPUT = HERE / "trajectory_eval_3_latest_20260817_final2"
    module.SEEDS = [1001, 1002, 1003]
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
    remote_root = PROJECT_ROOT / "analysis" / "psi_curriculum_seed2_seed32_vs_ref_20260817" / "remote_snapshots"
    for psi in ("0p5", "0p7", "0p9"):
        module.RUNS[f"remote_psi{psi}_seed2"] = {
            "label": f"remote | curriculum ON | deadline OFF | psi={psi.replace('p', '.')} | seed=2",
            "run_dir": remote_root / f"remote_psi{psi}" / "checkpoint_snapshot",
            "checkpoint_dir": remote_root / f"remote_psi{psi}" / "checkpoint_snapshot" / "models",
            "snapshot": False,
        }
    module.main()

    # The imported evaluator historically used a seven-model filename.  Keep
    # the generated artifact but give this five-model result an accurate name.
    old = module.OUTPUT / "trajectory_grid_best_of_7.png"
    new = module.OUTPUT / "trajectory_grid_best_of_5.png"
    if old.exists() and not new.exists():
        old.replace(new)


if __name__ == "__main__":
    main()
