"""Evaluate the five current seed-42 psi runs on ten common episodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "association_threshold_20260815" / "evaluate_trajectories_10_latest.py"


def load_base():
    spec = importlib.util.spec_from_file_location("association_eval_base_seed42", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_base()
    results = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
    remote_root = HERE / "remote_snapshots"
    module.HERE = HERE
    module.PROJECT_ROOT = PROJECT_ROOT
    module.RESULTS = results
    module.OUTPUT = HERE / "trajectory_eval_10_latest_seed42"
    module.SEEDS = list(range(1001, 1011))
    module.RUNS = {
        "remote_psi0p1_deadline_off": {
            "label": "remote psi=0.1 | deadline OFF",
            "run_dir": remote_root / "remote_psi0p1" / "checkpoint_snapshot",
            "checkpoint_dir": remote_root / "remote_psi0p1" / "checkpoint_snapshot",
            "snapshot": False,
        },
        "remote_psi0p3_deadline_off": {
            "label": "remote psi=0.3 | deadline OFF",
            "run_dir": remote_root / "remote_psi0p3" / "checkpoint_snapshot",
            "checkpoint_dir": remote_root / "remote_psi0p3" / "checkpoint_snapshot",
            "snapshot": False,
        },
        "remote_psi0p5_deadline_off": {
            "label": "remote psi=0.5 | deadline OFF",
            "run_dir": remote_root / "remote_psi0p5" / "checkpoint_snapshot",
            "checkpoint_dir": remote_root / "remote_psi0p5" / "checkpoint_snapshot",
            "snapshot": False,
        },
        "local_psi0p7_deadline_off": {
            "label": "local psi=0.7 | deadline OFF",
            "run_dir": results / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed42_60m_20260816" / "run1",
            "snapshot": True,
        },
        "local_psi0p9_deadline_off": {
            "label": "local psi=0.9 | deadline OFF",
            "run_dir": results / "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p9_nofilter_seed42_60m_20260816" / "run1",
            "snapshot": True,
        },
    }
    module.main()


if __name__ == "__main__":
    main()
