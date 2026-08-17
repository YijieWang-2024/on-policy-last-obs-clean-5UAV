"""Plot the six requested association-threshold training curves."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_SCRIPT = PROJECT_ROOT / "analysis" / "association_threshold_seed42_latest_20260816" / "plot.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("association_plot_base_six", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotter: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.PROJECT = PROJECT_ROOT
    module.RESULTS = PROJECT_ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
    module.CURVES = [
        ("remote psi=0.1 | seed=42 | deadline OFF", "psi0p1_r42", "#0072B2", "-", HERE / "data" / "remote" / "remote_psi0p1.tfevents"),
        ("remote psi=0.3 | seed=42 | deadline OFF", "psi0p3_r42", "#E69F00", "-", HERE / "data" / "remote" / "remote_psi0p3.tfevents"),
        ("remote psi=0.5 | seed=42 | deadline OFF", "psi0p5_r42", "#D55E00", "-", HERE / "data" / "remote" / "remote_psi0p5.tfevents"),
        ("local psi=0.7 | seed=42 | deadline OFF", "psi0p7_l42", "#009E73", "--", None),
        ("local psi=0.9 | seed=42 | deadline OFF", "psi0p9_l42", "#CC79A7", "--", None),
        ("local psi=0.1 | seed=2 | deadline OFF", "psi0p1_l2", "#56B4E9", ":", None),
    ]
    module.LOCAL_NAMES = {
        "psi0p7_l42": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p7_nofilter_seed42_60m_20260816",
        "psi0p9_l42": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p9_nofilter_seed42_60m_20260816",
        "psi0p1_l2": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p1_nofilter_seed2_60m_20260816",
    }
    module.main()


if __name__ == "__main__":
    main()
