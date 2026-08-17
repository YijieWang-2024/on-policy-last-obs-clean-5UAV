"""Compose the three deterministic evaluations from the current checkpoint snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE / "trajectories_latest_20505600"
SEEDS = (1001, 1002, 1003)


def main() -> None:
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(15.0, 5.1), squeeze=False)
    for column, seed in enumerate(SEEDS):
        run_dir = EVAL_ROOT / f"episode_seed{seed}"
        summary = json.loads((run_dir / "episode_summary.json").read_text(encoding="utf-8"))
        info = summary["final_environment_info"]
        axis = axes[0, column]
        axis.imshow(plt.imread(run_dir / "uav_trajectory_overview.png"))
        axis.axis("off")
        axis.set_title(
            f"seed={seed}\n"
            f"performance={float(info['system_performance_true_all_GUs'][0]):,.0f}\n"
            f"complete={float(info['complete_task_ratio']):.2%}, "
            f"admission={float(info['md_admission_ratio']):.2%}",
            fontsize=9,
        )
    fig.suptitle(
        "6-UAV current checkpoint (20.5056M): deterministic trajectory evaluation",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output = HERE / "trajectory_grid_latest.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
