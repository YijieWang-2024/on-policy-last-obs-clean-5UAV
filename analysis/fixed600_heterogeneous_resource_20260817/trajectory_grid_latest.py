"""Compose the three deterministic trajectories from the latest checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE / "trajectory_eval_3_latest_20260817_refresh3" / "episodes" / "local_heterogeneous_resource"
SEEDS = (1001, 1002, 1003)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), squeeze=False)
    for axis, seed in zip(axes[0], SEEDS):
        run_dir = EVAL_ROOT / f"episode_seed{seed}"
        summary = json.loads(
            (run_dir / "episode_summary.json").read_text(encoding="utf-8")
        )
        info = summary
        axis.imshow(plt.imread(run_dir / "uav_trajectory_overview.png"))
        axis.axis("off")
        axis.set_title(
            f"seed={seed}\n"
            f"performance={float(info['system_performance_true_all_GUs']):,.0f}\n"
            f"complete={float(info['complete_task_ratio']):.2%}, "
            f"admission={float(info['md_admission_ratio']):.2%}",
            fontsize=9,
        )
    fig.suptitle(
        "Heterogeneous-resource checkpoint: three deterministic episodes",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output = HERE / "trajectory_grid_latest.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
