"""Create a compact three-episode trajectory overview for the latest checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


HERE = Path(__file__).resolve().parent
EPISODES = HERE / "episodes" / "local_heterogeneous_resource"
SEEDS = (1001, 1002, 1003)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for axis, seed in zip(axes, SEEDS):
        episode = EPISODES / f"episode_seed{seed}"
        summary = json.loads((episode / "episode_summary.json").read_text(encoding="utf-8"))
        axis.imshow(mpimg.imread(episode / "uav_trajectory_overview.png"))
        axis.set_title(
            f"test seed {seed}\n"
            f"performance={summary['system_performance_true_all_GUs']:,.0f} | "
            f"final deployment={summary['final_small_region_uavs']} small / "
            f"{summary['final_large_region_uavs']} large",
            fontsize=10,
        )
        axis.axis("off")
    fig.suptitle("Latest heterogeneous-resource checkpoint: deterministic trajectories", fontsize=13)
    output = HERE / "trajectory_grid_3_episodes.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
