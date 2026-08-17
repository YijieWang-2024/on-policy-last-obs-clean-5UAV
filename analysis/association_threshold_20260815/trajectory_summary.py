"""Summarize and grid the three deterministic trajectory evaluations per run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SEEDS = (1001, 1002, 1003)
GROUPS = {
    "remote": [
        ("remote psi=0.3 | deadline ON", "remote_psi0p3"),
        ("remote psi=0.7 | deadline ON", "remote_psi0p7"),
        ("remote psi=0.9 | deadline ON", "remote_psi0p9"),
        ("baseline psi=0.5 | deadline ON", "remote_baseline_psi0p5"),
    ],
    "local": [
        ("local psi=0.3 | deadline OFF", "local_psi0p3"),
        ("local psi=0.5 | deadline OFF", "local_psi0p5"),
        ("local psi=0.7 | deadline OFF", "local_psi0p7"),
    ],
}


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_info(info: dict, key: str) -> float | None:
    value = info.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return float(np.mean(array)) if array.size else None


def collect_rows() -> list[dict]:
    rows = []
    for group, specs in GROUPS.items():
        root = HERE / f"trajectories_{group}"
        for label, tag in specs:
            for seed in SEEDS:
                run_dir = root / f"{tag}_seed{seed}"
                summary = read_summary(run_dir / "episode_summary.json")
                rewards = np.asarray(summary["total_reward_per_uav"], dtype=float)
                info = summary.get("final_environment_info", {})
                rows.append(
                    {
                        "group": group,
                        "label": label,
                        "tag": tag,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "checkpoint_dir": summary["checkpoint_dir"],
                        "training_step_at_snapshot": summary.get("training_step_at_snapshot"),
                        "total_reward": float(rewards.sum()),
                        "mean_uav_reward": float(rewards.mean()),
                        "system_performance_true_all_GUs": scalar_info(info, "system_performance_true_all_GUs"),
                        "complete_task_ratio": scalar_info(info, "complete_task_ratio"),
                        "md_admission_ratio": scalar_info(info, "md_admission_ratio"),
                        "average_active_mds": scalar_info(info, "average_active_mds"),
                        "delay_true_all_GUs": scalar_info(info, "delay_true_all_GUs"),
                        "frame_count": summary["frame_count"],
                        "trajectory_png": str(run_dir / "uav_trajectory_overview.png"),
                        "trajectory_gif": str(run_dir / "trajectory.gif"),
                    }
                )
    return rows


def make_grid(rows: list[dict], group: str, output: Path) -> None:
    selected = [row for row in rows if row["group"] == group]
    labels = []
    for row in selected:
        if row["label"] not in labels:
            labels.append(row["label"])
    fig, axes = plt.subplots(
        len(labels), len(SEEDS), figsize=(12.0, 3.1 * len(labels)), squeeze=False
    )
    for i, label in enumerate(labels):
        for j, seed in enumerate(SEEDS):
            row = next(row for row in selected if row["label"] == label and row["seed"] == seed)
            image = plt.imread(row["trajectory_png"])
            axes[i, j].imshow(image)
            axes[i, j].axis("off")
            axes[i, j].set_title(
                f"{label}\nseed={seed}, reward={row['total_reward']:,.0f}, step={row['training_step_at_snapshot']:,}",
                fontsize=8,
            )
    fig.suptitle(f"Association-threshold trajectory evaluation: {group}", y=0.998, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    for group in GROUPS:
        make_grid(rows, group, HERE / f"trajectory_grid_{group}.png")
    make_grid(rows, "remote", HERE / "trajectory_grid_remote_with_baseline.png")
    # The all-run grid is intentionally compact: one row per run and one column
    # per common seed, so the same episode seed is easy to compare vertically.
    labels = []
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])
    fig, axes = plt.subplots(len(labels), len(SEEDS), figsize=(12.0, 2.55 * len(labels)), squeeze=False)
    for i, label in enumerate(labels):
        for j, seed in enumerate(SEEDS):
            row = next(row for row in rows if row["label"] == label and row["seed"] == seed)
            axes[i, j].imshow(plt.imread(row["trajectory_png"]))
            axes[i, j].axis("off")
            axes[i, j].set_title(f"{label}\nseed={seed}, reward={row['total_reward']:,.0f}", fontsize=7.2)
    fig.suptitle("Association-threshold trajectories: all seven runs", y=0.999, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(HERE / "trajectory_grid_all_seven.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    (HERE / "trajectory_eval_summary.json").write_text(
        json.dumps({"seeds": SEEDS, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (HERE / "trajectory_eval_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for path in sorted(HERE.glob("trajectory_grid*.png")):
        print(path)


if __name__ == "__main__":
    main()
