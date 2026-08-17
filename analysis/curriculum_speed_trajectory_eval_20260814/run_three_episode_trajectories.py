#!/usr/bin/env python
"""Evaluate the latest curriculum speed checkpoints on three matched episodes.

For each speed, the live training directory is snapshotted once.  All three
episodes for that speed then use the same frozen checkpoint, so a later
checkpoint write cannot mix the three trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from onpolicy.scripts.eval.render_dynamic_mappo_episode import (
    checkpoint_manifest_step,
    snapshot_checkpoint,
)

PYTHON = Path(r"C:\Users\wyj2\.conda\envs\marl\python.exe")
RENDER = PROJECT / "onpolicy" / "scripts" / "eval" / "render_dynamic_mappo_episode.py"

RESULTS_ROOT = PROJECT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
EXPERIMENT_PREFIX = (
    "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
    "md12_uavcurr_p0p7_10m_25m_vmax"
)

SPECS = {
    "vmax10": RESULTS_ROOT / f"{EXPERIMENT_PREFIX}10_seed2_60m_20260813" / "run1",
    "vmax20": RESULTS_ROOT / f"{EXPERIMENT_PREFIX}20_seed2_60m_20260813" / "run1",
    # run1 is the old 0.6656M-step attempt; run2 is the current training run.
    "vmax30": RESULTS_ROOT / f"{EXPERIMENT_PREFIX}30_seed2_60m_20260813" / "run2",
}

EVAL_SEEDS = (20260814, 20260815, 20260816)
UAV_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")


def scalar(value: object) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else float("nan")
    return float(value)


def load_positions(path: Path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    timesteps = sorted({int(row["timestep"]) for row in rows})
    positions = []
    for timestep in timesteps:
        frame = sorted(
            (row for row in rows if int(row["timestep"]) == timestep),
            key=lambda row: int(row["uav"]),
        )
        positions.append([(float(row["x"]), float(row["y"])) for row in frame])
    return positions


def plot_trajectory(path: Path, title: str, positions, performance: float, completion: float):
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.set(xlim=(0, 600), ylim=(0, 600), xlabel="X position (m)", ylabel="Y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.add_patch(Rectangle((0, 0), 200, 200, fill=False, edgecolor="#E69F00", linestyle="--", linewidth=1.8, label="Small region 200×200 m"))
    ax.add_patch(Rectangle((200, 200), 400, 400, fill=False, edgecolor="#009E73", linestyle="-.", linewidth=1.8, label="Large region 400×400 m"))
    for uav_id in range(len(positions[0])):
        path_i = positions
        x = [frame[uav_id][0] for frame in path_i]
        y = [frame[uav_id][1] for frame in path_i]
        color = UAV_COLORS[uav_id]
        ax.plot(x, y, color=color, linewidth=1.8, label=f"UAV {uav_id + 1}")
        ax.scatter(x[0], y[0], color=color, marker="o", s=38, zorder=5)
        ax.scatter(x[-1], y[-1], color=color, marker="X", s=125, zorder=5)
    ax.set_title(f"{title}\ntrue_all_GUs={performance / 1000:.2f}k | completion={completion:.3f}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_gif(case_dir: Path) -> Path:
    frames = sorted((case_dir / "frames").glob("slot_*.png"))
    if not frames:
        raise FileNotFoundError(f"No rendered frames in {case_dir / 'frames'}")
    images = [Image.open(frame).convert("RGB") for frame in frames]
    gif_path = case_dir / "episode.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=160, loop=0, optimize=False)
    for image in images:
        image.close()
    return gif_path


def make_grid(records, output_path: Path, columns: int, title: str) -> None:
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(7.0 * columns, 7.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        image = Image.open(record["trajectory_plot"]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{record['label']} | seed={record['evaluation_seed']} | "
            f"true={record['true_all_GUs'] / 1000:.2f}k | "
            f"completion={record['complete_task_ratio']:.3f}",
            fontsize=10,
        )
        image.close()
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def snapshot_as_frozen_run(source_run: Path, group_dir: Path):
    snapshot_root = group_dir / "snapshot_work"
    checkpoint_dir, files = snapshot_checkpoint(source_run.resolve(), snapshot_root, allow_frozen_legacy=False)
    step = checkpoint_manifest_step(checkpoint_dir)
    if step is None:
        raise RuntimeError(f"Checkpoint manifest is unavailable for {source_run}")
    frozen_run = group_dir / "frozen_run"
    models_dir = frozen_run / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for path in models_dir.iterdir():
        if path.is_file():
            path.unlink()
    for path in checkpoint_dir.iterdir():
        if path.is_file() and path.name != "args.json":
            shutil.copy2(path, models_dir / path.name)
    shutil.copy2(checkpoint_dir / "args.json", frozen_run / "args.json")
    return frozen_run, int(step), files


def run_one_episode(frozen_run: Path, output_dir: Path, seed: int, step: int) -> None:
    command = [
        str(PYTHON), str(RENDER),
        "--run-dir", str(frozen_run),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--training-step", str(step),
        "--frame-interval", "10",
        "--confirm-checkpoint-frozen",
    ]
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    (output_dir.parent / f"{output_dir.name}.eval.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"Trajectory rendering failed for {output_dir}:\n{completed.stderr[-3000:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    records = []
    checkpoint_catalog = {}
    for label, source_run in SPECS.items():
        if not source_run.is_dir():
            raise FileNotFoundError(f"Expected run directory does not exist: {source_run}")
        group_dir = output_root / label
        group_dir.mkdir()
        frozen_run, step, checkpoint_files = snapshot_as_frozen_run(source_run, group_dir)
        checkpoint_catalog[label] = {
            "source_run": str(source_run),
            "frozen_run": str(frozen_run),
            "training_step": step,
            "checkpoint_files": checkpoint_files,
        }
        for episode_index, seed in enumerate(EVAL_SEEDS, start=1):
            episode_dir = group_dir / f"episode{episode_index}_seed{seed}"
            print(f"RENDER {label} episode={episode_index} seed={seed} step={step}", flush=True)
            run_one_episode(frozen_run, episode_dir, seed, step)
            gif_path = make_gif(episode_dir)
            summary = json.loads((episode_dir / "episode_summary.json").read_text(encoding="utf-8"))
            info = summary["final_environment_info"]
            positions = load_positions(episode_dir / "uav_trajectory.csv")
            trajectory_plot = episode_dir / "uav_trajectory_colorblind.png"
            plot_trajectory(
                trajectory_plot,
                f"{label} | episode {episode_index} | checkpoint {step / 1e6:.3f}M",
                positions,
                scalar(info["system_performance_true_all_GUs"]),
                scalar(info["complete_task_ratio"]),
            )
            records.append({
                "label": label,
                "episode_index": episode_index,
                "evaluation_seed": seed,
                "training_step": step,
                "true_all_GUs": scalar(info["system_performance_true_all_GUs"]),
                "complete_task_ratio": scalar(info["complete_task_ratio"]),
                "md_admission_ratio": scalar(info["md_admission_ratio"]),
                "trajectory_plot": str(trajectory_plot),
                "episode_gif": str(gif_path),
            })

        group_records = [record for record in records if record["label"] == label]
        make_grid(
            group_records,
            group_dir / "three_episode_trajectory_comparison.png",
            columns=3,
            title=f"{label}: three matched deterministic evaluation episodes",
        )

    with (output_root / "trajectory_catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    with (output_root / "checkpoint_catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(checkpoint_catalog, handle, ensure_ascii=False, indent=2)
    with (output_root / "trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    make_grid(
        records,
        output_root / "all_speeds_three_episodes_trajectory_grid.png",
        columns=3,
        title="Curriculum speed comparison: three trajectories per speed",
    )
    print(json.dumps({"output_root": str(output_root), "checkpoints": checkpoint_catalog}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
