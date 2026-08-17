#!/usr/bin/env python
"""Evaluate the three live continuation runs from coherent latest snapshots.

The training processes may still be writing checkpoints.  We therefore use the
same manifest/size/mtime consistency check as the renderer, freeze one snapshot
per speed, and evaluate four deterministic episodes from that frozen copy.
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
from PIL import Image

from onpolicy.scripts.eval.render_dynamic_mappo_episode import (
    checkpoint_manifest_step,
    snapshot_checkpoint,
)


PYTHON = Path(r"C:\Users\wyj2\.conda\envs\marl\python.exe")
RENDER = PROJECT / "onpolicy" / "scripts" / "eval" / "render_dynamic_mappo_episode.py"
RESULTS_ROOT = PROJECT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"

SPECS = {
    "vmax20_seed2": RESULTS_ROOT / (
        "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
        "md12_vmax20_seed2_resume40m_20260814/run1"
    ),
    "vmax30_seed32": RESULTS_ROOT / (
        "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
        "md12_vmax30_seed32_resume40m_20260814/run1"
    ),
    "vmax40_seed32": RESULTS_ROOT / (
        "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_"
        "md12_vmax40_seed32_resume40m_20260814/run1"
    ),
}

EVAL_SEEDS = (1001, 1002, 1003, 1004)
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


def plot_episode(path: Path, title: str, positions, performance: float, completion: float):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set(xlim=(0, 600), ylim=(0, 600), xlabel="X position (m)", ylabel="Y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)
    for uav_id in range(len(positions[0])):
        x = [frame[uav_id][0] for frame in positions]
        y = [frame[uav_id][1] for frame in positions]
        color = UAV_COLORS[uav_id]
        ax.plot(x, y, color=color, linewidth=1.45, label=f"UAV {uav_id + 1}")
        ax.scatter(x[0], y[0], color=color, marker="o", s=24, zorder=5)
        ax.scatter(x[-1], y[-1], color=color, marker="X", s=64, zorder=5)
    ax.set_title(
        f"{title}\ntrue-all-GUs={performance / 1000:.2f}k | completion={completion:.3f}",
        fontsize=9,
        pad=8,
    )
    ax.legend(loc="upper right", fontsize=6.8, framealpha=0.92, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_grid(records, output_path: Path, title: str, columns: int = 2):
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 5.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        image = Image.open(record["trajectory_plot"]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"episode {record['episode_index']} | eval seed {record['evaluation_seed']} | "
            f"{record['true_all_GUs'] / 1000:.2f}k, completion={record['complete_task_ratio']:.3f}",
            fontsize=9,
        )
        image.close()
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def snapshot_as_frozen_run(source_run: Path, group_dir: Path):
    snapshot_root = group_dir / "snapshot_work"
    checkpoint_dir, files = snapshot_checkpoint(
        source_run.resolve(), snapshot_root, allow_frozen_legacy=False
    )
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
    manifest = json.loads((checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    return frozen_run, int(step), files, manifest


def run_one_episode(frozen_run: Path, output_dir: Path, seed: int, step: int) -> None:
    command = [
        str(PYTHON),
        str(RENDER),
        "--run-dir",
        str(frozen_run),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--training-step",
        str(step),
        "--frame-interval",
        "10",
        "--confirm-checkpoint-frozen",
    ]
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    (output_dir.parent / f"{output_dir.name}.eval.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"Trajectory rendering failed for {output_dir}:\n{completed.stderr[-4000:]}"
        )


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
        frozen_run, step, checkpoint_files, manifest = snapshot_as_frozen_run(
            source_run, group_dir
        )
        checkpoint_catalog[label] = {
            "source_run": str(source_run),
            "frozen_run": str(frozen_run),
            "training_step": step,
            "manifest": manifest,
            "checkpoint_files": checkpoint_files,
        }
        for episode_index, seed in enumerate(EVAL_SEEDS, start=1):
            episode_dir = group_dir / f"episode{episode_index}_seed{seed}"
            print(f"RENDER {label} episode={episode_index} seed={seed} step={step}", flush=True)
            run_one_episode(frozen_run, episode_dir, seed, step)
            summary = json.loads((episode_dir / "episode_summary.json").read_text(encoding="utf-8"))
            info = summary["final_environment_info"]
            positions = load_positions(episode_dir / "uav_trajectory.csv")
            trajectory_plot = episode_dir / "uav_trajectory_colorblind.png"
            plot_episode(
                trajectory_plot,
                f"{label} | episode {episode_index} | checkpoint {step / 1e6:.3f}M",
                positions,
                scalar(info["system_performance_true_all_GUs"]),
                scalar(info["complete_task_ratio"]),
            )
            records.append(
                {
                    "label": label,
                    "episode_index": episode_index,
                    "evaluation_seed": seed,
                    "training_step": step,
                    "true_all_GUs": scalar(info["system_performance_true_all_GUs"]),
                    "complete_task_ratio": scalar(info["complete_task_ratio"]),
                    "md_admission_ratio": scalar(info["md_admission_ratio"]),
                    "trajectory_plot": str(trajectory_plot),
                    "episode_data": str(episode_dir / "episode_data.npz"),
                    "episode_summary": str(episode_dir / "episode_summary.json"),
                }
            )
        group_records = [record for record in records if record["label"] == label]
        make_grid(group_records, group_dir / "four_episode_trajectory_grid.png", label)

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
        output_root / "all_speeds_four_episode_trajectory_grid.png",
        "Latest continuation checkpoints: four deterministic episodes per speed",
        columns=4,
    )
    print(json.dumps({"output_root": str(output_root), "checkpoints": checkpoint_catalog}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
