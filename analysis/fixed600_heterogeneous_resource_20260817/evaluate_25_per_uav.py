"""Compare the running heterogeneous checkpoint with the remote final baseline.

This is an analysis-only evaluator.  It freezes the running local run at the
latest manifest-verified checkpoint, then evaluates both models on the same 25
deterministic episode seeds and writes overall and UAV-wise CSV/JSON results.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT / "analysis" / "psi_reviewer_final_metrics_20260816" / "evaluate_final_checkpoints.py"
HELPER_SCRIPT = PROJECT / "analysis" / "psi0p5_per_uav_30det_20260817" / "evaluate_psi0p5_per_uav.py"
RESULTS = PROJECT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
OUTPUT = Path(__file__).resolve().parent / "eval25_per_uav_20260817_refresh3"
SEEDS = list(range(7001, 7026))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_per_uav(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    result = []
    for model in sorted({row["model"] for row in rows}):
        for uav_id in range(5):
            selected = [
                row for row in rows
                if row["model"] == model and int(row["uav_id"]) == uav_id
            ]
            item = {
                "model": model,
                "uav_id": uav_id,
                "episodes": len(selected),
                "checkpoint_step": selected[0]["checkpoint_step"],
            }
            for field in fields:
                values = np.asarray([float(row[field]) for row in selected])
                values = values[np.isfinite(values)]
                item[f"{field}_mean"] = float(values.mean()) if values.size else float("nan")
                item[f"{field}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            result.append(item)
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    base = load_module(BASE_SCRIPT, "eval_base_25_per_uav")
    helper = load_module(HELPER_SCRIPT, "per_uav_helper_25")
    hetero_run = RESULTS / "dcppoR520_fixed600_200_het1_13_07_13_07_psi05_nf_cur_s2_60m_20260817" / "run2"
    remote_root = Path(__file__).resolve().parent / "remote_final_snapshot_59980800"
    remote_models = remote_root / "models"

    local_checkpoint, local_files = base.snapshot_checkpoint(
        hetero_run, OUTPUT / "snapshots" / "local_heterogeneous"
    )
    runs = {
        "heterogeneous_local_latest": {
            "label": "heterogeneous local | [1.0,1.3,0.7,1.3,0.7]",
            "run_dir": hetero_run,
            "checkpoint_dir": local_checkpoint,
            "checkpoint_files": local_files,
        },
        "homogeneous_remote_final": {
            "label": "homogeneous remote final | default equal resources",
            "run_dir": remote_root,
            "checkpoint_dir": remote_models,
            "checkpoint_files": {
                path.name: {"bytes": path.stat().st_size, "mtime": path.stat().st_mtime}
                for path in remote_models.iterdir() if path.is_file()
            },
        },
    }

    torch.set_num_threads(1)
    episode_rows = []
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "episodes": len(SEEDS),
        "test_seeds": SEEDS,
        "deterministic": True,
        "metric_semantics": {
            "service_md_per_episode": "active-MD associations assigned to one UAV over 400 steps",
            "on_time_service_md_per_episode": "those assignments whose post-step complete_task flag is true",
            "service_success_rate": "successful service MD / served service MD",
            "service_step_fraction": "steps with at least one active MD served / 400",
            "episode_reward": "sum of that UAV's env.step reward over 400 steps",
            "mean_step_reward": "episode_reward / 400",
        },
        "runs": {},
    }
    checkpoint_to_model = {}

    for model, run in runs.items():
        checkpoint_step = base.checkpoint_manifest_step(run["checkpoint_dir"])
        if checkpoint_step is None:
            raise RuntimeError(f"No verified checkpoint: {run['checkpoint_dir']}")
        checkpoint_to_model[int(checkpoint_step)] = model
        args, env, actors, normers = base.load_policy(
            Path(run["run_dir"]), Path(run["checkpoint_dir"])
        )
        metadata["runs"][model] = {
            "label": run["label"],
            "run_dir": str(run["run_dir"]),
            "checkpoint_dir": str(run["checkpoint_dir"]),
            "checkpoint_step": int(checkpoint_step),
            "n_uavs": int(args.n_UAVs),
            "episode_length": int(args.episode_length),
            "association_threshold": float(args.association_threshold),
            "offload_deadline_filter": bool(args.offload_deadline_filter),
            "uav_resource_mode": getattr(args, "uav_resource_mode", None),
            "uav_resource_scale_factors": getattr(args, "uav_resource_scale_factors", None),
            "checkpoint_files": run["checkpoint_files"],
        }
        try:
            for index, seed in enumerate(SEEDS, 1):
                row = base.evaluate_episode(
                    args, env, actors, normers, seed, checkpoint_step, deterministic=True
                )
                row["model"] = model
                episode_rows.append(row)
                if index == 1 or index % 5 == 0 or index == len(SEEDS):
                    print(json.dumps({"model": model, "episode": index, "episodes": len(SEEDS)}), flush=True)
        finally:
            env.close()

    overall = base.aggregate(episode_rows)
    per_uav_rows = helper.make_per_uav_rows(episode_rows, 5)
    for row in per_uav_rows:
        row["model"] = checkpoint_to_model[int(row["checkpoint_step"])]
    per_uav_fields = helper.PER_UAV_FIELDS
    per_uav_aggregate = aggregate_per_uav(per_uav_rows, per_uav_fields)

    write_csv(OUTPUT / "episode_metrics.csv", episode_rows)
    write_csv(OUTPUT / "overall_aggregate.csv", overall)
    write_csv(OUTPUT / "per_uav_episode_metrics.csv", per_uav_rows)
    write_csv(OUTPUT / "per_uav_aggregate.csv", per_uav_aggregate)
    metadata["overall_aggregate"] = overall
    metadata["per_uav_aggregate"] = per_uav_aggregate
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(OUTPUT),
        "episodes_per_model": len(SEEDS),
        "checkpoints": {
            model: info["checkpoint_step"]
            for model, info in metadata["runs"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
