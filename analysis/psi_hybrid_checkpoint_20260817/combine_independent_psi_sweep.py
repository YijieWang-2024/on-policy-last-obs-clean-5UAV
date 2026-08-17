"""Combine the five independent fixed-flight hybrid evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE_EVALUATOR = (
    PROJECT_ROOT
    / "analysis"
    / "psi_reviewer_final_metrics_20260816"
    / "evaluate_final_checkpoints.py"
)
SUMMARY_SCRIPT = (
    PROJECT_ROOT
    / "analysis"
    / "psi_curriculum_seed2_30det_latest_20260817"
    / "summarize_pooled.py"
)


def load_base():
    spec = importlib.util.spec_from_file_location("base_for_psi_sweep_combine", BASE_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-file", type=Path, default=None)
    parser.add_argument("--seed-start", type=int, default=8001)
    return parser.parse_args()


def main():
    cli = parse_args()
    if not cli.input_root.is_dir():
        raise FileNotFoundError(cli.input_root)
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)

    branch_dirs = [
        cli.input_root / "psi0p1",
        cli.input_root / "psi0p3",
        cli.input_root / "psi0p5",
        cli.input_root / "psi0p7",
        cli.input_root / "psi0p9",
    ]
    rows = []
    branch_metadata = []
    fieldnames = None
    for branch_dir in branch_dirs:
        input_csv = branch_dir / "episode_metrics.csv"
        if not input_csv.is_file():
            raise FileNotFoundError(input_csv)
        with input_csv.open(newline="", encoding="utf-8-sig") as handle:
            branch_rows = list(csv.DictReader(handle))
        if not branch_rows:
            raise ValueError(f"No rows in {input_csv}")
        # evaluate_hybrid.py predates the explicit eval_seed column; its rows
        # are written in seed order, so restore the common seed identity here.
        for index, row in enumerate(branch_rows):
            row["eval_seed"] = int(row.get("eval_seed", cli.seed_start + index))
        if fieldnames is None:
            fieldnames = list(branch_rows[0])
        elif list(branch_rows[0]) != fieldnames:
            raise ValueError(f"CSV fields differ in {input_csv}")
        rows.extend(branch_rows)
        metadata_path = branch_dir / "metadata.json"
        if metadata_path.is_file():
            branch_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    seeds_by_model = {}
    for row in rows:
        seeds_by_model.setdefault(row["model"], []).append(int(row["eval_seed"]))
    if len(set(tuple(sorted(seeds)) for seeds in seeds_by_model.values())) != 1:
        raise ValueError(f"Evaluation seed sets are not identical: {seeds_by_model}")

    cli.output_dir.mkdir(parents=True)
    base = load_base()
    base.write_csv(cli.output_dir / "episode_metrics.csv", rows)
    base.write_csv(cli.output_dir / "aggregate_metrics.csv", base.aggregate(rows))
    if cli.audit_file is not None:
        audit = json.loads(cli.audit_file.read_text(encoding="utf-8"))
    else:
        audit = None
    (cli.output_dir / "configuration_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cli.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "input_root": str(cli.input_root.resolve()),
                "branch_directories": [str(path.resolve()) for path in branch_dirs],
                "models": sorted(seeds_by_model),
                "seeds_by_model": seeds_by_model,
                "episodes_per_model": len(next(iter(seeds_by_model.values()))),
                "branch_metadata": branch_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (cli.output_dir / "protocol.md").write_text(
        """# Combined fixed-flight psi sweep\n\n
The five independent evaluation directories were concatenated only at the
per-episode level. Aggregate means/standard deviations and pooled task-count
ratios were recomputed from the combined rows. All branches used flight
psi=0.5 and the same deterministic seeds 8001--8025; each resource branch
supplied its own resource actor, normer, and runtime psi.\n""",
        encoding="utf-8",
    )

    if SUMMARY_SCRIPT.is_file():
        subprocess.run(
            [sys.executable, str(SUMMARY_SCRIPT), "--input-dir", str(cli.output_dir)],
            check=True,
        )
    print(
        json.dumps(
            {
                "output_dir": str(cli.output_dir),
                "models": len(seeds_by_model),
                "episodes_per_model": len(next(iter(seeds_by_model.values()))),
                "rows": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
