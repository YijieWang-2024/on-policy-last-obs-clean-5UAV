"""Summarize count-based reviewer metrics over common evaluation episodes.

The reviewer definitions use pooled task counts, e.g. OSR is the number of
successful offloaded task instances divided by all offloaded task instances.
This script therefore reports pooled ratios in addition to the evaluator's
episode-wise mean and standard deviation.
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent


COUNT_FIELDS = (
    "active_task_count",
    "offloaded_task_count",
    "successful_offloaded_task_count",
    "not_offloaded_task_count",
    "successful_local_task_count",
    "local_failure_count",
    "successful_task_count",
)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    cli = parser.parse_args()
    input_dir = cli.input_dir
    output = cli.output_dir or input_dir
    INPUT = input_dir / "episode_metrics.csv"
    OUTPUT = output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with INPUT.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    summaries = []
    for model, items in grouped.items():
        totals = {
            field: sum(float(item[field]) for item in items)
            for field in COUNT_FIELDS
        }
        active = totals["active_task_count"]
        offloaded = totals["offloaded_task_count"]
        local = totals["not_offloaded_task_count"]
        successful_offloaded = totals["successful_offloaded_task_count"]
        successful_local = totals["successful_local_task_count"]
        allocated = {
            field: sum(float(item[field]) for item in items) / len(items)
            for field in (
                "bandwidth_allocated_utilization",
                "bandwidth_useful_utilization",
                "bandwidth_failure_waste_ratio",
                "cpu_allocated_utilization",
                "cpu_useful_utilization",
                "cpu_failure_waste_ratio",
                "system_performance_true_all_GUs",
                "system_gain_per_active_task",
                "md_admission_ratio",
                "average_active_mds",
                "deployment_1plus4_final",
            )
        }
        first = items[0]
        summary = {
            "model": model,
            "episodes": len(items),
            "checkpoint_step": int(float(first["checkpoint_step"])),
            **{f"{key}_total": value for key, value in totals.items()},
            "offload_ratio_pooled": ratio(offloaded, active),
            "task_offloading_success_rate_pooled": ratio(
                successful_offloaded, offloaded
            ),
            "offload_yield_pooled": ratio(successful_offloaded, active),
            "local_share_pooled": ratio(local, active),
            "local_completion_rate_pooled": ratio(successful_local, local),
            "local_yield_pooled": ratio(successful_local, active),
            "overall_completion_pooled": ratio(
                successful_offloaded + successful_local, active
            ),
            "offload_failure_total": offloaded - successful_offloaded,
            "local_failure_total": local - successful_local,
            **allocated,
        }
        summaries.append(summary)

    csv_path = OUTPUT / "pooled_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    json_path = OUTPUT / "pooled_metrics.json"
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
