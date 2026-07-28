#!/usr/bin/env python
"""Plot matched DC-PPO GAE runs from TensorBoard scalar logs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = (
    ("agent0/system_performance_equivalent_full_GUs", "Equivalent performance of 60 MDs"),
    ("agent0/system_performance_true_all_GUs", "Performance of active MDs"),
    ("agent0/complete_task_ratio", "Task completion ratio"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-095", type=Path, required=True)
    parser.add_argument("--run-098", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(log_dir):
    accumulator = EventAccumulator(str(log_dir / "logs"), size_guidance={"scalars": 0})
    accumulator.Reload()
    result = {}
    for tag, _ in TAGS:
        events = accumulator.Scalars(tag)
        result[tag] = (
            np.asarray([event.step for event in events]),
            np.asarray([event.value for event in events]),
        )
    return result


def smooth(values, window=15):
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "GAE lambda = 0.95": load(args.run_095),
        "GAE lambda = 0.98": load(args.run_098),
    }
    colors = {"GAE lambda = 0.95": "#1f77b4", "GAE lambda = 0.98": "#d62728"}

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 12), sharex=True)
    for axis, (tag, title) in zip(axes, TAGS):
        for label, data in runs.items():
            steps, values = data[tag]
            x = steps / 1e6
            axis.plot(x, values, color=colors[label], alpha=0.14, linewidth=0.8)
            axis.plot(x, smooth(values), color=colors[label], linewidth=2.1, label=label)
        axis.set_title(title)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()
    axes[-1].set_xlabel("Environment steps (million)")
    figure.suptitle("Strict 1+5 Dynamic-MD DC-PPO Training Comparison (seed=2, 50M)", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(args.output_dir / "gae095_vs_gae098_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    with (args.output_dir / "gae095_vs_gae098_tail100_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("label", "tag", "points", "last_step", "tail100_mean", "tail100_std", "final_value"))
        for label, data in runs.items():
            for tag, _ in TAGS:
                steps, values = data[tag]
                writer.writerow((label, tag, len(values), int(steps[-1]), values[-100:].mean(),
                                 values[-100:].std(), values[-1]))

    print(args.output_dir)


if __name__ == "__main__":
    main()
