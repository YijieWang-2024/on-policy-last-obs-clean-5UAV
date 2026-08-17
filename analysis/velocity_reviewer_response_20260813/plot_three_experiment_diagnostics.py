#!/usr/bin/env python
"""Plot four-episode trajectories and training curves for three final runs."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
TRACE_ROOT = ROOT / "trajectory_3experiments_4eps_20260814"
FIG_ROOT = ROOT / "figures_3experiments_20260814"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "label": "vmax20_seed2",
        "short": "20 m/s, seed 2",
        "v_max": 20,
        "train_seed": 2,
        "step": 57625600,
        "run_dir": r"D:\wyj\Projects\on-policy-last-obs-clean-5UAV\onpolicy\scripts\results\mec\mappo\dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax20_seed2_60m_20260812\run1",
    },
    {
        "label": "vmax30_seed32",
        "short": "30 m/s, seed 32",
        "v_max": 30,
        "train_seed": 32,
        "step": 59980800,
        "run_dir": r"D:\wyj\Projects\on-policy-last-obs-clean-5UAV\onpolicy\scripts\results\mec\mappo\dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax30_seed32_60m_20260811_retry2\run1",
    },
    {
        "label": "vmax40_seed32",
        "short": "40 m/s, seed 32",
        "v_max": 40,
        "train_seed": 32,
        "step": 59980800,
        "run_dir": r"D:\wyj\Projects\on-policy-last-obs-clean-5UAV\onpolicy\scripts\results\mec\mappo\dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed32_60m_20260812\run1",
    },
]
EVAL_SEEDS = [1001, 1002, 1003, 1004]
COLORS = {20: "#009E73", 30: "#D55E00", 40: "#CC79A7"}
UAV_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_eval_rows():
    rows = []
    for model in MODELS:
        for eval_seed in EVAL_SEEDS:
            directory = TRACE_ROOT / model["label"] / f"eval{eval_seed}"
            data = np.load(directory / "speed_trace.npz")
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            pos = data["positions"]
            tail = pos[-200:]
            center = tail.mean(axis=0)
            tail_dist = np.linalg.norm(tail - center[None, :, :], axis=2)
            half_centers = np.stack([tail[:100].mean(axis=0), tail[100:].mean(axis=0)])
            rows.append({
                **model,
                "eval_seed": eval_seed,
                "data": data,
                "metrics": summary["metrics"],
                "tail_r95_mean_m": float(np.mean(np.quantile(tail_dist, 0.95, axis=0))),
                "tail_mean_distance_m": float(np.mean(tail_dist)),
                "tail_half_center_drift_mean_m": float(np.mean(np.linalg.norm(half_centers[0] - half_centers[1], axis=1))),
            })
    return rows


def save_figure(fig, stem):
    fig.savefig(FIG_ROOT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_overlay(rows):
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.25), sharex=True, sharey=True, constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        selected = [r for r in rows if r["label"] == model["label"]]
        for row in selected:
            pos = row["data"]["positions"]
            for uav in range(pos.shape[1]):
                ax.plot(pos[:, uav, 0], pos[:, uav, 1], color=UAV_COLORS[uav], alpha=0.22, linewidth=0.75)
                ax.plot(pos[-200:, uav, 0], pos[-200:, uav, 1], color=UAV_COLORS[uav], alpha=0.72, linewidth=1.0)
                ax.scatter(pos[0, uav, 0], pos[0, uav, 1], color=UAV_COLORS[uav], marker="o", s=10, edgecolor="white", linewidth=0.25)
                ax.scatter(pos[-1, uav, 0], pos[-1, uav, 1], color=UAV_COLORS[uav], marker="X", s=15, edgecolor="black", linewidth=0.25)
        ax.set_title(fr"$v_{{\max}}$={model['v_max']} m/s, train seed {model['train_seed']}")
        ax.set_xlim(0, 600); ax.set_ylim(0, 600); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 300, 600]); ax.set_yticks([0, 300, 600]); ax.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("y (m)")
    for ax in axes: ax.set_xlabel("x (m)")
    handles = [plt.Line2D([0], [0], color=UAV_COLORS[i], lw=1.5, label=f"UAV {i + 1}") for i in range(5)]
    handles += [plt.Line2D([0], [0], color="black", lw=1.0, alpha=0.25, label="full trajectory"),
                plt.Line2D([0], [0], color="black", lw=1.0, alpha=0.72, label="last 200 slots")]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Final-checkpoint trajectory overlays (four fixed-reset evaluations per experiment)", fontsize=10.5)
    save_figure(fig, "trajectory_overlay_3experiments_4eps")


def plot_trajectory_grid(rows):
    fig, axes = plt.subplots(3, 4, figsize=(10.2, 7.3), sharex=True, sharey=True, constrained_layout=True)
    for row_index, model in enumerate(MODELS):
        for col_index, eval_seed in enumerate(EVAL_SEEDS):
            ax = axes[row_index, col_index]
            row = next(r for r in rows if r["label"] == model["label"] and r["eval_seed"] == eval_seed)
            pos = row["data"]["positions"]
            for uav in range(pos.shape[1]):
                ax.plot(pos[:, uav, 0], pos[:, uav, 1], color=UAV_COLORS[uav], alpha=0.38, linewidth=0.75)
                ax.plot(pos[-200:, uav, 0], pos[-200:, uav, 1], color=UAV_COLORS[uav], alpha=0.85, linewidth=1.0)
                ax.scatter(pos[0, uav, 0], pos[0, uav, 1], color=UAV_COLORS[uav], marker="o", s=8, edgecolor="white", linewidth=0.2)
                ax.scatter(pos[-1, uav, 0], pos[-1, uav, 1], color=UAV_COLORS[uav], marker="X", s=12, edgecolor="black", linewidth=0.2)
            ax.set_xlim(0, 600); ax.set_ylim(0, 600); ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([0, 300, 600]); ax.set_yticks([0, 300, 600]); ax.grid(alpha=0.16, linewidth=0.5)
            ax.set_title(f"eval {eval_seed}", fontsize=8.5)
            if col_index == 0: ax.set_ylabel(f"{model['v_max']} m/s\ny (m)")
            if row_index == len(MODELS) - 1: ax.set_xlabel("x (m)")
    fig.suptitle("Episode-by-episode trajectories; dark segments mark the last 200 slots", fontsize=10.5)
    save_figure(fig, "trajectory_grid_3experiments_4eps")


def load_training_scalars(model, tags):
    logdir = Path(model["run_dir"]) / "logs"
    events = sorted(logdir.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError(f"expected one event file in {logdir}, found {len(events)}")
    acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    acc.Reload()
    result = {}
    for tag in tags:
        if tag not in acc.Tags().get("scalars", []):
            raise KeyError(f"missing scalar tag {tag} in {events[0]}")
        values = acc.Scalars(tag)
        result[tag] = (np.asarray([x.step for x in values], dtype=float), np.asarray([x.value for x in values], dtype=float))
    return result, events[0]


def smooth(values, window=41):
    if len(values) < 3: return values
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3: return values
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_training_curves():
    tags = [
        ("agent0/system_performance_true_all_GUs", 1e3, "System performance", r"System performance ($10^3$)"),
        ("agent0/complete_task_ratio", 1.0, "Complete-task ratio", "Complete-task ratio"),
        ("agent0/md_admission_ratio", 1.0, "MD admission ratio", "MD admission ratio"),
        ("agent0/actor_message_neighbor_fraction", 1.0, "Actor message-neighbor fraction", "Neighbor fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.25), constrained_layout=True)
    training_summary = []
    for ax, (tag, scale, title, ylabel) in zip(axes.flat, tags):
        for model in MODELS:
            scalars, event = load_training_scalars(model, [tag])
            steps, values = scalars[tag]
            values_scaled = values / scale
            smoothed = smooth(values_scaled, 41)
            x = steps / 1e6
            ax.plot(x, values_scaled, color=COLORS[model["v_max"]], alpha=0.12, linewidth=0.55)
            ax.plot(x, smoothed, color=COLORS[model["v_max"]], linewidth=1.45, label=f"{model['v_max']} m/s, seed {model['train_seed']}")
            n_tail = max(10, len(values_scaled) // 10)
            training_summary.append({
                "v_max": model["v_max"], "train_seed": model["train_seed"], "tag": tag,
                "n_points": len(values), "first_value": float(values_scaled[0]),
                "last_value": float(values_scaled[-1]), "last_10pct_mean": float(np.mean(values_scaled[-n_tail:])),
                "last_10pct_sd": float(np.std(values_scaled[-n_tail:], ddof=1)),
                "max_value": float(np.max(values_scaled)), "last_step": int(steps[-1]),
                "event_file": str(event),
            })
        ax.set_title(title); ax.set_xlabel("Training environment steps ($10^6$)"); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22, linewidth=0.6); ax.legend(frameon=False, loc="best")
    fig.suptitle("Training curves from TensorBoard event logs", fontsize=10.8)
    fig.text(0.5, -0.012, "Faint lines: raw logged values; solid lines: 41-point moving average. Curves use the selected final runs.", ha="center", fontsize=8)
    save_figure(fig, "training_performance_curves_3experiments")
    return training_summary


def write_summaries(rows, training_summary):
    metric_keys = [
        "system_performance_true_all_GUs", "complete_task_ratio", "md_admission_ratio",
        "actual_speed_mean_mps", "actual_speed_p95_mps", "mean_neighbor_degree",
        "mean_directed_message_edge_fraction", "connected_slot_fraction", "minimum_uav_separation_m",
    ]
    fields = ["v_max", "train_seed", "eval_seed", "checkpoint_step", "tail_r95_mean_m", "tail_mean_distance_m", "tail_half_center_drift_mean_m"] + metric_keys
    with (ROOT / "three_experiment_eval_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            out = {key: row[key] if key in row else row["metrics"].get(key) for key in fields}
            writer.writerow(out)
    with (ROOT / "three_experiment_training_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_summary[0])); writer.writeheader(); writer.writerows(training_summary)

    lines = [
        "# Three selected final experiments: trajectory and training diagnostics", "",
        "Each final checkpoint was evaluated on the same four fixed-reset seeds (1001--1004). The training curves are read from the single TensorBoard event file in each run1/logs directory.", "",
        "| experiment | training seed | checkpoint step | mean final performance (10^3) | mean actual speed (m/s) | mean degree | active-edge fraction | connected fraction | tail r95 (m) | half-tail center drift (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        rs = [r for r in rows if r["label"] == model["label"]]
        def avg(key, scale=1.0):
            vals = np.asarray([r["metrics"].get(key, r.get(key)) for r in rs], dtype=float) / scale
            return f"{vals.mean():.3f} +/- {vals.std(ddof=1):.3f}"
        lines.append(
            f"| {model['short']} | {model['train_seed']} | {model['step']:,} | {avg('system_performance_true_all_GUs', 1e3)} | "
            f"{avg('actual_speed_mean_mps')} | {avg('mean_neighbor_degree')} | {avg('mean_directed_message_edge_fraction')} | "
            f"{avg('connected_slot_fraction')} | {avg('tail_r95_mean_m')} | {avg('tail_half_center_drift_mean_m')} |"
        )
    lines += ["", "## Checkpoint directories", ""]
    for model in MODELS:
        lines.append(f"- {model['short']}: `{model['run_dir']}\\models` (manifest step {model['step']:,})")
    lines += ["", "## Interpretation guardrail", "", "The trajectory figures show final behavior for one training seed per experiment. Training curves are descriptive convergence diagnostics; because v_max and training seed both differ between these three selected runs, they are not a matched causal speed ablation."]
    (ROOT / "three_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = load_eval_rows()
    if len(rows) != 12: raise RuntimeError(f"expected 12 evaluation episodes, found {len(rows)}")
    plot_trajectory_overlay(rows)
    plot_trajectory_grid(rows)
    training_summary = plot_training_curves()
    write_summaries(rows, training_summary)
    print(json.dumps({
        "episodes": len(rows),
        "figures": sorted(p.name for p in FIG_ROOT.iterdir()),
        "eval_csv": str(ROOT / "three_experiment_eval_metrics.csv"),
        "training_csv": str(ROOT / "three_experiment_training_summary.csv"),
        "summary": str(ROOT / "three_experiment_summary.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
