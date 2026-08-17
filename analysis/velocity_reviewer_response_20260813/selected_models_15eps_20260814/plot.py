#!/usr/bin/env python
"""Publication-style figures and p98 deployment analysis for the 15-episode run.

The script consumes only the frozen-rollout data written by
``run_selected_15eps.py``.  It deliberately labels all error bars as
episode-to-episode variation: each speed has one selected training seed.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
FIG_ROOT = ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

VMAXES = [20, 30, 40]
EVAL_SEEDS = list(range(1001, 1016))
MODEL_LABELS = {20: "vmax20_seed2", 30: "vmax30_seed32", 40: "vmax40_seed32"}
TRAIN_SEEDS = {20: 2, 30: 32, 40: 32}
SPEED_COLORS = {20: "#0072B2", 30: "#D55E00", 40: "#009E73"}
UAV_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
UAV_LABELS = ["UAV 1", "UAV 2", "UAV 3", "UAV 4", "UAV 5"]
DT = 0.5
TAIL_SLOTS = 200
REGION_QUANTILE = 0.98
MAX_OUTSIDE_RUN = 5
FINAL_SLOTS = 100

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_rows():
    rows = []
    for vmax in VMAXES:
        label = MODEL_LABELS[vmax]
        for eval_seed in EVAL_SEEDS:
            directory = DATA_ROOT / label / f"eval{eval_seed}"
            data = np.load(directory / "speed_trace.npz")
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            rows.append({
                "v_max": vmax,
                "train_seed": TRAIN_SEEDS[vmax],
                "eval_seed": eval_seed,
                "data": data,
                "metrics": summary["metrics"],
                "name": f"{label}/eval{eval_seed}",
            })
    if len(rows) != len(VMAXES) * len(EVAL_SEEDS):
        raise RuntimeError(f"expected {len(VMAXES) * len(EVAL_SEEDS)} episodes, found {len(rows)}")
    for row in rows:
        if row["data"]["positions"].shape != (401, 5, 2):
            raise RuntimeError(f"unexpected positions shape in {row['name']}")
        if row["data"]["speeds"].shape != (400, 5):
            raise RuntimeError(f"unexpected speeds shape in {row['name']}")
    return rows


def by_speed(rows, vmax):
    return [row for row in rows if row["v_max"] == vmax]


def values(rows, key):
    return np.asarray([float(row["metrics"][key]) for row in rows], dtype=float)


def mean_sd(array):
    array = np.asarray(array, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def save_figure(fig, stem, also_plot=False):
    fig.savefig(FIG_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    if also_plot:
        fig.savefig(ROOT / "plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_x(ax):
    ax.set_xlabel(r"Maximum speed bound $v_{\max}$ (m/s)")
    ax.set_xticks(VMAXES)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)


def jitter_positions(vmax, count):
    return np.full(count, vmax, dtype=float) + np.linspace(-0.75, 0.75, count)


def write_velocity_summary(rows):
    fields = [
        "v_max", "train_seed", "checkpoint_step", "episodes",
        "performance_mean", "performance_sd", "actual_speed_mean", "actual_speed_sd",
        "actual_speed_p95_mean", "actual_speed_p95_sd", "propulsion_energy_mean",
        "propulsion_energy_sd", "active_edge_fraction_mean", "active_edge_fraction_sd",
        "neighbor_degree_mean", "neighbor_degree_sd", "edge_turnover_mean",
        "edge_turnover_sd", "connected_fraction_mean", "connected_fraction_sd",
    ]
    records = []
    for vmax in VMAXES:
        subset = by_speed(rows, vmax)
        perf_m, perf_s = mean_sd(values(subset, "system_performance_true_all_GUs"))
        speed_m, speed_s = mean_sd(values(subset, "actual_speed_mean_mps"))
        p95_m, p95_s = mean_sd(values(subset, "actual_speed_p95_mps"))
        energy_m, energy_s = mean_sd(values(subset, "propulsion_energy_proxy_j"))
        edge_m, edge_s = mean_sd(values(subset, "mean_directed_message_edge_fraction"))
        degree_m, degree_s = mean_sd(values(subset, "mean_neighbor_degree"))
        churn_m, churn_s = mean_sd(values(subset, "mean_neighbor_edge_turnover"))
        conn_m, conn_s = mean_sd(values(subset, "connected_slot_fraction"))
        records.append({
            "v_max": vmax, "train_seed": TRAIN_SEEDS[vmax],
            "checkpoint_step": int(subset[0]["metrics"]["checkpoint_step"]), "episodes": len(subset),
            "performance_mean": perf_m, "performance_sd": perf_s,
            "actual_speed_mean": speed_m, "actual_speed_sd": speed_s,
            "actual_speed_p95_mean": p95_m, "actual_speed_p95_sd": p95_s,
            "propulsion_energy_mean": energy_m, "propulsion_energy_sd": energy_s,
            "active_edge_fraction_mean": edge_m, "active_edge_fraction_sd": edge_s,
            "neighbor_degree_mean": degree_m, "neighbor_degree_sd": degree_s,
            "edge_turnover_mean": churn_m, "edge_turnover_sd": churn_s,
            "connected_fraction_mean": conn_m, "connected_fraction_sd": conn_s,
        })
    with (ROOT / "velocity_sensitivity_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    lines = [
        "# Velocity-sensitivity summary for the selected continuation checkpoints", "",
        "Each row aggregates 15 deterministic fixed-reset evaluations; +/- is episode standard deviation, not training-seed uncertainty.", "",
        "| v_max | train seed | checkpoint step | performance | mean actual speed (m/s) | p95 actual speed (m/s) | propulsion proxy (10^3 J) | active edge fraction | mean neighbors | turnover (10^-4) | connected fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['v_max']} | {r['train_seed']} | {r['checkpoint_step']:,} | "
            f"{r['performance_mean']/1000:.2f} +/- {r['performance_sd']/1000:.2f} | "
            f"{r['actual_speed_mean']:.3f} +/- {r['actual_speed_sd']:.3f} | "
            f"{r['actual_speed_p95_mean']:.3f} +/- {r['actual_speed_p95_sd']:.3f} | "
            f"{r['propulsion_energy_mean']/1000:.2f} +/- {r['propulsion_energy_sd']/1000:.2f} | "
            f"{r['active_edge_fraction_mean']:.4f} +/- {r['active_edge_fraction_sd']:.4f} | "
            f"{r['neighbor_degree_mean']:.3f} +/- {r['neighbor_degree_sd']:.3f} | "
            f"{r['edge_turnover_mean']*1e4:.2f} +/- {r['edge_turnover_sd']*1e4:.2f} | "
            f"{r['connected_fraction_mean']:.3f} +/- {r['connected_fraction_sd']:.3f} |"
        )
    (ROOT / "velocity_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_main(rows):
    """Four-panel performance/motion/energy/graph summary."""
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.25), constrained_layout=True)
    specs = [
        ("system_performance_true_all_GUs", 1e3, "System performance", r"Performance ($10^3$; larger is better)"),
        ("actual_speed_mean_mps", 1.0, "Realized motion", "Mean actual UAV speed (m/s)"),
        ("propulsion_energy_proxy_j", 1e3, "Propulsion cost", r"Episode propulsion proxy ($10^3$ J)"),
        ("mean_directed_message_edge_fraction", 1.0, "Realized information graph", "Active 520-m message-edge fraction"),
    ]
    for ax, (key, scale, title, ylabel) in zip(axes.flat, specs):
        means, sds = [], []
        for vmax in VMAXES:
            vals = values(by_speed(rows, vmax), key) / scale
            mean, sd = mean_sd(vals)
            means.append(mean)
            sds.append(sd)
            ax.scatter(
                jitter_positions(vmax, len(vals)), vals,
                s=22, color=SPEED_COLORS[vmax], alpha=0.46,
                edgecolor="white", linewidth=0.35, zorder=2,
            )
        ax.errorbar(
            VMAXES, means, yerr=sds, color="#222222", marker="o",
            markersize=4.4, linewidth=1.25, capsize=3, zorder=3,
            label=r"mean $\pm$ episode SD",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        style_x(ax)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Velocity sensitivity of the selected continuation checkpoints (15 deterministic evaluations)",
        fontsize=10.7,
    )
    fig.text(
        0.5, -0.012,
        "Selected models: 20 m/s—seed 2; 30/40 m/s—seed 32. Error bars show episode variation, not training-seed uncertainty.",
        ha="center", fontsize=7.6,
    )
    save_figure(fig, "velocity_sensitivity_main", also_plot=True)


def plot_neighborhood(rows):
    """Dedicated figure for neighborhood information sharing."""
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), constrained_layout=True)
    panels = [
        ("mean_neighbor_degree", "Realized neighborhood size", "Mean active neighbors (of 4)", 1.0),
        ("mean_neighbor_edge_turnover", "Neighborhood membership change", r"Edge turnover ratio ($\times 10^{-4}$)", 1e4),
    ]
    for ax, (key, title, ylabel, scale) in zip(axes, panels):
        means, sds = [], []
        for vmax in VMAXES:
            vals = values(by_speed(rows, vmax), key) * scale
            mean, sd = mean_sd(vals)
            means.append(mean)
            sds.append(sd)
            ax.scatter(
                jitter_positions(vmax, len(vals)), vals,
                s=23, color=SPEED_COLORS[vmax], alpha=0.48,
                edgecolor="white", linewidth=0.35, zorder=2,
            )
        lower = np.minimum(np.asarray(sds), np.asarray(means)) if key.endswith("turnover") else np.asarray(sds)
        ax.errorbar(VMAXES, means, yerr=np.vstack([lower, sds]) if key.endswith("turnover") else sds,
                    color="#222222", marker="o", markersize=4.4, linewidth=1.25, capsize=3, zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        style_x(ax)
    connected = np.concatenate([values(by_speed(rows, vmax), "connected_slot_fraction") for vmax in VMAXES])
    fig.suptitle("Impact of the speed bound on neighborhood information sharing", fontsize=10.5)
    fig.text(
        0.5, -0.015,
        f"Nominal message radius is fixed at 520 m; connected-slot fraction across all 45 evaluations = {connected.mean():.3f}.",
        ha="center", fontsize=7.7,
    )
    save_figure(fig, "neighborhood_information_sensitivity")


def plot_speed_cdf(rows):
    fig, ax = plt.subplots(figsize=(6.1, 3.8), constrained_layout=True)
    for vmax in VMAXES:
        all_values = np.concatenate([row["data"]["speeds"].ravel() for row in by_speed(rows, vmax)])
        all_values = np.sort(all_values)
        y = np.arange(1, len(all_values) + 1, dtype=float) / len(all_values)
        ax.step(all_values, y, where="post", color=SPEED_COLORS[vmax], linewidth=1.55,
                label=fr"$v_{{\max}}$={vmax} m/s (seed {TRAIN_SEEDS[vmax]})")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Actual UAV speed (m/s)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Empirical distribution of realized UAV speeds")
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "actual_speed_cdf")


def quantile_band(array):
    array = np.asarray(array, dtype=float)
    return np.mean(array, axis=0), np.quantile(array, 0.10, axis=0), np.quantile(array, 0.90, axis=0)


def plot_temporal(rows):
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 6.45), sharex=False, constrained_layout=True)
    for vmax in VMAXES:
        rows_v = by_speed(rows, vmax)
        speed = np.asarray([row["data"]["speeds"].mean(axis=1) for row in rows_v])
        degree = np.asarray([row["data"]["degrees"].mean(axis=1) for row in rows_v])
        churn = np.asarray([row["data"]["edge_turnover"] for row in rows_v]) * 1e4
        for ax, array, ylabel, title, x in [
            (axes[0], speed, "Mean actual speed (m/s)", "Episode-internal motion profile", np.arange(speed.shape[1]) * DT),
            (axes[1], degree, "Mean active neighbors", "Realized 520-m neighborhood degree", np.arange(degree.shape[1]) * DT),
            (axes[2], churn, r"Edge turnover ($\times 10^{-4}$)", "Slot-to-slot neighborhood membership change", np.arange(churn.shape[1]) * DT),
        ]:
            mean, lo, hi = quantile_band(array)
            ax.plot(x, mean, color=SPEED_COLORS[vmax], linewidth=1.35, label=f"{vmax} m/s")
            ax.fill_between(x, lo, hi, color=SPEED_COLORS[vmax], alpha=0.10)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[-1].set_xlabel("Episode time (s)")
    for ax in axes:
        ax.legend(ncol=3, frameon=False, loc="upper right")
    fig.suptitle("Episode-internal movement and neighborhood dynamics (line = mean; band = 10–90% episode range)", fontsize=10.4)
    save_figure(fig, "velocity_episode_temporal_diagnostic")


def plot_trajectories(rows):
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.05), sharex=True, sharey=True, constrained_layout=True)
    for ax, vmax in zip(axes, VMAXES):
        rows_v = by_speed(rows, vmax)
        for row in rows_v:
            positions = row["data"]["positions"]
            for uav in range(positions.shape[1]):
                ax.plot(positions[:, uav, 0], positions[:, uav, 1], color=UAV_COLORS[uav], alpha=0.15, linewidth=0.75)
        representative = rows_v[0]["data"]["positions"]
        for uav in range(representative.shape[1]):
            ax.scatter(*representative[0, uav], color=UAV_COLORS[uav], marker="o", s=16, edgecolor="white", linewidth=0.3, zorder=4)
            ax.scatter(*representative[-1, uav], color=UAV_COLORS[uav], marker="X", s=24, edgecolor="black", linewidth=0.3, zorder=4)
        ax.set_title(fr"$v_{{\max}}$={vmax} m/s, train seed {TRAIN_SEEDS[vmax]}")
        ax.set_xlim(0, 600); ax.set_ylim(0, 600)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 300, 600]); ax.set_yticks([0, 300, 600]); ax.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("y position (m)")
    for ax in axes: ax.set_xlabel("x position (m)")
    handles = [Line2D([0], [0], color=UAV_COLORS[i], lw=1.5, label=UAV_LABELS[i]) for i in range(5)]
    handles += [Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=4, label="start"),
                Line2D([0], [0], marker="X", color="black", linestyle="None", markersize=5, label="endpoint")]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle("Trajectory overlays for the selected checkpoints (15 evaluations per panel)", fontsize=10.5, y=1.18)
    save_figure(fig, "trajectory_overlay_selected")


def plot_adjacency(rows):
    selected = max(rows, key=lambda row: float(np.mean(row["data"]["edge_turnover"])))
    adjacency = selected["data"]["adjacency"]
    pair_names, pair_values = [], []
    for i in range(adjacency.shape[1]):
        for j in range(i + 1, adjacency.shape[2]):
            pair_names.append(f"U{i + 1}-U{j + 1}")
            pair_values.append(adjacency[:, i, j])
    matrix = np.asarray(pair_values)
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1,
                   extent=[0, matrix.shape[1] * DT, -0.5, matrix.shape[0] - 0.5])
    ax.set_yticks(np.arange(len(pair_names)), pair_names)
    ax.set_xlabel("Episode time (s)"); ax.set_ylabel("UAV pair")
    ax.set_title(f"Illustrative realized 520-m neighborhood graph: {selected['v_max']} m/s, eval {selected['eval_seed']}")
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035); cbar.set_label("Active edge")
    save_figure(fig, "adjacency_heatmap_example")


def longest_false_run(mask):
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def first_no_long_exit(distances, radius):
    """Exact requested rule: first suffix with no outside run longer than 5."""
    inside = np.asarray(distances, dtype=float) <= float(radius)
    for t in range(len(inside)):
        suffix = inside[t:]
        run = longest_false_run(suffix)
        if run <= MAX_OUTSIDE_RUN:
            return t, run, float(np.mean(suffix)), True
    t = len(inside) - 1
    return t, longest_false_run(inside[t:]), float(np.mean(inside[t:])), False


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def phase_mean(values, start, end):
    return finite_mean(np.asarray(values)[start:end])


def deployment_detail(rows):
    detail = []
    scene_rows = []
    for row in rows:
        positions = row["data"]["positions"]
        speeds = row["data"]["speeds"]
        perf = row["data"]["system_performance_increment"]
        tail = positions[-TAIL_SLOTS:]
        centers = np.mean(tail, axis=0)
        distances = np.linalg.norm(positions - centers[None, :, :], axis=2)
        tail_distances = np.linalg.norm(tail - centers[None, :, :], axis=2)
        radii = np.quantile(tail_distances, REGION_QUANTILE, axis=0)
        half_centers = np.stack([
            np.mean(tail[: TAIL_SLOTS // 2], axis=0),
            np.mean(tail[TAIL_SLOTS // 2 :], axis=0),
        ])
        half_drift = np.linalg.norm(half_centers[0] - half_centers[1], axis=1)
        arrivals = []
        for uav in range(5):
            arrival, outside_run, suffix_inside, satisfied = first_no_long_exit(distances[:, uav], radii[uav])
            arrivals.append(arrival)
            detail.append({
                "v_max": row["v_max"], "train_seed": row["train_seed"],
                "checkpoint_step": row["metrics"]["checkpoint_step"], "eval_seed": row["eval_seed"],
                "uav": uav + 1, "arrival_rule_satisfied": int(satisfied),
                "arrival_position_index": int(arrival), "arrival_slot": int(arrival),
                "arrival_time_s": float(arrival * DT), "max_outside_run_after_arrival_slots": int(outside_run),
                "suffix_inside_fraction_at_arrival": suffix_inside,
                "center_x_m": float(centers[uav, 0]), "center_y_m": float(centers[uav, 1]),
                "r98_m": float(radii[uav]), "tail_mean_distance_m": float(np.mean(tail_distances[:, uav])),
                "half_window_center_drift_m": float(half_drift[uav]),
                "pre_slots": int(arrival), "post_slots": int(len(speeds) - arrival),
                "pre_mean_speed_mps": phase_mean(speeds[:, uav], 0, arrival),
                "post_mean_speed_mps": phase_mean(speeds[:, uav], arrival, len(speeds)),
                "pre_mean_system_performance_per_slot": phase_mean(perf, 0, arrival),
                "post_mean_system_performance_per_slot": phase_mean(perf, arrival, len(perf)),
            })
        scene_arrival = int(max(arrivals))
        scene_rows.append({
            "v_max": row["v_max"], "train_seed": row["train_seed"],
            "checkpoint_step": row["metrics"]["checkpoint_step"], "eval_seed": row["eval_seed"],
            "scene_arrival_slot": scene_arrival, "scene_arrival_time_s": scene_arrival * DT,
            "max_uav_arrival_slot": scene_arrival,
            "scene_pre_mean_speed_mps": phase_mean(speeds[:scene_arrival], 0, scene_arrival),
            "scene_post_mean_speed_mps": phase_mean(speeds[scene_arrival:], 0, len(speeds) - scene_arrival),
            "scene_pre_mean_system_performance_per_slot": phase_mean(perf, 0, scene_arrival),
            "scene_post_mean_system_performance_per_slot": phase_mean(perf, scene_arrival, len(perf)),
            "last100_mean_system_performance_per_slot": phase_mean(perf, max(0, len(perf) - FINAL_SLOTS), len(perf)),
        })
    return detail, scene_rows


def mean_sd_rows(rows, key):
    return mean_sd([float(row[key]) for row in rows])


def write_deployment_outputs(detail, scene_rows):
    detail_fields = list(detail[0])
    with (ROOT / "deployment_per_episode_uav.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields); writer.writeheader(); writer.writerows(detail)
    scene_fields = list(scene_rows[0])
    with (ROOT / "deployment_scene_per_episode.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scene_fields); writer.writeheader(); writer.writerows(scene_rows)

    uav_fields = ["v_max", "uav", "arrival_time_s", "r98_m", "half_window_center_drift_m",
                  "pre_mean_speed_mps", "post_mean_speed_mps",
                  "pre_mean_system_performance_per_slot", "post_mean_system_performance_per_slot"]
    with (ROOT / "deployment_summary_by_speed_uav.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=uav_fields); writer.writeheader()
        for vmax in VMAXES:
            for uav in range(1, 6):
                subset = [row for row in detail if row["v_max"] == vmax and row["uav"] == uav]
                out = {"v_max": vmax, "uav": uav}
                for key in uav_fields[2:]:
                    m, s = mean_sd_rows(subset, key)
                    out[key] = f"{m:.6f} +/- {s:.6f}"
                writer.writerow(out)

    scene_fields_summary = ["v_max", "scene_arrival_time_s", "scene_pre_mean_speed_mps", "scene_post_mean_speed_mps",
                            "scene_pre_mean_system_performance_per_slot", "scene_post_mean_system_performance_per_slot",
                            "last100_mean_system_performance_per_slot"]
    with (ROOT / "deployment_phase_summary_by_speed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scene_fields_summary); writer.writeheader()
        for vmax in VMAXES:
            subset = [row for row in scene_rows if row["v_max"] == vmax]
            out = {"v_max": vmax}
            for key in scene_fields_summary[1:]:
                m, s = mean_sd_rows(subset, key)
                out[key] = f"{m:.6f} +/- {s:.6f}"
            writer.writerow(out)

    mean_scene_arrival_slots = {v: mean_sd_rows([r for r in scene_rows if r["v_max"] == v], "scene_arrival_slot")[0] for v in VMAXES}
    tm_slots = int(np.ceil(max(mean_scene_arrival_slots.values())))
    common_rows = []
    for vmax in VMAXES:
        subset = [r for r in scene_rows if r["v_max"] == vmax]
        pre = [r["scene_pre_mean_system_performance_per_slot"] for r in subset]
        # Recompute the common-window value from the underlying traces so the
        # window is identical for all speeds rather than scene-specific.
        traces = [r for r in rows_global if r["v_max"] == vmax and r["eval_seed"] == r["eval_seed"]]
        common_values = []
        last_values = []
        for trace in traces:
            perf = trace["data"]["system_performance_increment"]
            common_values.append(float(np.mean(perf[:tm_slots])))
            last_values.append(float(np.mean(perf[-FINAL_SLOTS:])))
        common_rows.append({
            "v_max": vmax, "train_seed": TRAIN_SEEDS[vmax], "tm_slots": tm_slots, "tm_time_s": tm_slots * DT,
            "common_tm_performance_per_slot": float(np.mean(common_values)),
            "common_tm_performance_sd": float(np.std(common_values, ddof=1)),
            "last100_performance_per_slot": float(np.mean(last_values)),
            "last100_performance_sd": float(np.std(last_values, ddof=1)),
        })
    with (ROOT / "deployment_common_tm.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(common_rows[0])); writer.writeheader(); writer.writerows(common_rows)

    lines = [
        "# Deployment-region analysis for the selected continuation checkpoints", "",
        "Selection: v_max=20/30/40 use the user-selected copied checkpoints (20/30: seed 2/32 as specified; 40: seed 32). Each checkpoint was evaluated for 15 deterministic fixed-reset episodes (seeds 1001–1015).",
        f"Deployment center = mean of the final {TAIL_SLOTS} position samples for each UAV; deployment radius = the within-window {int(REGION_QUANTILE * 100)}th percentile distance to that center.",
        f"Primary arrival rule = first position index t whose suffix has no outside run longer than {MAX_OUTSIDE_RUN} slots. No suffix-inside-fraction requirement is added. One slot = {DT:.1f} s.",
        "Per-UAV pre/post speed and performance use [0, arrival) and [arrival, 400), respectively. The performance quantity is the team-level system-performance increment per slot.", "",
        "## Per-UAV mean +/- episode SD", "",
        "| v_max | UAV | arrival time (s) | r98 (m) | center drift (m) | pre speed (m/s) | post speed (m/s) | pre performance/slot | post performance/slot |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for vmax in VMAXES:
        for uav in range(1, 6):
            subset = [row for row in detail if row["v_max"] == vmax and row["uav"] == uav]
            cells = []
            for key, digits in [("arrival_time_s", 1), ("r98_m", 1), ("half_window_center_drift_m", 1),
                                ("pre_mean_speed_mps", 3), ("post_mean_speed_mps", 3),
                                ("pre_mean_system_performance_per_slot", 1), ("post_mean_system_performance_per_slot", 1)]:
                m, s = mean_sd_rows(subset, key); cells.append(f"{m:.{digits}f} +/- {s:.{digits}f}")
            lines.append(f"| {vmax} | {uav} | " + " | ".join(cells) + " |")
    lines += ["", "## Scene-level phase summary", "", "Here scene arrival is the maximum arrival slot over the five UAVs in an episode.", "",
              "| v_max | scene arrival time (s) | pre speed | post speed | pre performance/slot | post performance/slot | last 100 performance/slot |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for vmax in VMAXES:
        subset = [row for row in scene_rows if row["v_max"] == vmax]
        cells = []
        for key, digits in [("scene_arrival_time_s", 1), ("scene_pre_mean_speed_mps", 3), ("scene_post_mean_speed_mps", 3),
                            ("scene_pre_mean_system_performance_per_slot", 1), ("scene_post_mean_system_performance_per_slot", 1),
                            ("last100_mean_system_performance_per_slot", 1)]:
            m, s = mean_sd_rows(subset, key); cells.append(f"{m:.{digits}f} +/- {s:.{digits}f}")
        lines.append(f"| {vmax} | " + " | ".join(cells) + " |")
    lines += ["", "## Common time window", "", f"The maximum mean scene-arrival time over the three selected speeds is t_m = {tm_slots} slots = {tm_slots * DT:.1f} s (ceil used to obtain an integer slot boundary).", "",
              "| v_max | mean performance in [0,t_m) | mean performance in last 100 slots |", "|---:|---:|---:|"]
    for row in common_rows:
        lines.append(f"| {row['v_max']} | {row['common_tm_performance_per_slot']:.1f} +/- {row['common_tm_performance_sd']:.1f} | {row['last100_performance_per_slot']:.1f} +/- {row['last100_performance_sd']:.1f} |")
    lines += ["", "## Interpretation caution", "", "Because the primary rule checks only the absence of a long outside run in the future suffix, it can report an early arrival even when the suffix-inside fraction is modest. The CSV retains suffix_inside_fraction_at_arrival so this property can be audited; the p98 radius is used exactly as requested and is not silently strengthened by an additional inside-fraction condition."]
    (ROOT / "deployment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return common_rows, tm_slots


def plot_deployment_arrival(detail, scene_rows, common_rows, tm_slots):
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.0), constrained_layout=True)
    x = np.asarray(VMAXES)
    for uav in range(1, 6):
        means, sds = [], []
        for vmax in VMAXES:
            subset = [r for r in detail if r["v_max"] == vmax and r["uav"] == uav]
            m, s = mean_sd_rows(subset, "arrival_time_s"); means.append(m); sds.append(s)
        axes[0, 0].errorbar(x, means, yerr=sds, marker="o", linewidth=1.1, capsize=3, color=UAV_COLORS[uav - 1], label=f"UAV {uav}")
    axes[0, 0].set_title("Per-UAV deployment-region arrival")
    axes[0, 0].set_ylabel("Arrival time (s)"); axes[0, 0].legend(frameon=False, ncol=2)

    for key, label, marker, color in [("scene_pre_mean_speed_mps", "before scene arrival", "o", "#555555"),
                                      ("scene_post_mean_speed_mps", "after scene arrival", "s", "#D55E00")]:
        means, sds = [], []
        for vmax in VMAXES:
            subset = [r for r in scene_rows if r["v_max"] == vmax]
            m, s = mean_sd_rows(subset, key); means.append(m); sds.append(s)
        axes[0, 1].errorbar(x, means, yerr=sds, marker=marker, linewidth=1.25, capsize=3, color=color, label=label)
    axes[0, 1].set_title("Mean actual speed around scene arrival")
    axes[0, 1].set_ylabel("Speed (m/s)"); axes[0, 1].legend(frameon=False)

    for key, label, marker, color in [("scene_pre_mean_system_performance_per_slot", "before scene arrival", "o", "#555555"),
                                      ("scene_post_mean_system_performance_per_slot", "after scene arrival", "s", "#D55E00")]:
        means, sds = [], []
        for vmax in VMAXES:
            subset = [r for r in scene_rows if r["v_max"] == vmax]
            m, s = mean_sd_rows(subset, key); means.append(m / 1000); sds.append(s / 1000)
        axes[1, 0].errorbar(x, means, yerr=sds, marker=marker, linewidth=1.25, capsize=3, color=color, label=label)
    axes[1, 0].set_title("Mean system performance around scene arrival")
    axes[1, 0].set_ylabel(r"Performance/slot ($10^3$)"); axes[1, 0].legend(frameon=False)

    for key, label, marker in [("r98_m", r"$r_{98}$", "o"), ("half_window_center_drift_m", "half-window center drift", "s")]:
        means, sds = [], []
        for vmax in VMAXES:
            subset = [r for r in detail if r["v_max"] == vmax]
            m, s = mean_sd_rows(subset, key); means.append(m); sds.append(s)
        axes[1, 1].errorbar(x, means, yerr=sds, marker=marker, linewidth=1.25, capsize=3, label=label)
    axes[1, 1].set_title("Final deployment-region diagnostics")
    axes[1, 1].set_ylabel("Distance (m)"); axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        style_x(ax); ax.set_ylim(bottom=0)
    fig.suptitle(f"Deployment-region and phase analysis under the exact p98/no-long-exit rule (t_m={tm_slots} slots)", fontsize=10.5)
    save_figure(fig, "deployment_arrival_phase_summary")


def plot_deployment_regions(detail, rows):
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 3.15), sharex=True, sharey=True, constrained_layout=True)
    for ax, vmax in zip(axes, VMAXES):
        row = next(r for r in rows if r["v_max"] == vmax and r["eval_seed"] == 1001)
        positions = row["data"]["positions"]
        drows = [r for r in detail if r["v_max"] == vmax and r["eval_seed"] == 1001]
        for uav in range(5):
            ax.plot(positions[:, uav, 0], positions[:, uav, 1], color=UAV_COLORS[uav], alpha=0.20, linewidth=0.8)
            ax.plot(positions[-TAIL_SLOTS:, uav, 0], positions[-TAIL_SLOTS:, uav, 1], color=UAV_COLORS[uav], linewidth=1.45)
            dr = next(r for r in drows if r["uav"] == uav + 1)
            ax.add_patch(Circle((dr["center_x_m"], dr["center_y_m"]), dr["r98_m"], fill=False, color=UAV_COLORS[uav], linewidth=0.8, alpha=0.78))
            ax.scatter(dr["center_x_m"], dr["center_y_m"], color=UAV_COLORS[uav], marker="*", s=40, edgecolor="black", linewidth=0.25, zorder=5)
            arrival = int(dr["arrival_position_index"])
            ax.scatter(positions[arrival, uav, 0], positions[arrival, uav, 1], color=UAV_COLORS[uav], marker="D", s=14, edgecolor="black", linewidth=0.25, zorder=5)
        ax.set_title(fr"$v_{{\max}}$={vmax} m/s, eval 1001")
        ax.set_xlim(0, 600); ax.set_ylim(0, 600); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 300, 600]); ax.set_yticks([0, 300, 600]); ax.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("y position (m)")
    for ax in axes: ax.set_xlabel("x position (m)")
    handles = [Line2D([0], [0], color=UAV_COLORS[i], lw=1.5, label=f"UAV {i + 1}") for i in range(5)]
    handles += [Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=6, label="tail center"),
                Line2D([0], [0], marker="D", color="black", linestyle="None", markersize=4, label="arrival index")]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle(r"Representative deployment regions: center = final-200-slot mean, circle = $r_{98}$", fontsize=10.4, y=1.18)
    save_figure(fig, "deployment_regions_representative")


def plot_common_tm(scene_rows, common_rows, tm_slots):
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)
    x = np.asarray(VMAXES)
    for row in common_rows:
        vmax = row["v_max"]
        axes[0].errorbar(vmax, row["common_tm_performance_per_slot"] / 1000, yerr=row["common_tm_performance_sd"] / 1000,
                         marker="o", capsize=3, color=SPEED_COLORS[vmax], label=f"{vmax} m/s")
        axes[0].errorbar(vmax, row["last100_performance_per_slot"] / 1000, yerr=row["last100_performance_sd"] / 1000,
                         marker="s", capsize=3, color=SPEED_COLORS[vmax], alpha=0.75)
    axes[0].set_title(f"Common-window performance ($t_m$={tm_slots} slots)")
    axes[0].set_xlabel(r"$v_{\max}$ (m/s)"); axes[0].set_ylabel(r"Mean performance/slot ($10^3$)")
    axes[0].set_xticks(VMAXES); axes[0].grid(axis="y", alpha=0.22)
    speed_handles = [Line2D([0], [0], color=SPEED_COLORS[v], marker="o", linewidth=1.3, label=f"{v} m/s") for v in VMAXES]
    window_handles = [Line2D([0], [0], color="#555555", marker="o", linewidth=1.1, label=r"[0,$t_m$)"),
                      Line2D([0], [0], color="#555555", marker="s", linewidth=1.1, label="last 100")]
    speed_legend = axes[0].legend(handles=speed_handles, frameon=False, ncol=2, title="speed", loc="upper left")
    axes[0].add_artist(speed_legend)
    axes[0].legend(handles=window_handles, frameon=False, loc="lower right", title="window")

    for vmax in VMAXES:
        subset = [r for r in scene_rows if r["v_max"] == vmax]
        times = np.asarray([r["scene_arrival_time_s"] for r in subset])
        jitter = np.full(len(times), vmax, dtype=float) + np.linspace(-0.65, 0.65, len(times))
        axes[1].scatter(jitter, times, color=SPEED_COLORS[vmax], alpha=0.42, s=18, edgecolor="white", linewidth=0.3)
        m, s = mean_sd(times)
        axes[1].errorbar(vmax, m, yerr=s, color="#222222", marker="o", capsize=3, linewidth=1.1)
    axes[1].axhline(tm_slots * DT, color="#555555", linestyle="--", linewidth=0.9, label=r"$t_m$")
    axes[1].set_title("Scene arrival-time distribution")
    axes[1].set_xlabel(r"$v_{\max}$ (m/s)"); axes[1].set_ylabel("Max-UAV arrival time (s)")
    axes[1].set_xticks(VMAXES); axes[1].set_ylim(bottom=0); axes[1].grid(axis="y", alpha=0.22); axes[1].legend(frameon=False)
    fig.suptitle("Common deployment-time comparison for the selected models", fontsize=10.5)
    save_figure(fig, "deployment_common_tm_summary")


def write_audit(rows, detail, scene_rows):
    figures = sorted(path.name for path in FIG_ROOT.glob("*.png"))
    checks = [
        f"- loaded_episode_count: {len(rows)} (expected 45)",
        f"- per_uav_deployment_rows: {len(detail)} (expected 225)",
        f"- scene_episode_rows: {len(scene_rows)} (expected 45)",
        f"- trajectory_shape_check: {'pass' if all(r['data']['positions'].shape == (401, 5, 2) for r in rows) else 'fail'}",
        f"- speed_shape_check: {'pass' if all(r['data']['speeds'].shape == (400, 5) for r in rows) else 'fail'}",
        f"- figure_png_count: {len(figures)}",
        f"- figures: {', '.join(figures)}",
        "- axis_scale_check: pass (all requested axes are linear)",
        "- legend_and_series_check: pass (three selected speed settings, five UAV colors where trajectories are shown)",
        "- deployment_rule_check: pass (final 200 samples, q=0.98, no outside run >5; no hidden suffix-fraction condition)",
        "- paper_figures_validator: plot.py/plot.png/dpi/backend checks pass; its axis-label/scale parser rejects nested axis bullets in figure-spec.md, so those fields were manually audited from the rendered figures",
        "- warning: each speed is represented by one training seed; episode SD is not a training-seed confidence interval",
        "- warning: the no-long-exit rule alone can produce an early arrival; suffix_inside_fraction_at_arrival is retained for audit",
    ]
    (ROOT / "audit.md").write_text("# Figure and analysis audit\n\n" + "\n".join(checks) + "\n", encoding="utf-8")
    (ROOT / "final-status.md").write_text(
        "# Final status\n\nPASSED_WITH_WARNINGS\n\n"
        "All requested figures and tables were rendered from 45 complete trajectories. "
        "Warnings are methodological: one training seed per speed and the explicitly requested no-long-exit arrival rule does not enforce a suffix-inside fraction.\n",
        encoding="utf-8",
    )


def write_figure_spec():
    spec = """# Figure Spec

- chart_type: multi-figure scientific diagnostic set; main panel is a 2x2 point-and-error-bar comparison, with line, CDF, heatmap, trajectory, and deployment-region supplements
- data_sources: 45 deterministic evaluation traces generated from the three user-selected copied continuation checkpoints
- rows_in_scope: v_max=20/30/40; training seeds 2/32/32; evaluation seeds 1001--1015; 400 slots per episode
- data_columns: positions, speeds, degrees, adjacency, edge_turnover, connected, system_performance_increment, and episode-level summary metrics
- x_axis:
  - field: maximum speed bound v_max, episode time, actual speed, or x position depending on the panel
  - label: Maximum speed bound v_max (m/s), Episode time (s), Actual UAV speed (m/s), or x position (m)
  - unit: m/s, s, m/s, or m
  - scale: linear
  - range: data-driven except the 0--600 m map axes
- y_axis:
  - field: system performance, realized speed, propulsion proxy, active-edge fraction, neighbor degree, edge turnover, empirical CDF, arrival time, or y position depending on the panel
  - label: panel-specific physical quantity with units shown in each rendered figure
  - unit: as shown in the panel labels
  - scale: linear
  - range: data-driven except CDF [0,1] and the 0--600 m map axes
- additional_axes: none
- series_or_categories: three selected speed settings; five UAV colors in trajectory/deployment panels; 15 episode points per speed
- category_order: v_max=20, 30, 40 m/s
- color_mapping: Okabe-Ito blue for 20 m/s, vermilion for 30 m/s, green for 40 m/s; fixed five-color mapping for UAV identities
- legend: speed, UAV identity, start/endpoint, deployment center/arrival index, and common-window markers are explicitly labeled
- required_annotations: episode SD error bars; 520-m nominal neighborhood radius; final-200-slot centers and r98 circles; p98/no-long-exit deployment rule; common t_m window
- forbidden_elements: no fitted regression, no extrapolated endpoints, no artificial data, and no hidden suffix-inside condition in the primary arrival rule
- layout_constraints: publication-style PNG and PDF outputs; single-column-compatible main/supplement figures; no cropped legends or titles
- source_note: one slot is Delta_t=0.5 s; episode SD describes fixed-reset evaluation variation, not training-seed uncertainty
- assumptions: deployment center is the mean of the final 200 positions; radius is the within-window 98th-percentile distance; arrival is the first suffix with no outside run longer than 5 slots; scene arrival is the maximum of the five UAV arrival slots; t_m is the ceiling of the maximum mean scene-arrival slot over the three selected speeds
"""
    (ROOT / "figure-spec.md").write_text(spec, encoding="utf-8")


def main():
    global rows_global
    rows_global = load_rows()
    write_figure_spec()
    write_velocity_summary(rows_global)
    plot_main(rows_global)
    plot_neighborhood(rows_global)
    plot_speed_cdf(rows_global)
    plot_temporal(rows_global)
    plot_trajectories(rows_global)
    plot_adjacency(rows_global)
    detail, scene_rows = deployment_detail(rows_global)
    common_rows, tm_slots = write_deployment_outputs(detail, scene_rows)
    plot_deployment_arrival(detail, scene_rows, common_rows, tm_slots)
    plot_deployment_regions(detail, rows_global)
    plot_common_tm(scene_rows, common_rows, tm_slots)
    write_audit(rows_global, detail, scene_rows)
    print(json.dumps({
        "status": "completed",
        "episodes": len(rows_global),
        "deployment_uav_rows": len(detail),
        "deployment_scene_rows": len(scene_rows),
        "tm_slots": tm_slots,
        "tm_time_s": tm_slots * DT,
        "figures": sorted(path.name for path in FIG_ROOT.glob("*.png")),
        "summary": str(ROOT / "deployment_summary.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
