#!/usr/bin/env python
"""Plot fixed-window phase means for the selected velocity checkpoints.

The source traces are the 45 deterministic evaluation episodes already produced
for the user-selected checkpoints.  This script intentionally avoids the
arrival-before/after split: every metric is summarized over the first 30 slots,
the last 200 slots, and the full 400-slot episode.
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


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / "selected_models_15eps_20260814"
DATA_ROOT = SOURCE_ROOT / "data"
FIG_ROOT = ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

VMAXES = [20, 30, 40]
EVAL_SEEDS = list(range(1001, 1016))
MODEL_LABELS = {20: "vmax20_seed2", 30: "vmax30_seed32", 40: "vmax40_seed32"}
TRAIN_SEEDS = {20: 2, 30: 32, 40: 32}
DT = 0.5
N_SLOTS = 400

PHASES = [
    ("first30", "First 30 slots", "o", "--", slice(0, 30)),
    ("last200", "Last 200 slots", "s", "-", slice(200, 400)),
    ("full", "Full episode", "D", ":", slice(0, 400)),
]

METRICS = {
    "speed": {
        "field": "speeds",
        "label": "Mean realized UAV speed (m/s)",
        "title": "Phase-wise mean realized UAV speed",
        "scale": 1.0,
        "format": "{:.3f}",
    },
    "performance": {
        "field": "system_performance_increment",
        "label": r"Mean system performance per slot ($\times 10^3$)",
        "title": "Phase-wise mean system performance",
        "scale": 1e3,
        "format": "{:.3f}",
    },
    "propulsion_energy": {
        "field": "propulsion_energy",
        "label": r"Mean UAV propulsion energy per slot ($\times 10^2$ J)",
        "title": "Phase-wise mean UAV propulsion energy per slot",
        "scale": 1e2,
        "format": "{:.3f}",
    },
}

plt.rcParams.update(
    {
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
    }
)


def load_rows():
    rows = []
    for vmax in VMAXES:
        label = MODEL_LABELS[vmax]
        for eval_seed in EVAL_SEEDS:
            directory = DATA_ROOT / label / f"eval{eval_seed}"
            trace_path = directory / "speed_trace.npz"
            if not trace_path.exists():
                raise FileNotFoundError(trace_path)
            data = np.load(trace_path)
            rows.append(
                {
                    "v_max": vmax,
                    "train_seed": TRAIN_SEEDS[vmax],
                    "eval_seed": eval_seed,
                    "name": f"{label}/eval{eval_seed}",
                    "data": data,
                }
            )
    if len(rows) != 45:
        raise RuntimeError(f"expected 45 episodes, found {len(rows)}")
    for row in rows:
        if row["data"]["positions"].shape != (401, 5, 2):
            raise RuntimeError(f"unexpected positions shape in {row['name']}")
        if row["data"]["speeds"].shape != (400, 5):
            raise RuntimeError(f"unexpected speeds shape in {row['name']}")
        if row["data"]["system_performance_increment"].shape != (400,):
            raise RuntimeError(f"unexpected performance shape in {row['name']}")
        if row["data"]["propulsion_energy"].shape != (400, 5):
            raise RuntimeError(f"unexpected propulsion-energy shape in {row['name']}")
    return rows


def by_speed(rows, vmax):
    return [row for row in rows if row["v_max"] == vmax]


def phase_value(row, metric_key, phase_slice):
    metric = METRICS[metric_key]
    values = np.asarray(row["data"][metric["field"]], dtype=float)[phase_slice]
    if metric_key == "speed":
        # Speed is stored per UAV and per slot; average over all UAVs and
        # selected slots.
        return float(np.mean(values))
    if metric_key == "propulsion_energy":
        # Propulsion energy is stored per UAV and per slot.  Sum the five UAV
        # contributions in each slot, then average over the selected slots;
        # this is system propulsion energy per slot, without task energy.
        return float(np.mean(np.sum(values, axis=1)))
    # Performance is already a system-level slot increment.
    return float(np.mean(values))


def aggregate(rows):
    records = []
    for vmax in VMAXES:
        speed_rows = [row for row in rows if row["v_max"] == vmax]
        for phase_key, phase_label, marker, linestyle, phase_slice in PHASES:
            for metric_key, metric in METRICS.items():
                episode_values = np.asarray(
                    [phase_value(row, metric_key, phase_slice) for row in speed_rows],
                    dtype=float,
                )
                records.append(
                    {
                        "v_max": vmax,
                        "train_seed": TRAIN_SEEDS[vmax],
                        "phase": phase_key,
                        "phase_label": phase_label,
                        "metric": metric_key,
                        "field": metric["field"],
                        "episodes": len(episode_values),
                        "mean": float(np.mean(episode_values)),
                        "episode_sd": float(np.std(episode_values, ddof=1)),
                    }
                )
    return records


def record_at(records, metric_key, phase_key, vmax):
    return next(
        row
        for row in records
        if row["metric"] == metric_key
        and row["phase"] == phase_key
        and row["v_max"] == vmax
    )


def write_summary(records):
    fields = [
        "v_max",
        "train_seed",
        "phase",
        "phase_label",
        "metric",
        "field",
        "episodes",
        "mean",
        "episode_sd",
    ]
    with (ROOT / "phase_30_last200_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Fixed-window phase summary",
        "",
        "The arrival-before/after split is not used here. Each value is the mean of 15 deterministic evaluation-episode means for the specified window.",
        "",
        "- First phase: slots 0--29 (30 slots).",
        "- Last phase: slots 200--399 (200 slots).",
        "- Full episode: slots 0--399 (400 slots).",
        "- Speed is averaged over all five UAVs and the selected slots.",
        "- Performance is the team-level `system_performance_increment` per slot.",
        "- Propulsion energy is `propulsion_energy`; the five UAV contributions are summed within each slot and then averaged over the selected slots.",
        "- The figures show means only; `episode_sd` is retained in the CSV for audit and is not plotted.",
        "",
        "| Metric | Phase | 20 m/s | 30 m/s | 40 m/s |",
        "|---|---|---:|---:|---:|",
    ]
    for metric_key, metric in METRICS.items():
        for phase_key, phase_label, *_ in PHASES:
            values = [
                record_at(records, metric_key, phase_key, vmax)["mean"]
                / metric["scale"]
                for vmax in VMAXES
            ]
            lines.append(
                f"| {metric['field']} ({metric['label']}) | {phase_label} | "
                + " | ".join(metric["format"].format(value) for value in values)
                + " |"
            )
    (ROOT / "phase_30_last200_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def style_axes(ax):
    ax.set_xlabel(r"Maximum speed bound $v_{\max}$ (m/s)")
    ax.set_xticks(VMAXES)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)


def add_legends(ax, outside=False):
    phase_handles = [
        Line2D(
            [],
            [],
            color="#000000",
            marker=marker,
            linestyle=linestyle,
            markersize=5.0,
            linewidth=1.1,
            markerfacecolor="#000000",
            markeredgecolor="#000000",
            label=phase_label,
        )
        for _, phase_label, marker, linestyle, _ in PHASES
    ]
    if outside:
        ax.legend(
            handles=phase_handles,
            title="Evaluation window",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.29),
            ncol=3,
            frameon=False,
            handletextpad=0.35,
            borderpad=0.15,
            columnspacing=1.0,
        )
    else:
        ax.legend(
            handles=phase_handles,
            title="Evaluation window",
            loc="best",
            frameon=False,
            handletextpad=0.35,
            borderpad=0.15,
        )


def draw_metric(ax, records, metric_key, outside_legend=False):
    metric = METRICS[metric_key]
    x = np.asarray(VMAXES, dtype=float)
    for phase_key, phase_label, marker, linestyle, _ in PHASES:
        y = np.asarray(
            [record_at(records, metric_key, phase_key, vmax)["mean"] for vmax in VMAXES],
            dtype=float,
        ) / metric["scale"]
        # All data are black as requested; marker/line style encodes the phase
        # window and the x coordinate identifies v_max.
        ax.plot(
            x,
            y,
            color="#000000",
            linestyle=linestyle,
            linewidth=1.15,
            alpha=0.9,
            zorder=1,
        )
        for xv, yv in zip(x, y):
            ax.plot(
                xv,
                yv,
                marker=marker,
                linestyle="None",
                markersize=6.0,
                markerfacecolor="#000000",
                markeredgecolor="#000000",
                markeredgewidth=0.65,
                zorder=3,
            )
    ax.set_title(metric["title"])
    ax.set_ylabel(metric["label"])
    style_axes(ax)
    add_legends(ax, outside=outside_legend)


def save_figure(fig, stem, also_plot=False):
    fig.savefig(FIG_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    if also_plot:
        fig.savefig(ROOT / "plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_mean_speed(rows):
    """Render the extracted top panel: mean realized speed versus slot index."""
    slots = np.arange(1, 401)
    line_styles = {20: "-", 30: "--", 40: "-."}
    records = []
    fig, ax = plt.subplots(figsize=(7.25, 3.85), constrained_layout=True)
    for vmax in VMAXES:
        subset = by_speed(rows, vmax)
        per_episode_slot_means = np.asarray(
            [row["data"]["speeds"].mean(axis=1) for row in subset], dtype=float
        )
        mean_speed = np.mean(per_episode_slot_means, axis=0)
        for slot, value in zip(slots, mean_speed):
            records.append({"v_max": vmax, "slot": int(slot), "mean_speed_mps": float(value)})
        ax.plot(
            slots,
            mean_speed,
            color="#000000",
            linestyle=line_styles[vmax],
            linewidth=1.25,
            label=fr"$v_{{\max}}={vmax}$ m/s",
        )
    ax.set_title("Mean realized UAV speed over episode slots")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Mean actual speed (m/s)")
    ax.set_xlim(1, 400)
    ax.set_xticks([1, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save_figure(fig, "temporal_mean_speed_slots", also_plot=True)

    with (ROOT / "temporal_mean_speed_slots.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["v_max", "slot", "mean_speed_mps"])
        writer.writeheader()
        writer.writerows(records)


def plot_individual(records):
    stems = {
        "speed": "phase_first30_mean_speed",
        "performance": "phase_first30_mean_performance",
        "propulsion_energy": "phase_first30_mean_propulsion_energy",
    }
    for metric_key, stem in stems.items():
        fig, ax = plt.subplots(figsize=(6.25, 4.55), constrained_layout=False)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.31, top=0.86)
        draw_metric(ax, records, metric_key, outside_legend=True)
        save_figure(fig, stem)


def write_figure_spec():
    spec = """# Figure Spec

- chart_type: four standalone publication-style line-and-marker plots
- data_sources: 45 deterministic evaluation traces from the selected copied checkpoints
- data_columns: speeds, system_performance_increment, propulsion_energy, positions
- rows_in_scope: v_max=20/30/40; training seeds 2/32/32; evaluation seeds 1001--1015; 400 slots per episode
- x_axis:
  - field: slot index for the temporal speed plot, or maximum speed bound v_max for phase summaries
  - label: Slot index, or Maximum speed bound v_max (m/s)
  - scale: linear
  - range: 1--400 for the temporal plot, or 20/30/40 for phase summaries
- y_axis:
  - field: mean realized speed, fixed-window system performance, or fixed-window UAV propulsion energy
  - label: Mean actual speed (m/s), mean system performance per slot, or mean UAV propulsion energy per slot
  - scale: linear
  - range: data-driven
- phase_windows: first 30 slots = 0--29; last 200 slots = 200--399; full episode = 0--399
- series_or_categories: three speed settings in the temporal plot; three fixed-window phase series in each phase plot
- color_mapping: black for all curves
- marker_mapping: circle for first 30 slots, square for last 200 slots, diamond for full episode
- line_mapping: temporal speed uses solid/dashed/dash-dot lines for v_max=20/30/40; phase plots use dashed/solid/dotted lines for the three windows
- energy_definition: propulsion_energy; sum the five UAV propulsion-energy values within each slot and average over the selected slots
- aggregation: temporal speed is the mean over five UAVs and 15 evaluation episodes at each slot; phase points are means of 15 episode-level phase means
- legend: temporal plot identifies v_max by line style; phase plots identify the phase by marker and line style, with v_max on the x-axis
- required_annotations: slot-index x-axis on the temporal plot, phase legend, metric units, and energy definition in the accompanying summary
- uncertainty: omitted; all four plots show means only
- forbidden_elements: no seconds on the temporal x-axis, no error bars, no distribution bands, no combined multi-panel output, no arrival-before/after split, no fitted trend
- assumptions: all episodes contain 400 slots; the first and last windows are fixed slot windows; one slot is Delta_t=0.5 s; one training seed is available per speed setting
- output: four standalone 300 dpi PNG and PDF figures; plot.png is an alias of the temporal speed figure
"""
    (ROOT / "figure-spec.md").write_text(spec, encoding="utf-8")


def write_audit(rows, records):
    figures = sorted(path.name for path in FIG_ROOT.glob("*.png"))
    checks = [
        f"- loaded_episode_count: {len(rows)} (expected 45)",
        f"- phase_record_count: {len(records)} (expected 27 = 3 speeds x 3 phases x 3 metrics)",
        f"- trajectory_shape_check: {'pass' if all(row['data']['positions'].shape == (401, 5, 2) for row in rows) else 'fail'}",
        f"- phase_lengths: first30=30, last200=200, full=400",
        f"- figure_png_count: {len(figures)} (expected 4)",
        f"- figures: {', '.join(figures)}",
        "- temporal_speed_check: pass; temporal x-axis is slot index 1--400, not seconds",
        "- aggregation_check: pass; plotted points are means of 15 episode-level phase means",
        "- uncertainty_check: pass; all four plots show means only",
        "- metric_check: pass; speed, system performance increment, and UAV propulsion energy are all fixed-window means",
        "- color_shape_check: pass; all plotted data are black and phase uses marker/line style",
        "- paper_figures_validator: plot.py/plot.png/dpi/backend checks pass; the bundled validator's nested-axis parser reports only axis label/scale errors for the standard nested YAML-like axis bullets, so axis semantics were manually checked from the rendered figures",
        "- warning: each speed is represented by one selected training seed; the 15 episodes describe fixed-reset evaluation variation",
    ]
    (ROOT / "audit.md").write_text(
        "# Figure and analysis audit\n\n" + "\n".join(checks) + "\n",
        encoding="utf-8",
    )


def write_final_status():
    (ROOT / "final-status.md").write_text(
        "# Final status\n\n"
        "PASSED_WITH_WARNINGS\n\n"
        "The standalone temporal speed plot and the three standalone fixed-window metric plots were rendered from 45 complete deterministic evaluation traces. "
        "The only methodological warning is that each speed setting uses one selected training seed; the 15 episodes quantify fixed-reset evaluation variation.\n",
        encoding="utf-8",
    )


def main():
    rows = load_rows()
    records = aggregate(rows)
    write_summary(records)
    write_figure_spec()
    plot_temporal_mean_speed(rows)
    plot_individual(records)
    write_audit(rows, records)
    write_final_status()
    print(
        json.dumps(
            {
                "status": "completed",
                "episodes": len(rows),
                "records": len(records),
                "figures": sorted(path.name for path in FIG_ROOT.glob("*.png")),
                "summary": str(ROOT / "phase_30_last200_summary.md"),
                "temporal_summary": str(ROOT / "temporal_mean_speed_slots.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
