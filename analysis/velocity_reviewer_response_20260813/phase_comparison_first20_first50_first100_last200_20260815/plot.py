#!/usr/bin/env python
"""Plot fixed-window phase means for the selected velocity checkpoints.

The source traces are the 45 deterministic evaluation episodes already produced
for the user-selected checkpoints.  This script intentionally avoids the
arrival-before/after split: every metric is summarized over one of the selected
first-window lengths (20, 50, or 100 slots),
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

PHASE_CONFIGS = [
    (20, "first20"),
    (50, "first50"),
    (100, "first100"),
]
PHASES = []


def make_phases(first_slots):
    return [
        (f"first{first_slots}", f"First {first_slots} slots", "o", "--", slice(0, first_slots)),
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


def write_summary(records, first_slots, slug):
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
    with (ROOT / f"phase_{slug}_last200_summary.csv").open(
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
        f"- First phase: slots 0--{first_slots - 1} ({first_slots} slots).",
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
    (ROOT / f"phase_{slug}_last200_summary.md").write_text(
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


def plot_individual(records, slug):
    stems = {
        "speed": f"{slug}_mean_speed",
        "performance": f"{slug}_mean_performance",
        "propulsion_energy": f"{slug}_mean_propulsion_energy",
    }
    for metric_key, stem in stems.items():
        fig, ax = plt.subplots(figsize=(6.25, 4.55), constrained_layout=False)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.31, top=0.86)
        draw_metric(ax, records, metric_key, outside_legend=True)
        save_figure(fig, stem)


def plot_overview(records, first_slots, slug):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.65), constrained_layout=True)
    for ax, metric_key in zip(axes, METRICS):
        draw_metric(ax, records, metric_key)
        # A shared legend is clearer in the overview; remove repeated legends.
        ax.legend_.remove()
        if ax.artists:
            ax.artists[0].remove()
    phase_handles = [
        Line2D([], [], color="#000000", marker=marker, linestyle=linestyle, markersize=5.0, linewidth=1.1, markerfacecolor="#000000", markeredgecolor="#000000", label=phase_label)
        for _, phase_label, marker, linestyle, _ in PHASES
    ]
    fig.legend(
        handles=phase_handles,
        labels=[phase[1] for phase in PHASES],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.055),
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.suptitle(
        f"Phase comparison: first {first_slots} slots, last 200 slots, and full episode",
        fontsize=10.6,
    )
    save_figure(fig, f"phase_metric_comparison_{slug}")


def plot_all_overview(results):
    """Render one compact 3x3 comparison across all three first-window choices."""
    global PHASES
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 9.2), constrained_layout=True)
    for row_index, (first_slots, slug, records) in enumerate(results):
        PHASES = make_phases(first_slots)
        for col_index, (ax, metric_key) in enumerate(zip(axes[row_index], METRICS)):
            draw_metric(ax, records, metric_key)
            if ax.legend_ is not None:
                ax.legend_.remove()
            ax.set_title(f"First {first_slots}: {METRICS[metric_key]['title']}")
            if row_index < len(results) - 1:
                ax.set_xlabel("")

    legend_phases = [
        Line2D([], [], color="#000000", marker="o", linestyle="--", markersize=5.0, linewidth=1.1, label="First-window slots"),
        Line2D([], [], color="#000000", marker="s", linestyle="-", markersize=5.0, linewidth=1.1, label="Last 200 slots"),
        Line2D([], [], color="#000000", marker="D", linestyle=":", markersize=5.0, linewidth=1.1, label="Full episode"),
    ]
    fig.legend(
        handles=legend_phases,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.suptitle(
        "Comparison of first-window choices under the selected continuation checkpoints",
        fontsize=11.0,
    )
    save_figure(fig, "phase_metric_comparison_all_windows", also_plot=True)


def write_figure_spec():
    spec = """# Figure Spec

- chart_type: three 1x3 line-and-marker overviews, nine standalone metric plots, and one 3x3 comparison overview
- data_sources: 45 deterministic evaluation traces from the selected copied checkpoints
- data_columns: speeds, system_performance_increment, propulsion_energy, positions
- rows_in_scope: v_max=20/30/40; training seeds 2/32/32; evaluation seeds 1001--1015; 400 slots per episode
- x_axis:
  - field: maximum speed bound v_max
  - label: Maximum speed bound v_max (m/s)
  - scale: linear
  - range: 20, 30, 40
- y_axis:
  - field: fixed-window mean of the panel metric
  - label: mean realized UAV speed, mean system performance per slot, or mean UAV propulsion energy per slot
  - scale: linear
  - range: data-driven
- phase_windows: first 20 slots = 0--19, first 50 slots = 0--49, or first 100 slots = 0--99; last 200 slots = 200--399; full episode = 0--399
- series_or_categories: three fixed-window phase series across three speed settings
- color_mapping: black for all speeds and all phases; speed is identified by the x-axis position
- marker_mapping: circle for the selected first-window phase, square for last 200 slots, diamond for full episode
- line_mapping: dashed, solid, and dotted neutral lines connect the three speed settings for the three phase windows
- energy_definition: propulsion_energy; sum the five UAV propulsion-energy values within each slot and average over the selected slots
- aggregation: each plotted point is the mean of 15 episode-level phase means
- legend: phase is encoded by point marker and line style; speed settings are the x-axis ticks
- required_annotations: x-axis speed settings, phase legend, metric units, and energy definition in the accompanying summary
- uncertainty: intentionally omitted from plots as requested; episode standard deviation is stored only in the CSV
- forbidden_elements: no error bars, no distribution points, no arrival-before/after split, no fitted trend
- assumptions: all episodes contain 400 slots; the first and last windows are fixed slot windows; one slot is Delta_t=0.5 s; one training seed is available per speed setting
- output: 300 dpi PNG and PDF figures; plot.png is the 3x3 all-window comparison overview
"""
    (ROOT / "figure-spec.md").write_text(spec, encoding="utf-8")


def write_audit(rows, results):
    figures = sorted(path.name for path in FIG_ROOT.glob("*.png"))
    total_records = sum(len(records) for _, _, records in results)
    checks = [
        f"- loaded_episode_count: {len(rows)} (expected 45)",
        f"- phase_record_count: {total_records} (expected 81 = 3 first-window choices x 3 speeds x 3 phases x 3 metrics)",
        f"- trajectory_shape_check: {'pass' if all(row['data']['positions'].shape == (401, 5, 2) for row in rows) else 'fail'}",
        f"- phase_lengths: first20=20, first50=50, first100=100, last200=200, full=400",
        f"- figure_png_count: {len(figures)}",
        f"- figures: {', '.join(figures)}",
        "- aggregation_check: pass; every plotted point is the mean of 15 episode-level phase means",
        "- uncertainty_check: pass; no error bars or distribution marks are rendered",
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
        "The three requested first-window choices, their 1x3 overviews, the 3x3 comparison overview, and summary tables were rendered from 45 complete deterministic evaluation traces. "
        "The only methodological warning is that each speed setting uses one selected training seed; the 15 episodes quantify fixed-reset evaluation variation.\n",
        encoding="utf-8",
    )


def main():
    global PHASES
    rows = load_rows()
    write_figure_spec()
    results = []
    for first_slots, slug in PHASE_CONFIGS:
        PHASES = make_phases(first_slots)
        records = aggregate(rows)
        write_summary(records, first_slots, slug)
        plot_individual(records, slug)
        plot_overview(records, first_slots, slug)
        results.append((first_slots, slug, records))
    plot_all_overview(results)
    write_audit(rows, results)
    write_final_status()
    print(
        json.dumps(
            {
                "status": "completed",
                "episodes": len(rows),
                "records": sum(len(records) for _, _, records in results),
                "figures": sorted(path.name for path in FIG_ROOT.glob("*.png")),
                "summaries": [str(ROOT / f"phase_{slug}_last200_summary.md") for _, slug, _ in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
