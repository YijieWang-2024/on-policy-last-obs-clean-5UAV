from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METRICS = [
    ("true60", "True60", lambda value: f"{value/1000:.1f}k", True),
    ("equivalent60", "Eq60", lambda value: f"{value/1000:.1f}k", True),
    ("system_performance", "System\nperformance", lambda value: f"{value/1000:.1f}k", True),
    ("admission", "Admission", lambda value: f"{100*value:.1f}%", True),
    ("upper_admission", "Upper\nadmission", lambda value: f"{100*value:.1f}%", True),
    ("completion", "Completion", lambda value: f"{100*value:.1f}%", True),
    ("active_mds", "Average\nactive MDs", lambda value: f"{value:.1f}", True),
    ("projection", "Action\nprojection", lambda value: f"{100*value:.1f}%", False),
]

rows = []
with (ROOT / "tail100_metrics.csv").open(encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        converted = dict(row)
        converted["max_step"] = float(row["max_step"])
        for metric, *_ in METRICS:
            converted[metric] = float(row[metric]) if row[metric] else np.nan
        rows.append(converted)
rows.sort(key=lambda row: row["true60"], reverse=True)

values = np.asarray([[row[metric] for metric, *_ in METRICS] for row in rows], dtype=float)
scores = np.empty_like(values)
for column, (_, _, _, higher_is_better) in enumerate(METRICS):
    data = values[:, column]
    finite = np.isfinite(data)
    span = np.nanmax(data) - np.nanmin(data)
    normalized = np.full_like(data, np.nan)
    normalized[finite] = 1.0 if span == 0 else (data[finite] - np.nanmin(data)) / span
    scores[:, column] = normalized if higher_is_better else 1.0 - normalized

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titlepad": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

fig, ax = plt.subplots(figsize=(9.2, 4.8))
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#D9D9D9")
image = ax.imshow(np.ma.masked_invalid(scores), cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(np.arange(len(METRICS)), [label for _, label, *_ in METRICS])
row_labels = [f"{row['experiment']}  [{row['max_step']/1e6:.1f}M]" for row in rows]
ax.set_yticks(np.arange(len(rows)), row_labels)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

for row_index, row in enumerate(rows):
    for column, (metric, _, formatter, _) in enumerate(METRICS):
        score = scores[row_index, column]
        if not np.isfinite(score):
            text, color = "—", "0.35"
        else:
            text = formatter(row[metric])
            color = "white" if score < 0.35 or score > 0.78 else "black"
        ax.text(column, row_index, text, ha="center", va="center", color=color, fontsize=8.3)

ax.set_title("Tail-100 means: exact values and within-column relative ranking", pad=26)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
ax.tick_params(which="minor", bottom=False, left=False)

colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
colorbar.set_label("Relative desirability within each column")
colorbar.set_ticks([0, 0.5, 1])
colorbar.set_ticklabels(["lower", "middle", "higher"])
fig.text(
    0.5,
    0.025,
    "Higher is better except action projection (lower is better). Em dash means the older run did not log that diagnostic.",
    ha="center",
    fontsize=8,
    color="0.35",
)
fig.subplots_adjust(top=0.79, bottom=0.12, left=0.31, right=0.93)
fig.savefig(HERE / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
