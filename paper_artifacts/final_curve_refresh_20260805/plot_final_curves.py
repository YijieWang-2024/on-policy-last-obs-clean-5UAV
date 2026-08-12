from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TAG = "agent0/system_performance_true_all_GUs"
COLORS = {
    0: "#0072B2",
    200: "#E69F00",
    260: "#E69F00",
    400: "#009E73",
    520: "#009E73",
    600: "#CC79A7",
    1000: "#D55E00",
}
SEED_STYLES = {2: "-", 12: "--", 22: ":"}
SMOOTH_WINDOW = 21


def radius_from_name(name: str) -> int:
    for radius in (1000, 600, 520, 400, 260, 200, 0):
        if f"_R{radius}_" in name or name.endswith(f"_R{radius}"):
            return radius
    raise ValueError(f"cannot infer radius from {name}")


def seed_from_name(name: str) -> int:
    match = re.search(r"_seed(\d+)(?:_|$)", name)
    return int(match.group(1)) if match else 0


def rolling(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return values.copy()
    half = min(window // 2, max(0, (len(values) - 1) // 2))
    return np.asarray(
        [values[max(0, i - half) : min(len(values), i + half + 1)].mean() for i in range(len(values))],
        dtype=float,
    )


def load_tensorboard_runs(root: Path, name_token: str) -> list[dict]:
    records = []
    event_paths = root.rglob("events.out.tfevents.*")
    for event_path in sorted(event_paths):
        run_name = event_path.parent.name
        if name_token not in run_name:
            continue
        try:
            radius = radius_from_name(run_name)
            seed = seed_from_name(run_name)
        except ValueError:
            continue
        acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        acc.Reload()
        if TAG not in acc.Tags().get("scalars", []):
            continue
        events = acc.Scalars(TAG)
        by_step = {int(event.step): float(event.value) for event in events}
        for step, value in sorted(by_step.items()):
            records.append(
                {
                    "radius": radius,
                    "seed": seed,
                    "step": step,
                    "value": value,
                    "run": run_name,
                }
            )
    return records


def load_json_runs(path: Path, names: list[str]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for name in names:
        radius = radius_from_name(name)
        seed = seed_from_name(name)
        for step, value in payload["experiments"][name]["metrics"][TAG]:
            records.append(
                {
                    "radius": radius,
                    "seed": seed,
                    "step": int(step),
                    "value": float(value),
                    "run": name,
                }
            )
    return records


def load_snapshot_csv_group(path: Path, group_name: str) -> list[dict]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != TAG:
                continue
            run_name = row["experiment"]
            group = "FixedDiag" if "fixeddiag" in run_name else "Diag4"
            if group != group_name:
                continue
            records.append(
                {
                    "radius": radius_from_name(run_name),
                    "seed": seed_from_name(run_name),
                    "step": int(float(row["step"])),
                    "value": float(row["value"]),
                    "run": run_name,
                }
            )
    return records


def load_long_csv_group(path: Path, group_name: str) -> list[dict]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "true60" or row["group"] != group_name:
                continue
            records.append(
                {
                    "radius": int(row["radius"]),
                    "seed": 2,
                    "step": int(float(row["step"])),
                    "value": float(row["value"]),
                    "run": f"{group_name}_R{row['radius']}_seed2",
                }
            )
    return records


def grouped(records: list[dict]) -> dict[tuple[int, int], list[dict]]:
    result = defaultdict(list)
    for record in records:
        result[(record["radius"], record["seed"])].append(record)
    for key in result:
        result[key].sort(key=lambda item: item["step"])
    return dict(result)


def endpoint_rows(group_name: str, records: list[dict]) -> list[dict]:
    rows = []
    for (radius, seed), run_records in sorted(grouped(records).items()):
        values = np.asarray([row["value"] for row in run_records], dtype=float)
        rows.append(
            {
                "group": group_name,
                "radius": radius,
                "seed": seed,
                "last_step": run_records[-1]["step"],
                "last_value": float(values[-1]),
                "tail25_mean": float(values[-25:].mean()),
                "tail100_mean": float(values[-100:].mean()),
                "points": len(values),
                "run": run_records[-1]["run"],
            }
        )
    return rows


def plot_group(group_name: str, records: list[dict], title: str, output_name: str) -> list[dict]:
    by_run = grouped(records)
    if not by_run:
        raise RuntimeError(f"No records for {group_name}")
    radii = sorted({radius for radius, _ in by_run})
    seeds = sorted({seed for _, seed in by_run})
    fig, (ax_curve, ax_tail) = plt.subplots(
        2, 1, figsize=(12.5, 9.0), gridspec_kw={"height_ratios": [2.5, 1.15]}, constrained_layout=True
    )

    for (radius, seed), run_records in sorted(by_run.items()):
        x = np.asarray([row["step"] for row in run_records], dtype=float) / 1e6
        y = np.asarray([row["value"] for row in run_records], dtype=float) / 1e3
        color = COLORS[radius]
        linestyle = SEED_STYLES.get(seed, "-")
        label = f"R{radius} / seed {seed}" if len(seeds) > 1 else f"R{radius}"
        ax_curve.plot(x, y, color=color, alpha=0.16, linewidth=0.65)
        ax_curve.plot(x, rolling(y), color=color, linestyle=linestyle, linewidth=1.9, label=label)
        ax_curve.scatter(x[-1], y[-1], color=color, s=24, zorder=4)
        ax_curve.annotate(
            f"{x[-1]:.2f}M",
            (x[-1], y[-1]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
        )

    last_step = max(row["step"] for row in records)
    ax_curve.set_xlim(0, last_step / 1e6 + 0.8)
    ax_curve.set_xlabel("Training environment steps (millions)")
    ax_curve.set_ylabel("True60 (×10³)")
    ax_curve.set_title(
        f"{title}\nfull logged traces; each line ends at its own last saved event",
        pad=10,
    )
    ax_curve.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.legend(ncol=min(4, max(1, len(by_run))), frameon=False, fontsize=8.5, loc="lower right")
    ax_curve.text(
        0.01,
        0.02,
        "faint = raw event values; solid/dashed/dotted = 21-event centered rolling mean; no common-step truncation",
        transform=ax_curve.transAxes,
        fontsize=8.2,
        color="#444444",
    )

    rows = endpoint_rows(group_name, records)
    positions = np.arange(len(radii), dtype=float)
    width = 0.78 / max(1, len(seeds))
    for index, seed in enumerate(seeds):
        values = []
        x_values = []
        for radius_index, radius in enumerate(radii):
            matches = [row for row in rows if row["radius"] == radius and row["seed"] == seed]
            if not matches:
                continue
            values.append(matches[0]["tail100_mean"] / 1e3)
            x_values.append(positions[radius_index] + (index - (len(seeds) - 1) / 2) * width)
        bars = ax_tail.bar(x_values, values, width=width, color=[COLORS[r] for r in radii], alpha=0.78, label=f"seed {seed}")
        for bar, value in zip(bars, values):
            ax_tail.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.012, f"{value:.1f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax_tail.set_xticks(positions, [f"R{radius}" for radius in radii])
    ax_tail.set_ylabel("Final tail100 True60 (×10³)")
    ax_tail.set_xlabel("Communication radius")
    ax_tail.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax_tail.spines[["top", "right"]].set_visible(False)
    if len(seeds) > 1:
        ax_tail.legend(frameon=False, ncol=min(3, len(seeds)), loc="upper left")
    fig.savefig(OUT / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def main() -> None:
    all_rows = []
    random_root = REPO / "paper_artifacts" / "randomlayout700_v2_final_20260804" / "events"
    moving_root = REPO / "paper_artifacts" / "randomlayout_moving_20260804" / "events"
    random_records = load_tensorboard_runs(random_root, "randomlayout700v2_staggered_gated_localA_task")
    moving_records = load_tensorboard_runs(moving_root, "movinghotspot700_gated_localA_task")
    all_rows.extend(plot_group("RandomLayout700-v2 Full template12", random_records, "RandomLayout700-v2 Full template12", "randomlayout700_full_final.png"))
    all_rows.extend(plot_group("MovingHotspot700", moving_records, "MovingHotspot700", "movinghotspot700_full_final.png"))

    raw_json = REPO.parent / "remote_randomlayout700_analysis_20260805" / "true60_metrics_raw.json"
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    curriculum_names = [name for name in payload["experiments"] if "uavcurriculum12v2" in name]
    subset_names = [name for name in payload["experiments"] if "template12_subset29" in name]
    curriculum_records = load_json_runs(raw_json, curriculum_names)
    subset_records = load_json_runs(raw_json, subset_names)
    all_rows.extend(plot_group("RandomLayout700-v2 Full template12 + UAV curriculum", curriculum_records, "Full template12 + UAV reset curriculum", "uavcurriculum12v2_full_final.png"))
    all_rows.extend(plot_group("RandomLayout700-v2 Subset29", subset_records, "Template12 subset [2,9]", "subset29_full_final.png"))

    snapshot_csv = REPO / "analysis" / "remote9001_randomlayout700_live_20260804" / "remote_scalars.csv"
    diag_csv = REPO / "analysis" / "remote9001_randomlayout700_uavcurriculum_vs_diag4_20260805" / "all_scalars.csv"
    diag_records = load_long_csv_group(diag_csv, "Diag4_no_curriculum")
    fixed_records = load_snapshot_csv_group(snapshot_csv, "FixedDiag")
    all_rows.extend(plot_group("700m Diag4", diag_records, "700m Diag4 (four layouts)", "diag4_700m_full_final.png"))
    all_rows.extend(plot_group("FixedDiag", fixed_records, "FixedDiag", "fixeddiag_full_final.png"))

    with (OUT / "final_curve_endpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    manifest = {
        "metric": TAG,
        "smoothing": "21-event centered rolling mean with shortened edge windows",
        "curve_contract": "every available run is drawn through its own last TensorBoard event; no common-step truncation",
        "sources": {
            "randomlayout700_full": str(random_root),
            "movinghotspot700": str(moving_root),
            "curriculum_and_subset29": str(raw_json),
            "diag4": str(diag_csv),
            "fixeddiag": str(snapshot_csv),
        },
        "figures": [
            "randomlayout700_full_final.png",
            "movinghotspot700_full_final.png",
            "uavcurriculum12v2_full_final.png",
            "subset29_full_final.png",
            "diag4_700m_full_final.png",
            "fixeddiag_full_final.png",
        ],
    }
    (OUT / "final_curve_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
