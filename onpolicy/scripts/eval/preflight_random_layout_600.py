"""Training-free value-of-communication preflight for RandomLayout v1.

The diagnostic uses a pre-generated candidate-arrival trace shared by all
communication radii.  Controllers differ only in the range over which their
stored hotspot evidence is exchanged.  It intentionally evaluates discovery,
admission, deployment time, and flight distance rather than PPO/resource
allocation quality.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAP_SIZE = 600.0
COVER_RADIUS = 120.0
MAX_STEP = 30.0 * 0.5
EPISODE_LENGTH = 400
ABSENCE_SLOTS = 4

LAYOUTS = np.array([
    [[0, 175, 0, 175], [200, 600, 200, 600]],
    [[425, 600, 0, 175], [0, 400, 200, 600]],
    [[0, 175, 425, 600], [200, 600, 0, 400]],
    [[425, 600, 425, 600], [0, 400, 0, 400]],
], dtype=np.float64)

# Bottom, top, left, right.  The small region never intersects these arms;
# the large region occupies exactly two adjacent arms.
ARM_BOUNDS = np.array([
    [200, 400, 0, 175],
    [200, 400, 425, 600],
    [0, 175, 200, 400],
    [425, 600, 200, 400],
], dtype=np.float64)
ARM_CENTERS = np.column_stack((
    (ARM_BOUNDS[:, 0] + ARM_BOUNDS[:, 1]) / 2,
    (ARM_BOUNDS[:, 2] + ARM_BOUNDS[:, 3]) / 2,
))
LAYOUT_ACTIVE_ARMS = (
    frozenset((1, 3)),  # large upper-right
    frozenset((1, 2)),  # large upper-left
    frozenset((0, 3)),  # large lower-right
    frozenset((0, 2)),  # large lower-left
)
INITIAL_POSITIONS = np.array([
    [200, 525], [110, 70], [220, 70], [330, 70], [440, 70]
], dtype=np.float64)


def sample_candidates(rng, layout_ids, steps):
    episodes = len(layout_ids)
    points = np.empty((episodes, steps, 6, 2), dtype=np.float32)
    for region_id, count, start in ((0, 1, 0), (1, 5, 1)):
        bounds = LAYOUTS[layout_ids, region_id]
        u = rng.random((episodes, steps, count, 2))
        lo = bounds[:, None, None, (0, 2)]
        hi = bounds[:, None, None, (1, 3)]
        points[:, :, start:start + count] = lo + u * (hi - lo)
    return points


def inside_bounds(points, bounds):
    return (
        (points[..., 0] >= bounds[0])
        & (points[..., 0] <= bounds[1])
        & (points[..., 1] >= bounds[2])
        & (points[..., 1] <= bounds[3])
    )


def service_targets(layout_id):
    small, large = LAYOUTS[layout_id]
    targets = [[(small[0] + small[1]) / 2, (small[2] + small[3]) / 2]]
    xs = (large[0] + 100, large[0] + 300)
    ys = (large[2] + 100, large[2] + 300)
    targets.extend((x, y) for y in ys for x in xs)
    return np.asarray(targets, dtype=np.float64)


def assigned_targets_by_layout():
    result = np.empty((4, 5, 2), dtype=np.float64)
    for layout_id in range(4):
        targets = service_targets(layout_id)
        best_cost = np.inf
        best = None
        for permutation in itertools.permutations(range(5)):
            assigned = targets[np.asarray(permutation)]
            cost = np.linalg.norm(INITIAL_POSITIONS - assigned, axis=1).sum()
            if cost < best_cost:
                best_cost = cost
                best = assigned
        result[layout_id] = best
    return result


ASSIGNED_TARGETS = assigned_targets_by_layout()


def configure_map_size(map_size):
    """Scale only the map separation; preserve region sizes and all dynamics."""
    global MAP_SIZE, LAYOUTS, ARM_BOUNDS, ARM_CENTERS, ASSIGNED_TARGETS
    MAP_SIZE = float(map_size)
    small_far = MAP_SIZE - 175.0
    large_near = MAP_SIZE - 400.0
    LAYOUTS = np.array([
        [[0, 175, 0, 175], [large_near, MAP_SIZE, large_near, MAP_SIZE]],
        [[small_far, MAP_SIZE, 0, 175], [0, 400, large_near, MAP_SIZE]],
        [[0, 175, small_far, MAP_SIZE], [large_near, MAP_SIZE, 0, 400]],
        [[small_far, MAP_SIZE, small_far, MAP_SIZE], [0, 400, 0, 400]],
    ], dtype=np.float64)
    ARM_BOUNDS = np.array([
        [large_near, large_near + 200, 0, 175],
        [large_near, large_near + 200, small_far, MAP_SIZE],
        [0, 175, large_near, large_near + 200],
        [small_far, MAP_SIZE, large_near, large_near + 200],
    ], dtype=np.float64)
    ARM_CENTERS = np.column_stack((
        (ARM_BOUNDS[:, 0] + ARM_BOUNDS[:, 1]) / 2,
        (ARM_BOUNDS[:, 2] + ARM_BOUNDS[:, 3]) / 2,
    ))
    ASSIGNED_TARGETS = assigned_targets_by_layout()


def configure_initial_positions(mode):
    """Select the fixed-reset geometry without changing any other contract."""
    global INITIAL_POSITIONS, ASSIGNED_TARGETS
    if mode == "legacy":
        INITIAL_POSITIONS = np.array([
            [200, 525], [110, 70], [220, 70], [330, 70], [440, 70]
        ], dtype=np.float64)
    elif mode == "symmetric_cross":
        center = MAP_SIZE / 2.0
        offset = 200.0
        INITIAL_POSITIONS = np.array([
            [center, center],
            [center - offset, center],
            [center + offset, center],
            [center, center - offset],
            [center, center + offset],
        ], dtype=np.float64)
    else:
        raise ValueError(f"unknown initial layout: {mode}")
    ASSIGNED_TARGETS = assigned_targets_by_layout()


def move_toward(positions, targets):
    delta = targets - positions
    distance = np.linalg.norm(delta, axis=1)
    travel = np.minimum(distance, MAX_STEP)
    nonzero = distance > 1e-12
    positions[nonzero] += delta[nonzero] / distance[nonzero, None] * travel[nonzero, None]
    return travel


def identify_layout(status):
    compatible = []
    for layout_id, active_arms in enumerate(LAYOUT_ACTIVE_ARMS):
        valid = True
        for arm_id, observed in enumerate(status):
            expected = 1 if arm_id in active_arms else 0
            if observed >= 0 and observed != expected:
                valid = False
                break
        if valid:
            compatible.append(layout_id)
    return compatible[0] if len(compatible) == 1 else -1


def merge_neighbor_evidence(status, distances, radius):
    if radius <= 0:
        return status
    adjacent = distances <= radius
    old = status.copy()
    merged = old.copy()
    for receiver in range(5):
        senders = np.flatnonzero(adjacent[receiver])
        for arm_id in range(4):
            values = old[senders, arm_id]
            if np.any(values == 1):
                merged[receiver, arm_id] = 1
            elif np.any(values == 0):
                merged[receiver, arm_id] = 0
    return merged


def simulate_episode(
    layout_id, candidates, radius, search_orders, relay_received_evidence=True
):
    positions = INITIAL_POSITIONS.copy()
    status = np.full((5, 4), -1, dtype=np.int8)
    exposure = np.zeros((5, 4), dtype=np.int16)
    search_pointer = np.zeros(5, dtype=np.int8)
    known_layout = np.full(5, -1, dtype=np.int8)
    identified_at = np.full(5, EPISODE_LENGTH + 1, dtype=np.int16)
    target = ASSIGNED_TARGETS[layout_id]
    admitted = 0
    flight_distance = 0.0
    deployment_time = EPISODE_LENGTH + 1
    four_known_time = EPISODE_LENGTH + 1

    for step in range(EPISODE_LENGTH):
        desired = np.empty((5, 2), dtype=np.float64)
        for agent in range(5):
            if known_layout[agent] >= 0:
                desired[agent] = ASSIGNED_TARGETS[known_layout[agent], agent]
            else:
                while (
                    search_pointer[agent] < 4
                    and status[agent, search_orders[agent, search_pointer[agent]]] >= 0
                ):
                    search_pointer[agent] += 1
                arm = search_orders[agent, min(search_pointer[agent], 3)]
                desired[agent] = ARM_CENTERS[arm]
        flight_distance += move_toward(positions, desired).sum()

        distance_to_candidates = np.linalg.norm(
            positions[:, None, :] - candidates[step][None, :, :], axis=2
        )
        seen = distance_to_candidates <= COVER_RADIUS
        admitted += int(np.any(seen, axis=0).sum())

        for agent in range(5):
            if known_layout[agent] >= 0:
                continue
            arm = search_orders[agent, min(search_pointer[agent], 3)]
            in_arm = inside_bounds(candidates[step], ARM_BOUNDS[arm])
            if np.any(seen[agent] & in_arm):
                status[agent, arm] = 1
            elif np.linalg.norm(positions[agent] - ARM_CENTERS[arm]) < 1e-6:
                exposure[agent, arm] += 1
                if exposure[agent, arm] >= ABSENCE_SLOTS:
                    status[agent, arm] = 0

        uav_distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        inference_status = merge_neighbor_evidence(status, uav_distances, radius)
        if relay_received_evidence:
            # Forwarded mode models persistent consensus-style evidence: once a
            # receiver learns a fact, it can rebroadcast it on the next slot.
            status = inference_status
        for agent in range(5):
            inferred = identify_layout(inference_status[agent])
            if inferred >= 0 and known_layout[agent] < 0:
                known_layout[agent] = inferred
                identified_at[agent] = step + 1

        if four_known_time > EPISODE_LENGTH and np.sum(known_layout == layout_id) >= 4:
            four_known_time = step + 1
        if (
            deployment_time > EPISODE_LENGTH
            and np.all(np.linalg.norm(positions - target, axis=1) <= 10.0)
        ):
            deployment_time = step + 1
        if np.all(np.linalg.norm(positions - target, axis=1) <= 1e-9):
            if step + 1 < EPISODE_LENGTH:
                remaining = candidates[step + 1:]
                remaining_seen = np.linalg.norm(
                    positions[None, :, None, :] - remaining[:, None, :, :], axis=-1
                ) <= COVER_RADIUS
                admitted += int(np.any(remaining_seen, axis=1).sum())
            break

    return {
        "admitted": admitted,
        "flight_distance": flight_distance,
        "correct_agents": int(np.sum(known_layout == layout_id)),
        "mean_identification_time": float(np.mean(identified_at)),
        "four_known_time": int(four_known_time),
        "deployment_time": int(deployment_time),
    }


def simulate_oracle(layout_id, candidates):
    positions = INITIAL_POSITIONS.copy()
    target = ASSIGNED_TARGETS[layout_id]
    admitted = 0
    flight_distance = 0.0
    deployment_time = EPISODE_LENGTH + 1
    for step in range(EPISODE_LENGTH):
        flight_distance += move_toward(positions, target).sum()
        seen = np.linalg.norm(
            positions[:, None, :] - candidates[step][None, :, :], axis=2
        ) <= COVER_RADIUS
        admitted += int(np.any(seen, axis=0).sum())
        if (
            deployment_time > EPISODE_LENGTH
            and np.all(np.linalg.norm(positions - target, axis=1) <= 10.0)
        ):
            deployment_time = step + 1
        if np.all(np.linalg.norm(positions - target, axis=1) <= 1e-9):
            if step + 1 < EPISODE_LENGTH:
                remaining = candidates[step + 1:]
                remaining_seen = np.linalg.norm(
                    positions[None, :, None, :] - remaining[:, None, :, :], axis=-1
                ) <= COVER_RADIUS
                admitted += int(np.any(remaining_seen, axis=1).sum())
            break
    return {
        "admitted": admitted,
        "flight_distance": flight_distance,
        "correct_agents": 5,
        "mean_identification_time": 0.0,
        "four_known_time": 0,
        "deployment_time": int(deployment_time),
    }


def paired_ci(values, rng, samples=4000):
    values = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def geometry_preflight(rng, episodes=10000, initial_slots=20):
    layout_ids = rng.integers(0, 4, size=episodes)
    candidates = sample_candidates(rng, layout_ids, initial_slots)
    seen = np.linalg.norm(
        INITIAL_POSITIONS[None, None, :, None, :] - candidates[:, :, None, :, :],
        axis=-1,
    ) <= COVER_RADIUS
    admitted = np.any(seen, axis=2)
    rows = []
    for layout_id in range(4):
        selected = layout_ids == layout_id
        first_slot = admitted[selected, 0]
        rows.append({
            "layout": layout_id,
            "episodes": int(selected.sum()),
            "initial_admitted_mean_of_6": float(first_slot.sum(axis=1).mean()),
            "initial_zero_active_probability": float((~first_slot.any(axis=1)).mean()),
            "small_discovered_by_20_probability": float(admitted[selected, :, 0].any(axis=1).mean()),
            "large_discovered_by_20_probability": float(admitted[selected, :, 1:].any(axis=(1, 2)).mean()),
        })
    topology = {}
    distances = np.linalg.norm(
        INITIAL_POSITIONS[:, None, :] - INITIAL_POSITIONS[None, :, :], axis=2
    )
    for radius in (0, 260, 1000):
        adjacency = distances <= radius
        topology[str(radius)] = {
            "mean_other_neighbors": float((adjacency.sum(axis=1) - 1).mean()),
            "other_neighbors_by_uav": (adjacency.sum(axis=1) - 1).tolist(),
        }
    return rows, topology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--geometry-episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--map-size", type=int, choices=(600, 700), default=600)
    parser.add_argument(
        "--initial-layout",
        choices=("legacy", "symmetric_cross"),
        default="legacy",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=("forwarded", "direct_only"),
        default="forwarded",
        help=(
            "forwarded permits received evidence to be rebroadcast on later "
            "slots; direct_only broadcasts only each UAV's own discoveries."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_map_size(args.map_size)
    configure_initial_positions(args.initial_layout)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root_seed = np.random.SeedSequence(args.seed)
    geometry_seed, trace_seed, order_seed, ci_seed = root_seed.spawn(4)
    geometry, topology = geometry_preflight(
        np.random.default_rng(geometry_seed), args.geometry_episodes
    )

    trace_rng = np.random.default_rng(trace_seed)
    order_rng = np.random.default_rng(order_seed)
    layout_ids = trace_rng.integers(0, 4, size=args.episodes)
    candidates = sample_candidates(trace_rng, layout_ids, EPISODE_LENGTH)
    base_orders = np.array([
        np.roll(order_rng.permutation(4), -agent) for agent in range(5)
    ])

    methods = ("R0", "R260", "R1000", "Oracle")
    records = []
    for episode in range(args.episodes):
        for method, radius in (("R0", 0), ("R260", 260), ("R1000", 1000)):
            result = simulate_episode(
                int(layout_ids[episode]),
                candidates[episode],
                radius,
                base_orders,
                relay_received_evidence=args.evidence_mode == "forwarded",
            )
            records.append({"episode": episode, "layout": int(layout_ids[episode]),
                            "method": method, **result})
        result = simulate_oracle(int(layout_ids[episode]), candidates[episode])
        records.append({"episode": episode, "layout": int(layout_ids[episode]),
                        "method": "Oracle", **result})

    csv_path = args.output_dir / "heuristic_episode_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    arrays = {
        method: np.array([row["admitted"] for row in records if row["method"] == method])
        for method in methods
    }
    summaries = {}
    for method in methods:
        selected = [row for row in records if row["method"] == method]
        summaries[method] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "admitted", "flight_distance", "correct_agents",
                "mean_identification_time", "four_known_time", "deployment_time",
            )
        }
        summaries[method]["admission_ratio"] = summaries[method]["admitted"] / (6 * EPISODE_LENGTH)

    ci_rng = np.random.default_rng(ci_seed)
    comparisons = {}
    for better, baseline in (("R260", "R0"), ("R1000", "R260"), ("Oracle", "R0")):
        difference = arrays[better] - arrays[baseline]
        comparisons[f"{better}-{baseline}"] = {
            "mean_admitted_difference": float(difference.mean()),
            "relative_percent": float(100 * difference.mean() / arrays[baseline].mean()),
            "paired_95_ci": paired_ci(difference, ci_rng),
            "positive_episode_fraction": float((difference > 0).mean()),
        }

    layout_summary = {}
    for layout_id in range(4):
        layout_summary[str(layout_id)] = {}
        mask = layout_ids == layout_id
        for method in methods:
            layout_summary[str(layout_id)][method] = float(arrays[method][mask].mean())

    oracle_gap = arrays["Oracle"] - arrays["R0"]
    recovered = {}
    valid = np.abs(oracle_gap) > 1e-9
    for method in ("R260", "R1000"):
        recovery = (arrays[method][valid] - arrays["R0"][valid]) / oracle_gap[valid]
        recovered[method] = float(recovery.mean())

    report = {
        "contract": {
            "seed": args.seed,
            "map_size": MAP_SIZE,
            "initial_layout": args.initial_layout,
            "initial_positions": INITIAL_POSITIONS.tolist(),
            "evidence_mode": args.evidence_mode,
            "heuristic_episodes": args.episodes,
            "geometry_episodes": args.geometry_episodes,
            "episode_length": EPISODE_LENGTH,
            "absence_slots": ABSENCE_SLOTS,
            "shared_candidate_trace": True,
        },
        "geometry_by_layout": geometry,
        "initial_topology": topology,
        "heuristic_summary": summaries,
        "paired_comparisons": comparisons,
        "oracle_gap_recovery": recovered,
        "admitted_by_layout": layout_summary,
    }
    (args.output_dir / "preflight_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    means = [summaries[m]["admitted"] for m in methods]
    axes[0].bar(methods, means, color=("#777777", "#4C78A8", "#F58518", "#54A24B"))
    axes[0].set_ylabel("Admitted candidate births / episode")
    axes[0].set_title("Shared-trace discovery heuristic")
    x = np.arange(4)
    width = 0.2
    for idx, method in enumerate(methods):
        axes[1].bar(x + (idx - 1.5) * width,
                    [layout_summary[str(i)][method] for i in range(4)],
                    width, label=method)
    axes[1].set_xticks(x, [f"Layout {i}" for i in range(4)])
    axes[1].set_ylabel("Admitted candidate births / episode")
    axes[1].set_title("Stratified by hidden layout")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "preflight_admission_comparison.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
