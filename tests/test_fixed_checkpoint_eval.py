import numpy as np
from types import SimpleNamespace

from onpolicy.scripts.eval.compare_fixed_evaluations import paired_stats
from onpolicy.scripts.eval.evaluate_dynamic_mappo_fixed import (
    SUMMARY_METRICS,
    aggregate_rows,
    config_sha256,
    final_position_stability_slot,
    matched_config,
)
from onpolicy.scripts.eval.render_dynamic_mappo_episode import (
    mask_actor_message_observations,
)
from scipy.stats import t as student_t


def test_settling_slot_finds_suffix_near_final_deployment():
    positions = np.array(
        [
            [[0.0, 0.0], [100.0, 0.0]],
            [[20.0, 0.0], [90.0, 0.0]],
            [[39.0, 0.0], [81.0, 0.0]],
            [[41.0, 0.0], [79.0, 0.0]],
            [[40.0, 0.0], [80.0, 0.0]],
        ]
    )

    assert final_position_stability_slot(positions, tolerance=2.0) == 2


def test_aggregate_rows_reports_sample_std_and_confidence_width():
    rows = []
    for value in (1.0, 3.0, 5.0):
        row = {metric: value for metric in SUMMARY_METRICS}
        rows.append(row)

    aggregate = aggregate_rows(rows)

    for metric in SUMMARY_METRICS:
        assert aggregate[metric]["count"] == 3
        assert aggregate[metric]["mean"] == 3.0
        np.testing.assert_allclose(aggregate[metric]["std"], 2.0)
        np.testing.assert_allclose(
            aggregate[metric]["ci95_half_width"],
            student_t.ppf(0.975, 2) * 2.0 / np.sqrt(3),
        )


def test_paired_stats_uses_within_seed_differences():
    baseline = np.array([100.0, 200.0, 300.0, 400.0])
    candidate = baseline + np.array([5.0, 7.0, 9.0, 11.0])

    stats = paired_stats(baseline, candidate)

    assert stats["count"] == 4
    assert stats["baseline_mean"] == 250.0
    assert stats["candidate_mean"] == 258.0
    assert stats["mean_difference"] == 8.0
    assert stats["win_fraction"] == 1.0
    assert stats["ci95_low"] > 0.0


def test_paired_stats_respects_lower_is_better_direction():
    baseline = np.array([10.0, 12.0, 14.0, 16.0])
    candidate = baseline - 2.0

    stats = paired_stats(baseline, candidate, direction=-1)

    assert stats["mean_difference"] == -2.0
    assert stats["direction_adjusted_mean_improvement"] == 2.0
    assert stats["win_fraction"] == 1.0


def test_matched_config_ignores_only_predeclared_radius_and_run_fields():
    base = {
        "neighbor_distance": 0.0,
        "externality_beta": 0.025,
        "experiment_name": "r0",
        "seed": 2,
        "md_lifetime_max": 10,
        "mean_velocity": 0.5,
    }
    candidate = dict(
        base,
        neighbor_distance=1000.0,
        externality_beta=0.1,
        experiment_name="r1000",
    )

    assert config_sha256(matched_config(base)) == config_sha256(
        matched_config(candidate)
    )
    candidate["mean_velocity"] = 2.0
    assert config_sha256(matched_config(base)) != config_sha256(
        matched_config(candidate)
    )


def test_actor_message_masking_zeros_only_declared_preserved_slice():
    obs = np.arange(2 * 12, dtype=np.float32).reshape(2, 12)
    normers = [
        SimpleNamespace(obs_preserve_slices=[(3, 7)]),
        SimpleNamespace(obs_preserve_slices=[(3, 7)]),
    ]

    masked = mask_actor_message_observations(normers, obs)

    np.testing.assert_array_equal(masked[:, :3], obs[:, :3])
    np.testing.assert_array_equal(masked[:, 3:7], 0.0)
    np.testing.assert_array_equal(masked[:, 7:], obs[:, 7:])
    np.testing.assert_array_equal(obs, np.arange(24, dtype=np.float32).reshape(2, 12))


def test_actor_message_masking_rejects_non_message_checkpoint():
    normers = [SimpleNamespace(obs_preserve_slices=[]) for _ in range(2)]
    with np.testing.assert_raises_regex(
        ValueError, "message-capable checkpoint"
    ):
        mask_actor_message_observations(normers, np.ones((2, 8)))
