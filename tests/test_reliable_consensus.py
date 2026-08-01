from types import SimpleNamespace

import numpy as np

from onpolicy.runner.separated.mec_runner import MECRunner


def test_finite_consensus_approaches_component_mean():
    runner = SimpleNamespace(num_agents=3, n_rollout_threads=1, episode_length=2)
    weights = np.array(
        [
            [2 / 3, 1 / 3, 0],
            [1 / 3, 1 / 3, 1 / 3],
            [0, 1 / 3, 2 / 3],
        ],
        dtype=np.float32,
    )
    runner.buffer = [
        SimpleNamespace(Metropolis_weights=np.array([[weights[i]]], dtype=np.float32))
        for i in range(3)
    ]
    local_advantages = np.array(
        [
            [[[0.0]], [[2.0]]],
            [[[6.0]], [[4.0]]],
            [[[0.0]], [[8.0]]],
        ],
        dtype=np.float32,
    )

    after_one = MECRunner.run_consensus_algorithm(runner, local_advantages, 1)
    after_fifty = MECRunner.run_consensus_algorithm(runner, local_advantages, 50)
    exact_mean = np.mean(local_advantages, axis=0, keepdims=True)

    assert np.mean(np.abs(after_fifty - exact_mean)) < np.mean(
        np.abs(after_one - exact_mean)
    )
    np.testing.assert_allclose(
        after_fifty, np.broadcast_to(exact_mean, after_fifty.shape), atol=1e-5
    )


def test_normalized_consensus_residual_has_expected_endpoints():
    local_advantages = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32
    ).reshape(5, 1, 1, 1)
    exact_mean = np.broadcast_to(
        np.mean(local_advantages, axis=0, keepdims=True), local_advantages.shape
    )

    self_only = MECRunner.normalized_consensus_residual(
        local_advantages, local_advantages
    )
    full_consensus = MECRunner.normalized_consensus_residual(
        local_advantages, exact_mean
    )
    partial_consensus = MECRunner.normalized_consensus_residual(
        local_advantages, 0.5 * local_advantages + 0.5 * exact_mean
    )

    np.testing.assert_allclose(self_only, 1.0)
    np.testing.assert_allclose(full_consensus, 0.0)
    np.testing.assert_allclose(partial_consensus, 0.5)
    assert self_only.shape == (1, 1, 1, 1)


def test_normalized_consensus_residual_is_zero_without_disagreement():
    local_advantages = np.full((5, 2, 3, 1), 7.0, dtype=np.float32)

    residual = MECRunner.normalized_consensus_residual(
        local_advantages, local_advantages.copy()
    )

    np.testing.assert_array_equal(residual, np.zeros((1, 2, 3, 1)))
