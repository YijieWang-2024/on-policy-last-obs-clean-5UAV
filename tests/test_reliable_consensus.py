from types import SimpleNamespace

import numpy as np

from onpolicy.envs.mec.mec import MEC
from onpolicy.runner.separated.mec_runner import MECRunner


def test_finite_consensus_approaches_component_mean():
    runner = SimpleNamespace(
        num_agents=3,
        n_rollout_threads=1,
        episode_length=2,
        neighbor_distance=1.1,
        uav_positions=np.array(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=np.float32
        ),
    )
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


def test_terminal_consensus_uses_positions_and_neighbor_distance_not_buffer_graph():
    runner = SimpleNamespace(
        num_agents=3,
        n_rollout_threads=1,
        episode_length=1,
        neighbor_distance=1.0,
        uav_positions=np.array(
            [[[0.0, 0.0], [0.5, 0.0], [10.0, 0.0]]], dtype=np.float32
        ),
        buffer=[
            SimpleNamespace(
                Metropolis_weights=np.array(
                    [[[1.0, 0.0, 0.0]]], dtype=np.float32
                )
            )
            for _ in range(3)
        ],
    )
    local_advantages = np.array(
        [[[[1.0]]], [[[5.0]]], [[[9.0]]]], dtype=np.float32
    )

    terminal_result = MECRunner.run_consensus_algorithm(
        runner, local_advantages, max_iterations=1
    )
    np.testing.assert_allclose(
        terminal_result[:, 0, 0, 0], [3.0, 3.0, 9.0], atol=1e-6
    )
    terminal_noise = MECRunner.per_agent_consensus_residual(
        local_advantages, terminal_result
    )
    np.testing.assert_allclose(
        terminal_noise[:, 0, 0, 0], [0.5, 1.0, 1.0], atol=1e-6
    )

    runner.neighbor_distance = 100.0
    connected_result = MECRunner.run_consensus_algorithm(
        runner, local_advantages, max_iterations=1
    )
    np.testing.assert_allclose(
        connected_result[:, 0, 0, 0], [5.0, 5.0, 5.0], atol=1e-6
    )
    connected_noise = MECRunner.per_agent_consensus_residual(
        local_advantages, connected_result
    )
    np.testing.assert_allclose(
        connected_noise[:, 0, 0, 0], [0.0, 0.0, 0.0], atol=1e-6
    )

    runner.neighbor_distance = 0.0
    self_only_result = MECRunner.run_consensus_algorithm(
        runner, local_advantages, max_iterations=1
    )
    np.testing.assert_allclose(
        self_only_result, local_advantages, atol=1e-6
    )


def test_exact_mean_local_advantage_has_zero_normalized_residual():
    local = np.asarray([1.0, 5.0, 9.0], dtype=np.float32)[:, None, None, None]
    consensus = np.asarray([3.0, 5.0, 7.0], dtype=np.float32)[:, None, None, None]

    residual = MECRunner.per_agent_consensus_residual(local, consensus)

    assert residual[1, 0, 0, 0] == 0.0


def test_exact_mean_local_advantage_detects_consensus_displacement():
    local = np.asarray([0.0, 5.0, 10.0], dtype=np.float32)[:, None, None, None]
    consensus = np.asarray([2.5, 7.5, 7.5], dtype=np.float32)[:, None, None, None]

    residual = MECRunner.per_agent_consensus_residual(local, consensus)

    assert residual[1, 0, 0, 0] == 1.0

def test_zero_radius_is_strict_self_only_even_for_overlapping_uavs():
    env = SimpleNamespace(
        neighbor_distance=0.0,
        n_UAVs=3,
        uav_uav_distances_2d=np.zeros((3, 3), dtype=np.float32),
    )

    np.testing.assert_array_equal(
        MEC.get_Metropolis_weights(env), np.eye(3, dtype=float)
    )
    np.testing.assert_array_equal(
        MEC.get_neighbor_weights(env), np.eye(3, dtype=float)
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


def test_externality_consensus_r0_is_exact_local_endpoint():
    local_advantages = np.array(
        [
            [[[-2.0]], [[0.0]], [[4.0]]],
            [[[1.0]], [[3.0]], [[8.0]]],
            [[[-5.0]], [[2.0]], [[6.0]]],
        ],
        dtype=np.float32,
    )
    self_only_consensus = local_advantages / local_advantages.shape[0]

    training, externality = MECRunner.externality_consensus_advantages(
        local_advantages,
        self_only_consensus,
        beta=0.4,
        component_sizes=np.ones((1, 3), dtype=np.float32),
        self_contribution_coefficients=np.ones((1, 3), dtype=np.float32),
    )

    np.testing.assert_array_equal(externality, np.zeros_like(local_advantages))
    np.testing.assert_allclose(
        training,
        local_advantages,
        atol=1e-7,
    )


def test_externality_consensus_full_graph_contains_only_other_uavs():
    local_advantages = np.array(
        [
            [[[-2.0]], [[0.0]], [[4.0]]],
            [[[1.0]], [[3.0]], [[8.0]]],
            [[[-5.0]], [[2.0]], [[6.0]]],
        ],
        dtype=np.float32,
    )
    exact_mean = np.broadcast_to(
        np.mean(local_advantages, axis=0, keepdims=True),
        local_advantages.shape,
    )

    training, externality = MECRunner.externality_consensus_advantages(
        local_advantages,
        exact_mean,
        beta=0.2,
        component_sizes=np.full((1, 3), 3, dtype=np.float32),
        self_contribution_coefficients=np.ones((1, 3), dtype=np.float32),
    )
    expected_externality = (
        np.sum(local_advantages, axis=0, keepdims=True) - local_advantages
    )
    expected_training = (
        local_advantages
        + 0.2 * expected_externality
    )

    np.testing.assert_allclose(externality, expected_externality, atol=1e-7)
    np.testing.assert_allclose(training, expected_training, atol=1e-7)


def test_externality_consensus_rejects_invalid_contract():
    local_advantages = np.zeros((3, 2, 1, 1), dtype=np.float32)

    for beta in (-0.1, 1.1):
        try:
            MECRunner.externality_consensus_advantages(
                local_advantages, local_advantages, beta
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid beta must raise ValueError")

    try:
        MECRunner.externality_consensus_advantages(
            local_advantages, local_advantages[:2], beta=0.1
        )
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise ValueError")

    try:
        MECRunner.externality_consensus_advantages(
            local_advantages,
            local_advantages,
            beta=0.1,
            component_sizes=np.ones((3, 1), dtype=np.float32),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("component-size shape mismatch must raise ValueError")


def test_externality_consensus_mixed_batch_keeps_isolated_samples_local():
    local_advantages = np.array(
        [
            [[[1.0], [2.0]], [[3.0], [4.0]]],
            [[[5.0], [6.0]], [[7.0], [8.0]]],
            [[[9.0], [10.0]], [[11.0], [12.0]]],
        ],
        dtype=np.float32,
    )
    consensus_advantages = local_advantages / 3.0
    consensus_advantages[:, :, 1] = np.mean(
        local_advantages[:, :, 1], axis=0, keepdims=True
    )
    component_sizes = np.array(
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32
    )

    training, externality = MECRunner.externality_consensus_advantages(
        local_advantages,
        consensus_advantages,
        beta=0.2,
        component_sizes=component_sizes,
        self_contribution_coefficients=np.ones((2, 3), dtype=np.float32),
    )

    np.testing.assert_array_equal(
        externality[:, :, 0], np.zeros_like(externality[:, :, 0])
    )
    np.testing.assert_allclose(
        training[:, :, 0], local_advantages[:, :, 0], atol=1e-7
    )


def test_finite_round_externality_removes_actual_self_contribution():
    local_advantages = np.array([1.0, 5.0], dtype=np.float32).reshape(2, 1, 1, 1)
    one_round_weights = np.array(
        [[0.75, 0.25], [0.25, 0.75]], dtype=np.float32
    )
    consensus_advantages = (one_round_weights @ local_advantages[:, 0, 0, 0])\
        .reshape(2, 1, 1, 1)
    self_coefficients = np.full((1, 2), 1.5, dtype=np.float32)

    training, externality = MECRunner.externality_consensus_advantages(
        local_advantages,
        consensus_advantages,
        beta=0.2,
        component_sizes=np.full((1, 2), 2, dtype=np.float32),
        self_contribution_coefficients=self_coefficients,
    )

    expected_externality = np.array([2.5, 0.5], dtype=np.float32).reshape(
        2, 1, 1, 1
    )
    np.testing.assert_allclose(externality, expected_externality, atol=1e-7)
    np.testing.assert_allclose(
        training,
        local_advantages + 0.2 * expected_externality,
        atol=1e-7,
    )
