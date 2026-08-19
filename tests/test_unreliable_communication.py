from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest

from onpolicy.config import get_config
from onpolicy.envs.mec.mec import MEC
from onpolicy.scripts.train.train_mec import parse_args
from onpolicy.runner.separated.mec_runner import MECRunner
from onpolicy.utils.unreliable_communication import (
    communication_distance_from_power,
    communication_range_mask,
    normalized_consensus_residual,
    running_sum_ratio_consensus,
    sample_timely_receptions,
    transmit_power_from_communication_distance,
)


def _mec_args(communication_mode):
    args = parse_args([], get_config())
    settings = {
        "n_UAVs": 5,
        "n_GUs": 60,
        "max_GUs_in_range": 20,
        "dynamic_md": True,
        "md_arrivals_min": 6,
        "md_arrivals_max": 6,
        "md_arrivals_per_region": [1, 5],
        "md_lifetime_min": 10,
        "md_lifetime_max": 10,
        "x_min_uav": 0,
        "x_max_uav": 600,
        "y_min_uav": 0,
        "y_max_uav": 600,
        "x_min_gu": 0,
        "x_max_gu": 600,
        "y_min_gu": 0,
        "y_max_gu": 600,
        "fix_hotspot": True,
        "random_hotspot": False,
        "state_is_k_hops": True,
        "all_uav_k_hops": True,
        "max_UAVs_obs_concat": 5,
        "max_UAVs_in_neighbor": 5,
        "neighbor_distance": 1000,
        "continuous_associate": True,
        "not_served_rew_to_nearest": True,
        "communication_mode": communication_mode,
        "episode_length": 1,
    }
    for name, value in settings.items():
        setattr(args, name, value)
    return args


def _off_diagonal_receptions(rounds, environments, uavs, value):
    receptions = np.full((rounds, environments, uavs, uavs), value, dtype=bool)
    diagonal = np.arange(uavs)
    receptions[:, :, diagonal, diagonal] = False
    return receptions


def test_default_and_bidirectional_pc_dcom_resolution():
    default = parse_args([], get_config())
    assert default.neighbor_distance == 520.0
    assert default.running_sum_rounds == 50
    assert default.a2a_rician_k_db == 10.0
    assert default.a2a_transmit_power_w == pytest.approx(1.1809658836)

    distance_only = parse_args(["--d_com", "260"], get_config())
    assert distance_only.a2a_transmit_power_w == pytest.approx(0.2570226288)

    power_only = parse_args(
        ["--a2a_transmit_power_w", "1.18"], get_config()
    )
    assert power_only.neighbor_distance == pytest.approx(519.8066, abs=1e-3)

    consistent = parse_args(
        ["--neighbor_distance", "520", "--a2a_transmit_power_w", "1.18"],
        get_config(),
    )
    assert consistent.neighbor_distance == 520.0

    with pytest.raises(SystemExit):
        parse_args(
            ["--neighbor_distance", "520", "--a2a_transmit_power_w", "0.257"],
            get_config(),
        )
    with pytest.raises(SystemExit):
        parse_args(["--a2a_transmit_power_w", "nan"], get_config())
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--neighbor_distance", "-1",
                "--a2a_transmit_power_w", "1.250345385551372e-06",
            ],
            get_config(),
        )
    with pytest.raises(SystemExit):
        parse_args(
            ["--neighbor_distance", "0", "--a2a_transmit_power_w", "1e-6"],
            get_config(),
        )
    zero = parse_args(
        ["--neighbor_distance", "0", "--a2a_transmit_power_w", "0"],
        get_config(),
    )
    assert zero.neighbor_distance == 0.0
    assert zero.a2a_transmit_power_w == 0.0


def test_pc_dcom_formula_round_trips_paper_radii():
    channel = dict(
        bandwidth_hz=2e6,
        reference_gain_db=-38.46,
        reference_distance_m=1.0,
        path_loss_exponent=2.2,
        noise_psd_dbm_hz=-130.0,
        decoding_threshold_db=-0.5,
    )
    expected = {260.0: 0.2570226288, 520.0: 1.1809658836, 780.0: 2.8816293679}
    for distance, power in expected.items():
        calculated_power = transmit_power_from_communication_distance(
            distance, **channel
        )
        assert calculated_power == pytest.approx(power)
        assert communication_distance_from_power(
            calculated_power, **channel
        ) == pytest.approx(distance)

    just_inside_power = transmit_power_from_communication_distance(524.999, **channel)
    parse_args(
        [
            "--neighbor_distance", "520",
            "--a2a_transmit_power_w", str(just_inside_power),
        ],
        get_config(),
    )
    just_outside_power = transmit_power_from_communication_distance(525.001, **channel)
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--neighbor_distance", "520",
                "--a2a_transmit_power_w", str(just_outside_power),
            ],
            get_config(),
        )


def test_range_graph_is_tunable_strictly_off_diagonal_and_zero_safe():
    positions = np.asarray([[[0.0, 0.0], [520.0, 0.0], [520.01, 0.0]]])
    mask = communication_range_mask(positions, 520.0)
    assert mask[0, 0, 1]
    assert not mask[0, 0, 2]
    assert not np.any(np.diagonal(mask, axis1=1, axis2=2))

    colocated = np.zeros((1, 3, 2), dtype=np.float64)
    assert not np.any(communication_range_mask(colocated, 0.0))


def test_range_graph_running_sum_uses_actual_out_degree_and_components():
    chain = np.asarray([[
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]])
    receptions = np.broadcast_to(chain, (200,) + chain.shape).copy()
    local = np.asarray([0.0, 6.0, 12.0], dtype=np.float32).reshape(3, 1, 1, 1)
    one_round = running_sum_ratio_consensus(
        local, receptions[:1], adjacency=chain
    )
    np.testing.assert_allclose(one_round[:, 0, 0, 0], [2.4, 6.0, 9.6])
    estimate = running_sum_ratio_consensus(local, receptions, adjacency=chain)
    np.testing.assert_allclose(estimate, 6.0, atol=1e-5)

    components = np.asarray([[
        [False, True, False, False],
        [True, False, False, False],
        [False, False, False, True],
        [False, False, True, False],
    ]])
    receptions = np.broadcast_to(components, (40,) + components.shape).copy()
    local = np.asarray([0.0, 2.0, 10.0, 14.0], dtype=np.float32).reshape(4, 1, 1, 1)
    estimate = running_sum_ratio_consensus(local, receptions, adjacency=components)
    np.testing.assert_allclose(estimate[:, 0, 0, 0], [1.0, 1.0, 12.0, 12.0])


def test_running_sum_masks_non_edges_without_mutating_receptions():
    adjacency = np.asarray([[
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ]])
    receptions = _off_diagonal_receptions(3, 1, 3, True)
    original = receptions.copy()
    local = np.asarray([0.0, 2.0, 9.0], dtype=np.float32).reshape(3, 1, 1, 1)

    estimate = running_sum_ratio_consensus(local, receptions, adjacency=adjacency)

    np.testing.assert_array_equal(receptions, original)
    np.testing.assert_allclose(estimate[2], local[2])


def test_type_a_uses_terminal_range_graph_and_counts_only_nominal_edges():
    runner = object.__new__(MECRunner)
    runner.num_agents = 3
    runner.neighbor_distance = 1.1
    runner.uav_positions = np.asarray([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])
    runner.communication_rng = np.random.default_rng(9)
    runner.all_args = SimpleNamespace(
        running_sum_rounds=2,
        advantage_payload_bits=16000.0,
        advantage_deadline_ms=21.54,
    )
    captured = []

    def fake_sampler(positions, rounds, rng, args, *, payload_bits, deadline_ms):
        captured.append(positions.copy())
        return _off_diagonal_receptions(rounds, 1, 3, True)

    local = np.asarray([0.0, 2.0, 9.0], dtype=np.float32).reshape(3, 1, 1, 1)
    with patch(
        "onpolicy.runner.separated.mec_runner.sample_configured_timely_receptions",
        side_effect=fake_sampler,
    ):
        estimate, rate = MECRunner.run_unreliable_consensus(runner, local)

    np.testing.assert_array_equal(captured[0], runner.uav_positions)
    np.testing.assert_allclose(estimate[2], local[2])
    assert rate == 1.0


def test_all_success_running_sum_reaches_exact_mean_in_one_round():
    local = np.array(
        [
            [[[0.0]], [[2.0]]],
            [[[6.0]], [[4.0]]],
            [[[0.0]], [[8.0]]],
        ],
        dtype=np.float32,
    )
    receptions = _off_diagonal_receptions(1, 1, 3, True)

    estimate = running_sum_ratio_consensus(local, receptions)
    exact_mean = np.mean(local, axis=0, keepdims=True)

    np.testing.assert_allclose(estimate, np.broadcast_to(exact_mean, local.shape))


def test_running_sum_recovers_after_initial_packet_loss():
    local = np.array([[[[0.0]]], [[[10.0]]]], dtype=np.float32)
    receptions = _off_diagonal_receptions(30, 1, 2, True)
    receptions[0] = False

    estimate = running_sum_ratio_consensus(local, receptions)

    np.testing.assert_allclose(estimate, 5.0, atol=1e-6)


def test_normalized_consensus_residual_has_system_level_endpoints():
    local = np.arange(-2.0, 3.0, dtype=np.float32).reshape(5, 1, 1, 1)
    exact_mean = np.mean(local, axis=0, keepdims=True)

    returned_mean, self_only = normalized_consensus_residual(local, local)
    _, exact = normalized_consensus_residual(
        local, np.broadcast_to(exact_mean, local.shape)
    )
    _, halfway = normalized_consensus_residual(
        local, 0.5 * local + 0.5 * exact_mean
    )
    _, clipped = normalized_consensus_residual(local, 3.0 * local)

    np.testing.assert_allclose(returned_mean, exact_mean[0])
    np.testing.assert_allclose(self_only, 1.0)
    np.testing.assert_allclose(exact, 0.0)
    np.testing.assert_allclose(halfway, 0.5)
    np.testing.assert_allclose(clipped, 1.0)
    assert self_only.shape == (1, 1, 1, 1)


def test_normalized_consensus_residual_is_zero_without_disagreement():
    local = np.full((5, 2, 3, 1), 7.0, dtype=np.float32)
    _, residual = normalized_consensus_residual(local, local.copy())
    np.testing.assert_array_equal(residual, np.zeros((1, 2, 3, 1)))


def test_physical_channel_is_directed_and_distance_dependent():
    samples = 20_000
    positions = np.zeros((samples * 2, 2, 2), dtype=np.float64)
    positions[:samples, 1, 0] = 260.0
    positions[samples:, 1, 0] = 520.0

    receptions = sample_timely_receptions(
        positions,
        1,
        np.random.default_rng(7),
        transmit_power_w=2.0,
        bandwidth_hz=2e6,
        reference_gain_db=-38.46,
        reference_distance_m=1.0,
        path_loss_exponent=2.2,
        rician_k_db=6.0,
        noise_psd_dbm_hz=-130.0,
        spectral_efficiency=0.5,
        decoding_threshold_db=-0.5,
        gamma_shape=2.5,
        gamma_scale_ms=1.0,
        payload_bits=16000.0,
        deadline_ms=21.54,
    )

    near_rate = receptions[0, :samples, 0, 1].mean()
    far_rate = receptions[0, samples:, 0, 1].mean()
    assert near_rate > far_rate + 0.15
    assert not np.any(receptions[0, :, np.arange(2), np.arange(2)])
    assert np.any(receptions[0, :, 0, 1] != receptions[0, :, 1, 0])


def test_physical_sampler_never_delivers_outside_dcom():
    positions = np.asarray([[[0.0, 0.0], [100.0, 0.0], [600.0, 0.0]]])
    receptions = sample_timely_receptions(
        positions,
        20,
        np.random.default_rng(13),
        transmit_power_w=1e6,
        bandwidth_hz=2e6,
        reference_gain_db=-38.46,
        reference_distance_m=1.0,
        path_loss_exponent=2.2,
        rician_k_db=10.0,
        noise_psd_dbm_hz=-130.0,
        spectral_efficiency=0.5,
        decoding_threshold_db=-0.5,
        gamma_shape=2.5,
        gamma_scale_ms=1.0,
        payload_bits=8000.0,
        deadline_ms=100.0,
        max_distance_m=520.0,
    )

    assert np.any(receptions[:, 0, 0, 1])
    assert not np.any(receptions[:, 0, 0, 2])
    assert not np.any(receptions[:, 0, 2, 0])


def test_type_s_packets_build_and_store_masked_critic_states_each_slot():
    args = _mec_args("unreliable")
    sampled_positions = []

    def fake_sampler(positions, rounds, rng, config, *, payload_bits, deadline_ms):
        sampled_positions.append(positions.copy())
        receptions = np.zeros((1, 1, args.n_UAVs, args.n_UAVs), dtype=bool)
        sender = 1 if len(sampled_positions) == 1 else 2
        receptions[0, 0, sender, 0] = True
        assert rounds == 1
        assert payload_bits == args.state_payload_bits
        assert deadline_ms == args.state_deadline_ms
        return receptions

    env = MEC(args)
    env.seed(11)
    with patch(
        "onpolicy.envs.mec.mec.sample_configured_timely_receptions",
        side_effect=fake_sampler,
    ):
        obs, state, _, _, attention_mask = env.reset()
        np.testing.assert_allclose(sampled_positions[0][0, :, :2], env.uav_positions[:, :2])
        critic_local = env.get_critic_local_obs(obs)
        state_blocks = state.reshape(args.n_UAVs, args.n_UAVs, -1)
        np.testing.assert_allclose(state_blocks[0, 0], critic_local[0])
        np.testing.assert_allclose(state_blocks[0, 1], critic_local[1])
        np.testing.assert_allclose(state_blocks[0, 2:], 0)
        np.testing.assert_array_equal(attention_mask[0], [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(attention_mask[1], [1, 0, 0, 0, 0])

        action_dim = sum(int(np.prod(space.shape)) for space in env.action_space.spaces)
        action = np.random.default_rng(5).uniform(
            0.1, 0.9, (args.n_UAVs, action_dim)
        ).astype(np.float32)
        env.delay_true_coverd_GUs[:, 1] = 1
        next_obs, _, _, next_state, _, info, _, next_attention_mask = env.step(action)

    np.testing.assert_allclose(sampled_positions[1][0, :, :2], env.uav_positions[:, :2])
    next_critic_local = env.get_critic_local_obs(next_obs)
    next_blocks = next_state.reshape(args.n_UAVs, args.n_UAVs, -1)
    np.testing.assert_allclose(next_blocks[0, 0], next_critic_local[0])
    np.testing.assert_allclose(next_blocks[0, 1], 0)
    np.testing.assert_allclose(next_blocks[0, 2], next_critic_local[2])
    np.testing.assert_array_equal(next_attention_mask[0], [1, 0, 1, 0, 0])
    assert info["state_timely_reception_rate"] == 2 / (2 * 5 * 4)


def test_type_s_channel_rng_does_not_change_environment_trajectory():
    action = np.random.default_rng(17).uniform(0.1, 0.9, (5, 62)).astype(np.float32)

    def one_step(mode):
        env = MEC(_mec_args(mode))
        env.seed(19)
        initial_obs = env.reset()[0]
        next_obs, rewards = env.step(action)[:2]
        return initial_obs, next_obs, rewards, env.uav_positions.copy(), env.gu_tasks.copy()

    reliable = one_step("reliable")
    unreliable = one_step("unreliable")
    for reliable_value, unreliable_value in zip(reliable, unreliable):
        np.testing.assert_allclose(reliable_value, unreliable_value)


def test_type_s_reception_rate_counts_only_current_episode_geometric_links():
    args = _mec_args("unreliable")
    args.neighbor_distance = 1.0
    env = MEC(args)
    env.uav_positions[:, :2] = np.asarray([
        [0.0, 0.0], [0.5, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]
    ])
    env._update_distance_matrices()

    all_success = np.ones((1, 1, args.n_UAVs, args.n_UAVs), dtype=bool)
    with patch(
        "onpolicy.envs.mec.mec.sample_configured_timely_receptions",
        return_value=all_success,
    ):
        reception_mask = env._sample_state_reception_mask()

    # Only the directed 0->1 and 1->0 neighbor links are attempted.
    assert env.state_packets_attempted == 2
    assert env.state_packets_received == 2
    assert reception_mask[0, 1]
    assert reception_mask[1, 0]
    assert not reception_mask[0, 2]

    env.neighbor_distance = 0.0
    env.uav_positions[1, :2] = env.uav_positions[0, :2]
    env._update_distance_matrices()
    env.state_packets_attempted = 0
    env.state_packets_received = 0
    with patch(
        "onpolicy.envs.mec.mec.sample_configured_timely_receptions",
        return_value=all_success,
    ):
        zero_radius_mask = env._sample_state_reception_mask()
    assert env.state_packets_attempted == 0
    assert env.state_packets_received == 0
    assert not zero_radius_mask[0, 1]

    env.state_packets_attempted = 17
    env.state_packets_received = 9
    with patch.object(env, "_build_critic_state", wraps=env._build_critic_state):
        env.reset()
    # reset() starts a fresh episode before its initial critic state samples
    # the first slot, so old-episode counters must not survive.
    assert env.state_packets_attempted != 17
    assert env.state_packets_received != 9
