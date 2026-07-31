from unittest.mock import patch

import numpy as np

from onpolicy.config import get_config
from onpolicy.envs.mec.mec import MEC
from onpolicy.scripts.train.train_mec import parse_args
from onpolicy.utils.unreliable_communication import (
    relative_estimation_error,
    running_sum_ratio_consensus,
    sample_timely_receptions,
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


def test_communication_estimate_only_controls_noise_scale():
    local = np.array([[[[0.0]]], [[[10.0]]]], dtype=np.float32)
    communication_estimate = local.copy()

    exact_mean, noise_scale = relative_estimation_error(local, communication_estimate)

    np.testing.assert_allclose(exact_mean, 5.0)
    np.testing.assert_allclose(noise_scale, 1.0)
    np.testing.assert_allclose(local + exact_mean, np.array([[[[5.0]]], [[[15.0]]]]))


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
