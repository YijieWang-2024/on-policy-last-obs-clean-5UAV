import numpy as np

from onpolicy.utils.unreliable_communication import (
    relative_estimation_error,
    running_sum_ratio_consensus,
    sample_timely_receptions,
)


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
