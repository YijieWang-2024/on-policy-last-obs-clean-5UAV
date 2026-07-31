import numpy as np


def sample_timely_receptions(
    positions,
    rounds,
    rng,
    *,
    transmit_power_w,
    bandwidth_hz,
    reference_gain_db,
    reference_distance_m,
    path_loss_exponent,
    rician_k_db,
    noise_psd_dbm_hz,
    spectral_efficiency,
    decoding_threshold_db,
    gamma_shape,
    gamma_scale_ms,
    payload_bits,
    deadline_ms,
):
    """Sample directed timely packet receptions for fixed UAV positions.

    The returned axes are ``(round, environment, sender, receiver)``. Self
    entries are false because running-sum retains the self-share locally.
    """
    positions = np.asarray(positions)
    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError("positions must have shape (environments, UAVs, >=2)")
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    if bandwidth_hz <= 0 or spectral_efficiency <= 0 or reference_distance_m <= 0:
        raise ValueError(
            "bandwidth_hz, spectral_efficiency, and reference_distance_m must be positive"
        )

    xy = positions[..., :2]
    distances = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=-1)
    distances = np.maximum(distances, np.finfo(np.float64).tiny)
    diagonal = np.arange(positions.shape[1])
    distances[:, diagonal, diagonal] = reference_distance_m

    sample_shape = (rounds,) + distances.shape
    k_linear = 10.0 ** (rician_k_db / 10.0)
    scatter = (
        rng.normal(size=sample_shape) + 1j * rng.normal(size=sample_shape)
    ) / np.sqrt(2.0)
    fading = np.abs(
        np.sqrt(k_linear / (k_linear + 1.0))
        + np.sqrt(1.0 / (k_linear + 1.0)) * scatter
    ) ** 2

    reference_gain = 10.0 ** (reference_gain_db / 10.0)
    channel_gain = (
        reference_gain
        * (distances[None, ...] / reference_distance_m) ** (-path_loss_exponent)
        * fading
    )
    noise_psd_w_hz = 10.0 ** ((noise_psd_dbm_hz - 30.0) / 10.0)
    snr = transmit_power_w * channel_gain / (noise_psd_w_hz * bandwidth_hz)
    decoded = snr >= 10.0 ** (decoding_threshold_db / 10.0)

    transmission_ms = 1000.0 * payload_bits / (bandwidth_hz * spectral_efficiency)
    additional_latency_ms = rng.gamma(
        shape=gamma_shape, scale=gamma_scale_ms, size=sample_shape
    )
    timely = transmission_ms + additional_latency_ms <= deadline_ms
    receptions = decoded & timely

    receptions[:, :, diagonal, diagonal] = False
    return receptions


def running_sum_ratio_consensus(local_advantages, receptions):
    """Apply finite-round running-sum ratio consensus.

    ``local_advantages`` has axes ``(UAV, time, environment, 1)`` and
    ``receptions`` has axes ``(round, environment, sender, receiver)``.
    The returned array has the same shape and dtype as ``local_advantages``.
    """
    local_advantages = np.asarray(local_advantages)
    receptions = np.asarray(receptions, dtype=bool)
    if local_advantages.ndim != 4 or local_advantages.shape[-1] != 1:
        raise ValueError(
            "local_advantages must have shape (UAVs, time, environments, 1)"
        )

    num_uavs, _, num_envs, _ = local_advantages.shape
    expected = (num_envs, num_uavs, num_uavs)
    if receptions.ndim != 4 or receptions.shape[1:] != expected:
        raise ValueError(
            "receptions must have shape (rounds, environments, sender, receiver)"
        )

    advantages = np.transpose(local_advantages[..., 0], (2, 0, 1))
    zeta = np.concatenate(
        [advantages.astype(np.float64, copy=False), np.ones((num_envs, num_uavs, 1))],
        axis=-1,
    )
    sigma = np.zeros_like(zeta)
    rho = np.zeros((num_envs, num_uavs, num_uavs, zeta.shape[-1]), dtype=np.float64)

    for received in receptions:
        sigma += zeta / num_uavs
        updated_rho = np.where(received[..., None], sigma[:, :, None, :], rho)
        zeta = zeta / num_uavs + np.sum(updated_rho - rho, axis=1)
        rho = updated_rho

    estimate = zeta[..., :-1] / zeta[..., -1:]
    estimate = np.transpose(estimate, (1, 2, 0))[..., None]
    return estimate.astype(local_advantages.dtype, copy=False)


def relative_estimation_error(local_advantages, communication_estimate):
    """Return the exact all-UAV mean and the communication-error noise scale."""
    exact_mean = np.mean(local_advantages, axis=0)
    numerator = np.abs(communication_estimate - exact_mean)
    denominator = np.abs(local_advantages - exact_mean)
    noise_scale = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator != 0,
    )
    return exact_mean, noise_scale
