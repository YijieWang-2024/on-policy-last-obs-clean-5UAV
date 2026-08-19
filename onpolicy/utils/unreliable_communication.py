import numpy as np


def communication_distance_from_power(
    transmit_power_w,
    *,
    bandwidth_hz,
    reference_gain_db,
    reference_distance_m,
    path_loss_exponent,
    noise_psd_dbm_hz,
    decoding_threshold_db,
):
    """Return the nominal A2A decoding radius implied by transmit power."""
    values = (
        transmit_power_w, bandwidth_hz, reference_gain_db, reference_distance_m,
        path_loss_exponent, noise_psd_dbm_hz, decoding_threshold_db,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("A2A link-budget parameters must be finite")
    if transmit_power_w <= 0:
        raise ValueError("transmit_power_w must be positive")
    if bandwidth_hz <= 0 or reference_distance_m <= 0 or path_loss_exponent <= 0:
        raise ValueError(
            "bandwidth_hz, reference_distance_m, and path_loss_exponent must be positive"
        )
    reference_gain = 10.0 ** (reference_gain_db / 10.0)
    noise_psd_w_hz = 10.0 ** ((noise_psd_dbm_hz - 30.0) / 10.0)
    decoding_threshold = 10.0 ** (decoding_threshold_db / 10.0)
    link_budget = (
        transmit_power_w * reference_gain
        / (noise_psd_w_hz * bandwidth_hz * decoding_threshold)
    )
    return float(reference_distance_m * link_budget ** (1.0 / path_loss_exponent))


def transmit_power_from_communication_distance(
    distance_m,
    *,
    bandwidth_hz,
    reference_gain_db,
    reference_distance_m,
    path_loss_exponent,
    noise_psd_dbm_hz,
    decoding_threshold_db,
):
    """Return the power whose nominal decoding radius is ``distance_m``."""
    values = (
        distance_m, bandwidth_hz, reference_gain_db, reference_distance_m,
        path_loss_exponent, noise_psd_dbm_hz, decoding_threshold_db,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("A2A link-budget parameters must be finite")
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if bandwidth_hz <= 0 or reference_distance_m <= 0 or path_loss_exponent <= 0:
        raise ValueError(
            "bandwidth_hz, reference_distance_m, and path_loss_exponent must be positive"
        )
    reference_gain = 10.0 ** (reference_gain_db / 10.0)
    noise_psd_w_hz = 10.0 ** ((noise_psd_dbm_hz - 30.0) / 10.0)
    decoding_threshold = 10.0 ** (decoding_threshold_db / 10.0)
    path_loss = (distance_m / reference_distance_m) ** path_loss_exponent
    return float(
        decoding_threshold * noise_psd_w_hz * bandwidth_hz * path_loss
        / reference_gain
    )


def resolve_communication_parameters(args, *, default_distance_m=520.0):
    """Resolve ``P_c`` and ``d_com`` from either value and validate both."""
    distance = getattr(args, "neighbor_distance", None)
    power = getattr(args, "a2a_transmit_power_w", None)
    tolerance = float(getattr(args, "a2a_distance_tolerance_m", 5.0))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("a2a_distance_tolerance_m must be finite and non-negative")
    if distance is not None and not np.isfinite(float(distance)):
        raise ValueError("neighbor_distance/d_com must be finite")
    if power is not None and not np.isfinite(float(power)):
        raise ValueError("a2a_transmit_power_w must be finite")
    if distance is not None and float(distance) < 0:
        raise ValueError("neighbor_distance/d_com must be non-negative")

    channel = {
        "bandwidth_hz": args.a2a_bandwidth_hz,
        "reference_gain_db": args.a2a_reference_gain_db,
        "reference_distance_m": args.a2a_reference_distance_m,
        "path_loss_exponent": args.a2a_path_loss_exponent,
        "noise_psd_dbm_hz": args.a2a_noise_psd_dbm_hz,
        "decoding_threshold_db": args.a2a_decoding_threshold_db,
    }
    if distance is not None and float(distance) == 0.0:
        if power is not None and float(power) != 0.0:
            raise ValueError("d_com=0 is only consistent with P_c=0")
        # Preserve the established R=0/no-communication ablation. Physical
        # packet sampling with P_c=0 then deterministically receives nothing.
        args.neighbor_distance = 0.0
        args.a2a_transmit_power_w = 0.0
        return args
    if distance is None and power is None:
        distance = float(default_distance_m)
        power = transmit_power_from_communication_distance(distance, **channel)
    elif distance is None:
        distance = communication_distance_from_power(float(power), **channel)
    elif power is None:
        distance = float(distance)
        power = transmit_power_from_communication_distance(distance, **channel)
    else:
        distance = float(distance)
        derived_distance = communication_distance_from_power(float(power), **channel)
        if abs(derived_distance - distance) > tolerance:
            raise ValueError(
                "inconsistent A2A parameters: --a2a_transmit_power_w implies "
                f"d_com={derived_distance:.3f} m, but --neighbor_distance/--d_com "
                f"is {distance:.3f} m (tolerance {tolerance:.3f} m)"
            )

    args.neighbor_distance = float(distance)
    args.a2a_transmit_power_w = float(power)
    return args


def communication_range_mask(positions, distance_m):
    """Return directed off-diagonal links inside the tunable ``d_com`` graph."""
    positions = np.asarray(positions)
    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError("positions must have shape (environments, UAVs, >=2)")
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("distance_m must be finite and non-negative")
    if distance_m == 0:
        return np.zeros(positions.shape[:2] + (positions.shape[1],), dtype=bool)
    xy = positions[..., :2]
    distances = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=-1)
    mask = distances <= float(distance_m)
    diagonal = np.arange(positions.shape[1])
    mask[:, diagonal, diagonal] = False
    return mask


def sample_configured_timely_receptions(
    positions, rounds, rng, args, *, payload_bits, deadline_ms
):
    """Sample packet receptions using the A2A parameters in training args."""
    return sample_timely_receptions(
        positions,
        rounds,
        rng,
        transmit_power_w=args.a2a_transmit_power_w,
        bandwidth_hz=args.a2a_bandwidth_hz,
        reference_gain_db=args.a2a_reference_gain_db,
        reference_distance_m=args.a2a_reference_distance_m,
        path_loss_exponent=args.a2a_path_loss_exponent,
        rician_k_db=args.a2a_rician_k_db,
        noise_psd_dbm_hz=args.a2a_noise_psd_dbm_hz,
        spectral_efficiency=args.a2a_spectral_efficiency,
        decoding_threshold_db=args.a2a_decoding_threshold_db,
        gamma_shape=args.a2a_gamma_shape,
        gamma_scale_ms=args.a2a_gamma_scale_ms,
        payload_bits=payload_bits,
        deadline_ms=deadline_ms,
        max_distance_m=args.neighbor_distance,
    )


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
    max_distance_m=None,
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
    if max_distance_m is not None:
        receptions &= communication_range_mask(positions, max_distance_m)[None, ...]

    receptions[:, :, diagonal, diagonal] = False
    return receptions


def running_sum_ratio_consensus(local_advantages, receptions, adjacency=None):
    """Apply finite-round running-sum ratio consensus.

    ``local_advantages`` has axes ``(UAV, time, environment, 1)`` and
    ``receptions`` has axes ``(round, environment, sender, receiver)``.
    ``adjacency`` is the range graph's off-diagonal directed mask with axes
    ``(environment, sender, receiver)``. Omitting it preserves the legacy
    complete-graph contract.
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
    if adjacency is None:
        adjacency = np.ones((num_envs, num_uavs, num_uavs), dtype=bool)
        diagonal = np.arange(num_uavs)
        adjacency[:, diagonal, diagonal] = False
    else:
        adjacency = np.asarray(adjacency, dtype=bool)
        if adjacency.shape != (num_envs, num_uavs, num_uavs):
            raise ValueError(
                "adjacency must have shape (environments, sender, receiver)"
            )
        adjacency = adjacency.copy()
        diagonal = np.arange(num_uavs)
        adjacency[:, diagonal, diagonal] = False
    receptions = receptions & adjacency[None, ...]
    out_degrees = np.sum(adjacency, axis=-1)
    share_count = out_degrees.astype(np.float64)[..., None] + 1.0

    advantages = np.transpose(local_advantages[..., 0], (2, 0, 1))
    zeta = np.concatenate(
        [advantages.astype(np.float64, copy=False), np.ones((num_envs, num_uavs, 1))],
        axis=-1,
    )
    sigma = np.zeros_like(zeta)
    rho = np.zeros((num_envs, num_uavs, num_uavs, zeta.shape[-1]), dtype=np.float64)

    for received in receptions:
        sigma += zeta / share_count
        updated_rho = np.where(received[..., None], sigma[:, :, None, :], rho)
        zeta = zeta / share_count + np.sum(updated_rho - rho, axis=1)
        rho = updated_rho

    estimate = zeta[..., :-1] / zeta[..., -1:]
    estimate = np.transpose(estimate, (1, 2, 0))[..., None]
    return estimate.astype(local_advantages.dtype, copy=False)


def normalized_consensus_residual(
    local_advantages, communication_estimate, epsilon=1e-8
):
    """Return the exact mean and one bounded L2 residual per rollout sample."""
    exact_mean = np.mean(local_advantages, axis=0)
    target = exact_mean[None, ...]
    denominator = np.linalg.norm(
        local_advantages - target, axis=0, keepdims=True
    )
    numerator = np.linalg.norm(
        communication_estimate - target, axis=0, keepdims=True
    )
    residual = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > epsilon,
    )
    return exact_mean, np.clip(residual, 0.0, 1.0)
