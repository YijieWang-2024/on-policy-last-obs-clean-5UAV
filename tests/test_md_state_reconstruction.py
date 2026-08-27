import copy
from unittest.mock import patch

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.mec.mec import MEC
from onpolicy.scripts.train.train_mec import parse_args
from onpolicy.utils.md_roster import order_md_candidates
from onpolicy.utils.md_state_reconstruction import (
    MDGRUPredictor,
    MDStateReconstructor,
    _DenseMemoryBank,
    _RolloutSessionBuffer,
)


def _args(mode="last_obs", **overrides):
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
        "critic_neighbor_distance": 1000,
        "continuous_associate": True,
        "not_served_rew_to_nearest": True,
        "communication_mode": "unreliable",
        "state_reconstruction": mode,
        "critic_md_metadata": True,
        "episode_length": 3,
        "md_gru_hidden_dim": 8,
        "md_gru_target_batch_size": 8,
        "md_gru_batches_per_rollout": 2,
        "md_gru_min_ready_samples": 1,
    }
    settings.update(overrides)
    if "critic_neighbor_distance" not in overrides:
        settings["critic_neighbor_distance"] = settings["neighbor_distance"]
    for name, value in settings.items():
        setattr(args, name, value)
    return args


def _batched(data):
    return {name: np.expand_dims(value, 0) for name, value in data.items()}


def _direct_block(data, sender):
    valid = data["record_valid"][0, sender].astype(np.float64)
    metadata = np.stack((valid, valid, np.zeros_like(valid)), axis=-1)
    return np.concatenate((
        data["critic_prefix"][0, sender],
        data["record_features"][0, sender].reshape(-1),
        metadata.reshape(-1),
    ))


def _synthetic_data(args, lifetime=5):
    uavs = args.n_UAVs
    capacity = args.max_GUs_in_range
    positions = np.asarray(
        [[0, 0], [100, 0], [200, 0], [300, 0], [400, 0]],
        dtype=np.float64,
    )
    prefix = np.concatenate((positions, np.zeros((uavs, 1))), axis=1)
    records = np.zeros((uavs, capacity, 10), dtype=np.float64)
    session_ids = np.full((uavs, capacity), -1, dtype=np.int64)
    valid = np.zeros((uavs, capacity), dtype=bool)
    record = np.asarray(
        [100, 20, 5, 0.2, 0.2, lifetime, 1e-10, 2e6, 5e8, 1.0],
        dtype=np.float64,
    )
    records[1, 0] = record
    session_ids[1, 0] = 20
    valid[1, 0] = True
    prefix[1, -1] = 1
    geometric = np.ones((uavs, uavs), dtype=bool)
    reception = np.ones((uavs, uavs), dtype=bool)
    np.fill_diagonal(reception, False)
    return {
        "critic_prefix": prefix[None],
        "record_features": records[None],
        "session_ids": session_ids[None],
        "record_valid": valid[None],
        "visible_count": valid.sum(axis=1, dtype=np.int32)[None],
        "uav_positions": positions[None],
        "reception_mask": reception[None],
        "geometric_mask": geometric[None],
        "payload_schema_bits": np.asarray([8000], dtype=np.int32),
    }


def test_type_s_codec_matches_canonical_critic_block_elementwise():
    args = _args()
    env = MEC(args)
    env.seed(31)
    obs, _, _, _, _ = env.reset()
    critic_local = env.get_critic_local_obs(obs)
    data = _batched(env.get_type_s_data())

    for sender in range(args.n_UAVs):
        np.testing.assert_allclose(_direct_block(data, sender), critic_local[sender])

    data["geometric_mask"][:] = True
    data["reception_mask"][:] = True
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    states, attention = reconstructor.reconstruct(data)
    for receiver in range(args.n_UAVs):
        order = [receiver] + [i for i in range(args.n_UAVs) if i != receiver]
        expected = np.concatenate([_direct_block(data, sender) for sender in order])
        np.testing.assert_allclose(states[0, receiver], expected)
        np.testing.assert_array_equal(attention[0, receiver], 1)


def test_type_s_data_keeps_dcom_attempts_but_filters_critic_sources():
    args = _args(
        neighbor_distance=520.0,
        critic_neighbor_distance=260.0,
    )
    env = MEC(args)
    env.seed(37)
    env.reset()
    env.uav_positions[:, :2] = np.asarray([
        [0.0, 0.0],
        [200.0, 0.0],
        [400.0, 0.0],
        [600.0, 0.0],
        [800.0, 0.0],
    ])
    env._update_distance_matrices()
    all_success = np.ones((1, 1, env.n_UAVs, env.n_UAVs), dtype=bool)
    env.state_packets_attempted = 0
    env.state_packets_received = 0

    with patch(
        "onpolicy.envs.mec.mec.sample_configured_timely_receptions",
        return_value=all_success,
    ):
        env.last_state_reception_mask = env._sample_state_reception_mask()

    data = env.get_type_s_data()

    assert env.state_packets_attempted == 14
    assert data["reception_mask"][0, 2]
    assert data["geometric_mask"][0, 1]
    assert not data["geometric_mask"][0, 2]


def test_speed_layout_context_and_actor_message_do_not_change_type_s_codec():
    args = _args(
        md_arrivals_min=5,
        md_arrivals_max=5,
        md_arrivals_per_region=[1, 4],
        md_lifetime_min=12,
        md_lifetime_max=12,
        hotspot_layout_mode="episode_template4_600_200",
        episode_layout_context=True,
        episode_layout_context_units="meters_v2",
        actor_message_mode="task_summary",
        actor_message_pool="receiver_gated_sum",
        actor_message_contract="absolute_raw_v2",
        neighbor_distance=520,
        spatial_flight_actor=True,
        completion_priority_user_sort=True,
    )
    env = MEC(args)
    env.seed(41)
    obs, _, _, _, _ = env.reset()
    critic_local = env.get_critic_local_obs(obs)
    data = _batched(env.get_type_s_data())

    assert data["critic_prefix"].shape[-1] == (
        int(args.ob_state_with_timestep)
        + (args.n_UAVs if args.ob_state_with_id else 0)
        + 2
        + 8
        + 1
    )
    for sender in range(args.n_UAVs):
        np.testing.assert_allclose(_direct_block(data, sender), critic_local[sender])

    data["geometric_mask"][:] = True
    data["reception_mask"][:] = True
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    states, _ = reconstructor.reconstruct(data)
    for receiver in range(args.n_UAVs):
        order = [receiver] + [i for i in range(args.n_UAVs) if i != receiver]
        expected = np.concatenate([_direct_block(data, sender) for sender in order])
        np.testing.assert_allclose(states[0, receiver], expected)


def test_reliable_and_unreliable_modes_keep_actor_input_identical():
    common = dict(
        md_arrivals_min=5,
        md_arrivals_max=5,
        md_arrivals_per_region=[1, 4],
        md_lifetime_min=12,
        md_lifetime_max=12,
        hotspot_layout_mode="episode_template4_600_200",
        episode_layout_context=True,
        episode_layout_context_units="meters_v2",
        actor_message_mode="task_summary",
        actor_message_pool="receiver_gated_sum",
        actor_message_contract="absolute_raw_v2",
        neighbor_distance=520,
        spatial_flight_actor=True,
        completion_priority_user_sort=True,
    )
    reliable = MEC(_args(mode="zero", communication_mode="reliable", **common))
    unreliable = MEC(_args(mode="zero", communication_mode="unreliable", **common))
    reliable.seed(43)
    reliable_obs, _, _, _, _ = reliable.reset()
    unreliable.seed(43)
    unreliable_obs, _, _, _, _ = unreliable.reset()

    np.testing.assert_allclose(reliable_obs, unreliable_obs)


def test_six_uav_unreliable_gru_uses_generic_state_and_predictor_shapes():
    args = _args(
        mode="md_gru",
        n_UAVs=6,
        n_GUs=72,
        max_UAVs_obs_concat=6,
        max_UAVs_in_neighbor=6,
        md_arrivals_min=6,
        md_arrivals_max=6,
        md_arrivals_per_region=[1, 5],
        md_lifetime_min=12,
        md_lifetime_max=12,
        hotspot_layout_mode="episode_template4_600_200",
        uav_start_positions=[
            110, 180, 220, 180, 330, 180,
            440, 180, 550, 180, 400, 400,
        ],
        neighbor_distance=520,
        episode_layout_context=True,
        episode_layout_context_units="meters_v2",
    )
    env = MEC(args)
    env.seed(47)
    obs, state, _, _, attention = env.reset()

    assert obs.shape[0] == 6
    assert state.shape == (6, env.state_dim)
    assert attention.shape == (6, 6)
    np.testing.assert_allclose(env.uav_positions[:, :2], np.asarray(
        [[110, 180], [220, 180], [330, 180],
         [440, 180], [550, 180], [400, 400]],
        dtype=np.float32,
    ))
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructed, reconstructed_attention = reconstructor.reconstruct(
        _batched(env.get_type_s_data())
    )
    assert len(reconstructor.predictors) == 6
    assert reconstructor.bank.ids.shape[0] == 6
    assert reconstructed.shape == (1, 6, env.state_dim)
    assert reconstructed_attention.shape == (1, 6, 6)
    assert np.all(np.isfinite(reconstructed))

    action_dim = sum(int(np.prod(space.shape)) for space in env.action_space.spaces)
    action = np.random.default_rng(53).uniform(
        0.1, 0.9, (args.n_UAVs, action_dim)
    ).astype(np.float32)
    next_obs, rewards, _, next_state, _, _, _, next_attention = env.step(action)
    assert next_obs.shape[0] == 6
    assert rewards.shape[0] == 6
    assert next_state.shape == (6, env.state_dim)
    assert next_attention.shape == (6, 6)
    next_reconstructed, _ = reconstructor.reconstruct(
        _batched(env.get_type_s_data())
    )
    assert np.all(np.isfinite(next_reconstructed))


def test_heterogeneous_resources_preserve_total_budget_under_unreliable_mode():
    factors = np.asarray([1.0, 1.3, 0.7, 1.3, 0.7])
    args = _args(
        mode="md_gru",
        uav_resource_mode="heterogeneous",
        uav_resource_scale_factors=factors,
    )
    env = MEC(args)

    np.testing.assert_allclose(env.uav_bandwidth_capacities, env.B * factors)
    np.testing.assert_allclose(env.uav_compute_capacities, env.F_m * factors)
    np.testing.assert_allclose(env.uav_bandwidth_capacities.sum(), env.B * 5)
    np.testing.assert_allclose(env.uav_compute_capacities.sum(), env.F_m * 5)


def test_receiver_local_md_gru_checkpoint_round_trip_preserves_readiness():
    args = _args(mode="md_gru")
    source = MDStateReconstructor(args, torch.device("cpu"))
    source.predictor_ready = [True] * args.n_UAVs
    state = source.checkpoint_state()
    target = MDStateReconstructor(args, torch.device("cpu"))

    target.load_checkpoint_state(state)

    assert all(target.predictor_ready)
    for source_predictor, target_predictor in zip(
        source.predictors, target.predictors
    ):
        for source_parameter, target_parameter in zip(
            source_predictor.parameters(), target_predictor.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)


def test_new_md_gru_head_starts_as_exact_last_obs_residual():
    torch.manual_seed(3)
    predictor = MDGRUPredictor(feature_dim=7, hidden_dim=8)
    hidden = torch.randn(11, 8)
    age = torch.rand(11, 1)
    baseline = torch.randn(11, 7)

    prediction = predictor.predict(hidden, age, baseline)

    torch.testing.assert_close(prediction, baseline, rtol=0, atol=0)


def test_receiver_batched_online_gru_matches_independent_models():
    torch.manual_seed(59)
    args = _args(mode="md_gru")
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    receivers = np.asarray([0, 0, 1, 3, 3, 3, 4], dtype=np.int64)
    features = torch.randn(len(receivers), reconstructor.feature_dim)
    ages = torch.rand(len(receivers), 1)
    hidden = torch.randn(len(receivers), reconstructor.hidden_dim)

    expected_update = torch.empty_like(hidden)
    for receiver, predictor in enumerate(reconstructor.predictors):
        selected = receivers == receiver
        if np.any(selected):
            expected_update[selected] = predictor.update(
                features[selected], ages[selected], hidden[selected]
            )
    actual_update = reconstructor._online_update(
        receivers, features.numpy(), ages.numpy(), hidden
    )
    torch.testing.assert_close(actual_update, expected_update)

    reconstructor.residual_prediction = [True, False, True, False, True]
    baseline = torch.randn(len(receivers), reconstructor.feature_dim)
    expected_prediction = torch.empty_like(baseline)
    for receiver, predictor in enumerate(reconstructor.predictors):
        selected = receivers == receiver
        if np.any(selected):
            receiver_baseline = (
                baseline[selected]
                if reconstructor.residual_prediction[receiver]
                else None
            )
            expected_prediction[selected] = predictor.predict(
                hidden[selected], ages[selected], receiver_baseline
            )
    actual_prediction = reconstructor._online_predict(
        receivers, hidden, ages.numpy(), baseline.numpy()
    )
    torch.testing.assert_close(actual_prediction, expected_prediction)


def test_md_gru_speed_codec_covers_speed_branch_velocity_range():
    args = _args(
        mean_velocity=3.0,
        md_velocity_init_max_factor=1.6,
        md_velocity_update_clip_max=5.0,
    )
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    record = np.asarray(
        [100.0, 20.0, 5.0, 0.2, 0.2, 5.0, 1e-10, 2e6, 5e8, 1.0]
    )

    encoded = reconstructor._encode_feature(record)
    decoded = reconstructor._decode_feature(encoded)

    assert encoded[2] <= 1.0
    np.testing.assert_allclose(decoded[2], 5.0, atol=1e-6)


def test_packet_selection_uses_configured_sort_then_caps_at_twenty():
    for distance_only in (False, True):
        args = _args(distance_only_user_sort=distance_only)
        env = MEC(args)
        env.seed(37)
        env.reset()
        env.active_md_mask[:] = False
        env.active_md_mask[:25] = True
        env.md_session_ids[:] = -1
        env.md_session_ids[:25] = np.arange(100, 125)
        env.md_remaining_lifetime[:] = 0
        env.md_remaining_lifetime[:25] = 5
        env.gu_positions[:25, :2] = np.c_[np.arange(1, 26), np.zeros(25)]
        env.gu_positions[:25, 2] = env.H_GU
        env.gu_velocities[:25] = env.mean_velocity
        env.gu_directions[:25] = 0
        env.gu_directions_0[:25] = 0
        env.gu_tasks[:] = 0
        env.gu_tasks[:25, 0] = env.D_min
        env.gu_tasks[:12, 1] = 0.5 * env.F_n
        env.gu_tasks[:12, 2] = 1.0
        env.gu_tasks[12:25, 1] = 2.0 * env.F_n
        env.gu_tasks[12:25, 2] = 1.0
        env.uav_positions[0, :2] = 0
        env.uav_positions[1:, :2] = 600
        env._update_distance_matrices()
        env._calculate_channel_gains()
        env.nearby_gus_of_uavs = env.get_nearby_users_sorted_all()
        obs = env.get_local_obs()
        env.get_critic_local_obs(obs)
        packet = env.get_type_s_data()

        assert packet["visible_count"][0] == 25
        assert packet["record_valid"][0].sum() == 20
        expected = (
            np.arange(100, 120)
            if distance_only
            else np.r_[np.arange(112, 125), np.arange(100, 107)]
        )
        np.testing.assert_array_equal(packet["session_ids"][0], expected)


def test_completion_priority_roster_places_unknown_tasks_last_before_top_k():
    distances = np.r_[np.arange(8, 0, -1), np.arange(16, 8, -1), np.arange(22, 16, -1)]
    session_ids = np.arange(100, 122)
    task_valid = np.r_[np.ones(16, dtype=bool), np.zeros(6, dtype=bool)]
    can_finish = np.r_[np.zeros(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(6, dtype=bool)]

    order = order_md_candidates(
        distances, session_ids, can_finish, task_valid, distance_only=False
    )

    np.testing.assert_array_equal(order[:8], np.arange(7, -1, -1))
    np.testing.assert_array_equal(order[8:16], np.arange(15, 7, -1))
    np.testing.assert_array_equal(order[16:20], np.arange(21, 17, -1))


def test_lost_packet_reconstruction_does_not_read_lost_type_s_truth():
    args = _args()
    initial = _synthetic_data(args)
    left = MDStateReconstructor(args, torch.device("cpu"))
    right = MDStateReconstructor(args, torch.device("cpu"))
    left.reconstruct(copy.deepcopy(initial))
    right.reconstruct(copy.deepcopy(initial))

    current_left = copy.deepcopy(initial)
    current_right = copy.deepcopy(initial)
    current_left["reception_mask"][0, 0, 1] = False
    current_right["reception_mask"][0, 0, 1] = False
    current_left["critic_prefix"][0, 1, :2] = [500, 500]
    current_right["critic_prefix"][0, 1, :2] = [250, 450]
    current_left["record_features"][0, 1, 0, :2] = [510, 510]
    current_right["record_features"][0, 1, 0, :2] = [260, 460]
    current_left["session_ids"][0, 1, 0] = 999
    current_right["session_ids"][0, 1, 0] = 777

    left_state, _ = left.reconstruct(current_left)
    right_state, _ = right.reconstruct(current_right)
    np.testing.assert_allclose(left_state[0, 0], right_state[0, 0])


def test_lost_packet_reconstruction_uses_current_control_plane_position():
    args = _args()
    initial = _synthetic_data(args)
    near = MDStateReconstructor(args, torch.device("cpu"))
    far = MDStateReconstructor(args, torch.device("cpu"))
    near.reconstruct(copy.deepcopy(initial))
    far.reconstruct(copy.deepcopy(initial))

    current_near = copy.deepcopy(initial)
    current_far = copy.deepcopy(initial)
    current_near["reception_mask"][0, 0, 1] = False
    current_far["reception_mask"][0, 0, 1] = False
    current_far["uav_positions"][0, 1] = [500, 500]

    near_state, _ = near.reconstruct(current_near)
    far_state, _ = far.reconstruct(current_far)
    block_dim = near_state.shape[-1] // args.n_UAVs
    near_sender_block = near_state[0, 0, block_dim:2 * block_dim]
    far_sender_block = far_state[0, 0, block_dim:2 * block_dim]

    np.testing.assert_allclose(near_sender_block[:2], [100, 0])
    np.testing.assert_allclose(far_sender_block[:2], [500, 500])
    assert not np.allclose(near_sender_block, far_sender_block)


def test_surrogate_work_is_batched_only_for_missing_blocks():
    args = _args()
    data = _synthetic_data(args)
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor.reconstruct(copy.deepcopy(data))
    original = reconstructor._surrogate_blocks
    reconstructed_counts = []

    def record_count(packet, features, environments, receivers, senders):
        reconstructed_counts.append(len(environments))
        return original(packet, features, environments, receivers, senders)

    reconstructor._surrogate_blocks = record_count
    reconstructor.reconstruct(copy.deepcopy(data))
    assert reconstructed_counts == []

    lost = copy.deepcopy(data)
    lost["reception_mask"][0, 0, 1] = False
    lost["reception_mask"][0, 3, 2] = False
    reconstructor.reconstruct(lost)
    assert reconstructed_counts == [2]


def test_receiver_memory_deduplicates_ids_and_expires_by_lifetime():
    args = _args()
    data = _synthetic_data(args, lifetime=1)
    data["record_features"][0, 2, 0] = data["record_features"][0, 1, 0]
    data["record_features"][0, 2, 0, 6] *= 2
    data["session_ids"][0, 2, 0] = 20
    data["record_valid"][0, 2, 0] = True
    data["critic_prefix"][0, 2, -1] = 1
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor.reconstruct(data)
    assert np.count_nonzero(reconstructor.bank.ids[0] == 20) == 1

    empty = _synthetic_data(args)
    empty["record_features"][:] = 0
    empty["session_ids"][:] = -1
    empty["record_valid"][:] = False
    empty["visible_count"][:] = 0
    empty["critic_prefix"][..., -1] = 0
    reconstructor.reconstruct(empty)
    assert 20 not in reconstructor.bank.ids[0]


def test_reobserved_id_is_not_cleared_when_predicted_lifetime_reaches_zero():
    args = _args(mode="md_gru")
    data = _synthetic_data(args, lifetime=1)
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor.reconstruct(copy.deepcopy(data))
    bank = reconstructor.bank
    slot_before = bank.find(0, 20)
    hidden_before = bank.hidden[0, slot_before].clone()

    refreshed = copy.deepcopy(data)
    refreshed["record_features"][0, 1, 0, 0] = 101
    reconstructor.reconstruct(refreshed)

    slot_after = bank.find(0, 20)
    assert slot_after == slot_before
    # clock points to the next reconstruction slot after reconstruct() returns.
    assert bank.expire_at[0, slot_after] - (reconstructor.clock[0] - 1) == 1
    assert not torch.equal(hidden_before, bank.hidden[0, slot_after])


def test_receiver_memory_releases_expired_slots_before_replacements():
    args = _args()
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor._ensure_banks(1)
    bank = reconstructor.bank
    bank.ids[0] = np.arange(args.n_GUs)
    bank.free_count[0] = 0
    bank.expire_at[0] = reconstructor.clock[0] + 10
    bank.expire_at[0, -6:] = reconstructor.clock[0]

    data = _synthetic_data(args, lifetime=10)
    data["record_features"][:] = 0
    data["session_ids"][:] = -1
    data["record_valid"][:] = False
    data["visible_count"][:] = 0
    data["critic_prefix"][..., -1] = 0
    for slot, session_id in enumerate(range(100, 106)):
        data["record_features"][0, 1, slot] = np.asarray(
            [100 + slot, 20, 0.5, 0.2, 0.2, 10, 1e-10, 2e6, 5e8, 1.0]
        )
        data["session_ids"][0, 1, slot] = session_id
        data["record_valid"][0, 1, slot] = True
    data["visible_count"][0, 1] = 6
    data["critic_prefix"][0, 1, -1] = 6

    reconstructor.reconstruct(data)

    assert np.count_nonzero(bank.ids[0] != -1) == args.n_GUs
    assert not np.any(np.isin(np.arange(54, 60), bank.ids[0]))
    assert np.all(np.isin(np.arange(100, 106), bank.ids[0]))


def test_spatial_actor_and_surrogate_share_effective_distance_sort():
    args = _args(spatial_flight_actor=True)
    data = _synthetic_data(args)
    data["record_features"][0, 1, 0] = np.asarray(
        [105, 0, 0.5, 0.0, 0.0, 5, 1e-10, 2e6, 0.5e9, 1.0]
    )
    data["record_features"][0, 1, 1] = np.asarray(
        [110, 0, 0.5, 0.0, 0.0, 5, 1e-10, 2e6, 2.0e9, 1.0]
    )
    data["session_ids"][0, 1, :2] = [20, 21]
    data["record_valid"][0, 1, :2] = True
    data["visible_count"][0, 1] = 2
    data["critic_prefix"][0, 1, -1] = 2
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor.reconstruct(copy.deepcopy(data))

    lost = copy.deepcopy(data)
    lost["reception_mask"][0, 0, 1] = False
    state, _ = reconstructor.reconstruct(lost)
    block_dim = state.shape[-1] // args.n_UAVs
    sender_block = state[0, 0, block_dim:2 * block_dim]
    record_start = 3
    reconstructed_x = [
        sender_block[record_start],
        sender_block[record_start + 10],
    ]

    assert reconstructor.distance_only_user_sort
    np.testing.assert_allclose(reconstructed_x, [105, 110])


def test_md_gru_collects_only_reobservation_targets_and_trains():
    torch.manual_seed(5)
    args = _args(mode="md_gru")
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    first = _synthetic_data(args)
    reconstructor.reconstruct(first)

    lost = copy.deepcopy(first)
    lost["reception_mask"][0, 0, 1] = False
    lost["record_features"][0, 1, 0, 0] = 110
    reconstructor.reconstruct(lost)
    assert not reconstructor.predictor_ready[0]
    bank = reconstructor.bank
    remembered_slot = bank.find(0, 20)
    assert reconstructor.session_buffers[0].target_count() == 0

    received = copy.deepcopy(lost)
    received["reception_mask"][0, 0, 1] = True
    received["record_features"][0, 1, 0, 0] = 120
    reconstructor.reconstruct(received)
    assert reconstructor.session_buffers[0].target_count() == 1

    before = [parameter.detach().clone() for parameter in reconstructor.predictors[0].parameters()]
    metrics = reconstructor.train_predictors()
    assert reconstructor.predictor_ready[0]
    assert metrics[0]["md_prediction_samples"] == 1
    assert metrics[0]["md_prediction_replay_size"] == 1
    assert metrics[0]["md_prediction_replay_label_age2_fraction"] == 1.0
    assert metrics[0]["md_prediction_rollout_query_age1_fraction"] == 1.0
    assert np.isfinite(metrics[0]["md_prediction_loss"])
    assert np.isfinite(metrics[0]["md_prediction_position_rmse_m"])
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, reconstructor.predictors[0].parameters())
    )
    after = [
        parameter.detach().clone()
        for parameter in reconstructor.predictors[0].parameters()
    ]
    idle_metrics = reconstructor.train_predictors()
    assert idle_metrics[0]["md_prediction_samples"] == 0
    assert idle_metrics[0]["md_prediction_train_samples"] == 0
    assert all(
        torch.equal(old, new)
        for old, new in zip(after, reconstructor.predictors[0].parameters())
    )


def test_natural_reobservations_preserve_age_distribution_without_fake_masks():
    args = _args(mode="md_gru")
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    data = _synthetic_data(args)
    reconstructor.reconstruct(copy.deepcopy(data))

    # Consecutive reception -> age 1.
    reconstructor.reconstruct(copy.deepcopy(data))
    # One missed slot -> age 2.
    lost = copy.deepcopy(data)
    lost["reception_mask"][0, 0, 1] = False
    reconstructor.reconstruct(lost)
    reconstructor.reconstruct(copy.deepcopy(data))
    # Two missed slots -> age 3.
    reconstructor.reconstruct(copy.deepcopy(lost))
    reconstructor.reconstruct(copy.deepcopy(lost))
    reconstructor.reconstruct(copy.deepcopy(data))

    ages = reconstructor.session_buffers[0].age_values().tolist()
    assert ages == [1, 2, 3]
    metrics = reconstructor.train_predictors()[0]
    assert metrics["md_prediction_replay_label_count"] == 3
    assert metrics["md_prediction_replay_label_age1_fraction"] == 1 / 3
    assert metrics["md_prediction_replay_label_age2_fraction"] == 1 / 3
    assert metrics["md_prediction_replay_label_age3_fraction"] == 1 / 3


def test_rollout_session_buffer_clears_after_training_and_restarts_after_reset():
    args = _args(mode="md_gru")
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    data = _synthetic_data(args)
    reconstructor.reconstruct(copy.deepcopy(data))
    reconstructor.reconstruct(copy.deepcopy(data))
    reconstructor.train_predictors()
    assert reconstructor.predictor_ready[0]
    assert reconstructor.session_buffers[0].target_count() == 0
    reconstructor.reset(np.asarray([True]))
    reconstructor.reconstruct(copy.deepcopy(data), collect_samples=False)

    lost = copy.deepcopy(data)
    lost["reception_mask"][0, 0, 1] = False
    reconstructor.reconstruct(lost)
    received = copy.deepcopy(data)
    received["record_features"][0, 1, 0, 0] += 30.0
    reconstructor.reconstruct(received)
    metrics = reconstructor.train_predictors()[0]

    assert metrics["md_prediction_prequential_samples"] == 1
    assert np.isfinite(metrics["md_prediction_prequential_rmse_m"])
    assert np.isfinite(metrics["md_last_obs_prequential_rmse_m"])
    reconstructor.reset(np.asarray([True]))
    assert np.all(reconstructor.bank.ids == -1)
    assert reconstructor.session_buffers[0].target_count() == 0


def test_rollout_session_buffer_stores_one_sequence_without_prefix_copies():
    buffer = _RolloutSessionBuffer(1, 2, 6, 7)
    for marker, gap in enumerate((0, 1, 1, 3, 1, 3)):
        buffer.append([0], [1], [[marker] * 7], [gap])

    assert buffer.target_count() == 5
    assert buffer.session_pairs().tolist() == [[0, 1]]
    assert buffer.age_values().tolist() == [1, 1, 3, 1, 3]
    buffer.reset()
    assert buffer.target_count() == 0


def test_rollout_session_buffer_builds_disjoint_whole_session_target_batches():
    buffer = _RolloutSessionBuffer(1, 6, 6, 7)
    lengths = (2, 4, 3, 6, 2)
    for session_id, length in enumerate(lengths):
        for step in range(length):
            buffer.append(
                [0], [session_id], [[session_id, step, 0, 0, 0, 0, 0]],
                [0 if step == 0 else 1],
            )
    pairs = buffer.session_pairs()

    batches, used = buffer.target_batches(
        pairs, target_batch_size=4, max_batches=2
    )

    assert [buffer.target_count(batch) for batch in batches] == [4, 7]
    assert used == 4
    assert np.array_equal(np.concatenate(batches), pairs[:used])
    assert len({tuple(pair) for pair in np.concatenate(batches)}) == used


def test_md_gru_uses_disjoint_target_budget_batches_once_per_rollout():
    args = _args(
        mode="md_gru",
        md_gru_target_batch_size=2,
        md_gru_batches_per_rollout=2,
    )
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor._ensure_banks(1)
    buffer = reconstructor.session_buffers[0]
    for session_id in range(5):
        buffer.append([0], [session_id], [[0.0] * 7], [0])
        buffer.append([0], [session_id], [[0.1] + [0.0] * 6], [1])

    metrics = reconstructor.train_predictors()[0]

    assert metrics["md_prediction_train_samples"] == 4
    assert metrics["md_prediction_train_sessions"] == 4
    assert metrics["md_prediction_optimizer_steps"] == 2
    assert metrics["md_prediction_validation_samples"] == 1
    assert metrics["md_prediction_sampled_fraction"] == 0.8
    assert buffer.target_count() == 0


def test_dense_memory_allocates_unsorted_environments_in_one_batch():
    bank = _DenseMemoryBank(3, 4, 20, 7, 0, torch.device("cpu"))
    environments = np.asarray([2, 0, 2, 1, 0], dtype=np.int64)
    session_ids = np.asarray([12, 3, 11, 7, 4], dtype=np.int64)

    slots = bank.lookup_or_allocate(environments, session_ids)

    assert bank.hidden is None
    assert len(np.unique(slots[environments == 0])) == 2
    assert len(np.unique(slots[environments == 2])) == 2
    np.testing.assert_array_equal(
        bank.ids[environments, slots], session_ids
    )
    np.testing.assert_array_equal(
        bank.lookup_or_allocate(environments[::-1], session_ids[::-1]), slots[::-1]
    )


def test_receiver_local_predictors_diverge_after_local_training_only():
    torch.manual_seed(7)
    args = _args(mode="md_gru")
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor._ensure_banks(1)
    reconstructor.session_buffers[0].append(
        [0], [0], [[-0.5] + [0.0] * 6], [0]
    )
    reconstructor.session_buffers[0].append(
        [0], [0], [[-0.3] + [0.0] * 6], [2]
    )

    before_receiver_1 = [
        parameter.detach().clone()
        for parameter in reconstructor.predictors[1].parameters()
    ]
    before_receiver_0 = [
        parameter.detach().clone()
        for parameter in reconstructor.predictors[0].parameters()
    ]
    assert reconstructor.session_buffers[0].target_count() == 1
    assert reconstructor.session_buffers[1].target_count() == 0
    reconstructor.train_predictors()

    assert reconstructor.predictor_ready[0]
    assert not reconstructor.predictor_ready[1]
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before_receiver_0, reconstructor.predictors[0].parameters())
    )
    assert all(
        parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
        for parameter in reconstructor.predictors[0].parameters()
    )
    assert all(
        torch.equal(old, new)
        for old, new in zip(before_receiver_1, reconstructor.predictors[1].parameters())
    )


def test_md_gru_waits_for_minimum_rollout_targets_before_predictions_are_enabled():
    args = _args(
        mode="md_gru",
        md_gru_min_ready_samples=2,
        md_gru_target_batch_size=1,
    )
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor._ensure_banks(1)
    reconstructor.session_buffers[0].append([0], [0], [[0.0] * 7], [0])
    reconstructor.session_buffers[0].append(
        [0], [0], [[0.1] + [0.0] * 6], [1]
    )

    metrics = reconstructor.train_predictors()

    assert not reconstructor.predictor_ready[0]
    assert metrics[0]["md_prediction_rollout_target_count"] == 1
    assert metrics[0]["md_prediction_train_samples"] == 0


def test_md_gru_cli_defaults_to_rollout_local_2048_by_10_single_epoch():
    args = parse_args([], get_config())
    assert args.md_gru_target_batch_size == 2048
    assert args.md_gru_batches_per_rollout == 10
    assert args.md_gru_epochs == 1

    legacy_alias = parse_args(
        ["--md_gru_train_samples", "17"], get_config()
    )
    assert legacy_alias.md_gru_target_batch_size == 17
