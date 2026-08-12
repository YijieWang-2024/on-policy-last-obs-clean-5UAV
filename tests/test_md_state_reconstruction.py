import copy

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.mec.mec import MEC
from onpolicy.scripts.train.train_mec import parse_args
from onpolicy.utils.md_roster import order_md_candidates
from onpolicy.utils.md_state_reconstruction import MDStateReconstructor, _Reservoir


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
        "continuous_associate": True,
        "not_served_rew_to_nearest": True,
        "communication_mode": "unreliable",
        "state_reconstruction": mode,
        "critic_md_metadata": True,
        "episode_length": 3,
        "md_gru_hidden_dim": 8,
        "md_gru_epochs": 2,
        "md_gru_batch_size": 8,
        "md_gru_max_samples": 128,
    }
    settings.update(overrides)
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
    assert np.count_nonzero(reconstructor.banks[0].ids[0] == 20) == 1

    empty = _synthetic_data(args)
    empty["record_features"][:] = 0
    empty["session_ids"][:] = -1
    empty["record_valid"][:] = False
    empty["visible_count"][:] = 0
    empty["critic_prefix"][..., -1] = 0
    reconstructor.reconstruct(empty)
    assert 20 not in reconstructor.banks[0].ids[0]


def test_reobserved_id_is_not_cleared_when_predicted_lifetime_reaches_zero():
    args = _args(mode="md_gru")
    data = _synthetic_data(args, lifetime=1)
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor.reconstruct(copy.deepcopy(data))
    bank = reconstructor.banks[0]
    slot_before = bank.find(0, 20)
    hidden_before = bank.hidden[0, slot_before].clone()

    refreshed = copy.deepcopy(data)
    refreshed["record_features"][0, 1, 0, 0] = 101
    reconstructor.reconstruct(refreshed)

    slot_after = bank.find(0, 20)
    assert slot_after == slot_before
    assert bank.lifetime[0, slot_after] == 1
    assert not torch.equal(hidden_before, bank.hidden[0, slot_after])


def test_receiver_memory_releases_expired_slots_before_replacements():
    args = _args()
    reconstructor = MDStateReconstructor(args, torch.device("cpu"))
    reconstructor._ensure_banks(1)
    bank = reconstructor.banks[0]
    bank.ids[0] = np.arange(args.n_GUs)
    bank.lifetime[0] = 10
    bank.lifetime[0, -6:] = 1

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
    assert not reconstructor.predictor_ready
    bank = reconstructor.banks[0]
    remembered_slot = bank.find(0, 20)
    np.testing.assert_allclose(
        bank.estimate[0, remembered_slot], bank.last_feature[0, remembered_slot]
    )
    assert not any(sample[0] == 0 for sample in reconstructor.reservoir.items)

    received = copy.deepcopy(lost)
    received["reception_mask"][0, 0, 1] = True
    received["record_features"][0, 1, 0, 0] = 120
    reconstructor.reconstruct(received)
    assert sum(sample[0] == 0 for sample in reconstructor.reservoir.items) == 1

    before = [parameter.detach().clone() for parameter in reconstructor.predictors[0].parameters()]
    metrics = reconstructor.train_predictors()
    assert reconstructor.predictor_ready
    assert metrics[0]["md_prediction_samples"] == 1
    assert np.isfinite(metrics[0]["md_prediction_loss"])
    assert np.isfinite(metrics[0]["md_prediction_position_rmse_m"])
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, reconstructor.predictors[0].parameters())
    )


def test_reservoir_selects_before_context_transfer_when_capacity_is_full():
    reservoir = _Reservoir(capacity=1, seed=3)
    selected, destinations = reservoir.plan_batch(1000)
    context_hidden = np.zeros((len(selected), 8), dtype=np.float32)
    context_input = np.zeros((len(selected), 8), dtype=np.float32)
    ages = np.zeros((len(selected), 1), dtype=np.float32)
    targets = np.zeros((len(selected), 7), dtype=np.float32)
    reservoir.commit_batch(
        destinations, 0, context_hidden, context_input, ages, targets
    )

    assert reservoir.seen == 1000
    assert reservoir.copied == 1
    assert len(reservoir.items) == 1
    assert reservoir.items[0] is not None


def test_reservoir_commits_last_source_for_repeated_batch_destinations():
    class FixedDraws:
        @staticmethod
        def integers(_low, high):
            assert len(high) == 4
            return np.asarray([0, 0, 1, 0], dtype=np.int64)

    reservoir = _Reservoir(capacity=2, seed=0)
    reservoir.items = [("old-0",), ("old-1",)]
    reservoir.seen = 2
    reservoir.rng = FixedDraws()
    selected, destinations = reservoir.plan_batch(4)
    markers = np.arange(4, dtype=np.float32)[:, None]
    reservoir.commit_batch(
        destinations,
        0,
        markers[selected],
        markers[selected],
        markers[selected],
        markers[selected],
    )

    assert reservoir.items[0][-1].item() == 3
    assert reservoir.items[1][-1].item() == 2
