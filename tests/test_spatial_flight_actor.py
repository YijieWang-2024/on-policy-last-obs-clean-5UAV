import unittest

import numpy as np
import torch

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.config import get_config
from onpolicy.envs.mec.mec import MEC
from onpolicy.envs.mec.vec_normalize import Normer, normalize_batch
from onpolicy.scripts.train.train_mec import parse_args


class SpatialFlightActorTest(unittest.TestCase):
    @staticmethod
    def make_args(**overrides):
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
            "continuous_associate": True,
            "not_served_rew_to_nearest": True,
            "cartesian_flight": True,
            "actor_neighbor_obs": False,
            "spatial_flight_actor": True,
            "completion_priority_user_sort": False,
            "episode_length": 400,
            "num_env_steps": 100_000_000,
            "n_rollout_threads": 64,
            "use_recurrent_policy": False,
            "use_naive_recurrent_policy": False,
        }
        settings.update(overrides)
        for name, value in settings.items():
            setattr(args, name, value)
        return args

    def test_flight_is_invariant_to_current_iid_tasks(self):
        torch.manual_seed(3)
        np.random.seed(3)
        args = self.make_args()
        env = MEC(args)
        obs, _, available, _, _ = env.reset()
        actor = R_Actor(args, env.observation_space, env.action_space)
        actor.eval()

        obs_a = torch.as_tensor(obs, dtype=torch.float32)
        obs_b = obs_a.clone()
        users = obs_b[:, 3:].reshape(args.n_UAVs, args.max_GUs_in_range, 10)
        users[:, :, 7:10] = torch.randn_like(users[:, :, 7:10]) * 1000
        available_tensor = torch.as_tensor(available, dtype=torch.float32)
        rnn = torch.zeros(args.n_UAVs, args.recurrent_N, args.hidden_size)
        masks = torch.ones(args.n_UAVs, 1)

        with torch.no_grad():
            actions_a, _, _ = actor(
                obs_a, rnn, masks, available_tensor.clone(), deterministic=True
            )
            actions_b, _, _ = actor(
                obs_b, rnn, masks, available_tensor.clone(), deterministic=True
            )
        torch.testing.assert_close(actions_a[:, :2], actions_b[:, :2])

    def test_flight_set_encoder_is_permutation_invariant(self):
        torch.manual_seed(5)
        args = self.make_args()
        env = MEC(args)
        obs, _, available, _, _ = env.reset()
        actor = R_Actor(args, env.observation_space, env.action_space)

        obs_a = torch.as_tensor(obs, dtype=torch.float32)
        available_a = torch.as_tensor(available, dtype=torch.float32)
        permutation = torch.randperm(args.max_GUs_in_range)
        obs_b = obs_a.clone()
        blocks = obs_b[:, 3:].reshape(args.n_UAVs, args.max_GUs_in_range, 10)
        obs_b[:, 3:] = blocks[:, permutation].reshape(args.n_UAVs, -1)
        available_b = available_a[:, permutation]

        with torch.no_grad():
            features_a = actor.flight_base(obs_a, available_a)
            features_b = actor.flight_base(obs_b, available_b)
        torch.testing.assert_close(features_a, features_b, rtol=1e-5, atol=1e-6)

    def test_spatial_mode_uses_task_independent_distance_order(self):
        args = self.make_args()
        env = MEC(args)
        env.seed(9)
        env.reset()
        before = env.nearby_gus_of_uavs.copy()
        env.gu_tasks[:, 1] = np.linspace(env.C_min, env.C_max, env.n_GUs)
        env.gu_tasks[:, 2] = np.where(np.arange(env.n_GUs) % 2, env.delay_min, env.delay_max)
        after = env.get_nearby_users_sorted_all()
        np.testing.assert_array_equal(before, after)

    def test_completion_priority_overrides_spatial_legacy_distance_sort(self):
        env = MEC(self.make_args(completion_priority_user_sort=True))
        self.assertFalse(env.distance_only_user_sort)
        env.reset()

        env.uav_gu_distances_2d[0, :4] = [10.0, 20.0, 30.0, 40.0]
        env.coverage_mask[0] = False
        env.coverage_mask[0, :4] = True
        env.gu_tasks[:4, 1] = [env.F_n, env.F_n, env.F_n, env.F_n]
        env.gu_tasks[:4, 2] = [2.0, 0.5, 2.0, 0.5]

        ordered = env.get_nearby_users_sorted_all()[0, :4]
        np.testing.assert_array_equal(ordered, [1, 3, 0, 2])

    def test_explicit_sort_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(AssertionError, "mutually exclusive"):
            MEC(self.make_args(
                completion_priority_user_sort=True,
                distance_only_user_sort=True,
            ))

    def test_spatial_flight_actor_supports_polar_action_contract(self):
        torch.manual_seed(13)
        args = self.make_args(
            cartesian_flight=False,
            completion_priority_user_sort=True,
        )
        env = MEC(args)
        obs, _, available, _, _ = env.reset()
        actor = R_Actor(args, env.observation_space, env.action_space)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        available = torch.as_tensor(available, dtype=torch.float32)
        rnn = torch.zeros(args.n_UAVs, args.recurrent_N, args.hidden_size)
        masks = torch.ones(args.n_UAVs, 1)

        with torch.no_grad():
            actions, rollout_log_prob, _ = actor(obs, rnn, masks, available.clone())
            evaluated_log_prob, _ = actor.evaluate_actions(
                obs, rnn, actions, masks, available.clone()
            )

        np.testing.assert_array_equal(env.flight_action_space.low, [0.0, 0.0])
        np.testing.assert_array_equal(env.flight_action_space.high, [1.0, 1.0])
        clipped_flight = np.clip(actions[:, :2].numpy(), 0.0, 1.0)
        self.assertTrue(np.all((clipped_flight >= 0.0) & (clipped_flight <= 1.0)))
        torch.testing.assert_close(rollout_log_prob, evaluated_log_prob)

    def test_spatial_flight_actor_accepts_episode_layout_context(self):
        torch.manual_seed(17)
        args = self.make_args(
            x_max_uav=700,
            y_max_uav=700,
            x_max_gu=700,
            y_max_gu=700,
            hotspot_layout_mode="episode_template12",
            hotspot_layout_indices=[0],
            episode_layout_context=True,
        )
        env = MEC(args)
        obs, _, available, _, _ = env.reset()
        self.assertEqual(env.layout_context_dim, 8)
        actor = R_Actor(args, env.observation_space, env.action_space)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        available = torch.as_tensor(available, dtype=torch.float32)
        rnn = torch.zeros(args.n_UAVs, args.recurrent_N, args.hidden_size)
        masks = torch.ones(args.n_UAVs, 1)

        with torch.no_grad():
            actions, log_prob, _ = actor(
                obs, rnn, masks, available.clone(), deterministic=True
            )

        self.assertEqual(actions.shape[0], args.n_UAVs)
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_curriculum_is_training_only_and_finishes_by_half_budget(self):
        training_args = self.make_args(
            uav_reset_curriculum=True,
            uav_reset_curriculum_training=True,
        )
        training_env = MEC(training_args)
        self.assertAlmostEqual(training_env._curriculum_random_reset_probability(), 0.5)
        training_env.curriculum_reset_count = int(np.ceil(
            0.50 * training_args.num_env_steps
            / (training_args.episode_length * training_args.n_rollout_threads)
        ))
        self.assertEqual(training_env._curriculum_random_reset_probability(), 0.0)

        training_env.curriculum_reset_count = 0
        training_env.seed(2)
        training_env.reset()
        fixed = np.array([[200, 525], [110, 70], [220, 70], [330, 70], [440, 70]])
        self.assertFalse(np.allclose(training_env.uav_positions[:, :2], fixed))
        distances = np.linalg.norm(
            training_env.uav_positions[:, None, :2]
            - training_env.uav_positions[None, :, :2], axis=2
        )
        self.assertTrue(np.all(distances[np.triu_indices(5, 1)] >= training_env.Cover_R))

        eval_args = self.make_args(
            uav_reset_curriculum=True,
            uav_reset_curriculum_training=False,
        )
        eval_env = MEC(eval_args)
        eval_env.reset()
        np.testing.assert_allclose(eval_env.uav_positions[:, :2], fixed)

    def test_actor_message_radius_endpoints_and_task_payload(self):
        zero_env = MEC(self.make_args(
            actor_message_mode="task_summary",
            neighbor_distance=0.0,
        ))
        zero_env.seed(2)
        zero_env.reset()
        np.testing.assert_array_equal(
            zero_env.get_actor_message_block(),
            np.zeros((5, 40), dtype=np.float32),
        )

        geometry_env = MEC(self.make_args(
            actor_message_mode="geometry",
            actor_message_contract="absolute_raw_v2",
            neighbor_distance=1000.0,
        ))
        geometry_env.seed(2)
        geometry_env.reset()
        geometry = geometry_env.get_actor_message_block().reshape(5, 4, 10)
        np.testing.assert_array_equal(geometry[:, :, 9], 1.0)
        np.testing.assert_array_equal(geometry[:, :, 2:9], 0.0)
        for receiver_id in range(geometry_env.n_UAVs):
            sender_ids = [
                sender_id for sender_id in range(geometry_env.n_UAVs)
                if sender_id != receiver_id
            ]
            np.testing.assert_allclose(
                geometry[receiver_id, :, :2],
                geometry_env.uav_positions[sender_ids, :2],
            )

        task_env = MEC(self.make_args(
            actor_message_mode="task_summary",
            actor_message_contract="absolute_raw_v2",
            neighbor_distance=1000.0,
        ))
        task_env.seed(2)
        task_env.reset()
        task_messages = task_env.get_actor_message_block().reshape(5, 4, 10)
        np.testing.assert_array_equal(task_messages[:, :, 9], 1.0)
        self.assertGreater(np.max(task_messages[:, :, 2]), 0.0)
        self.assertTrue(np.all(np.isfinite(task_messages)))
        for receiver_id in range(task_env.n_UAVs):
            sender_ids = [
                sender_id for sender_id in range(task_env.n_UAVs)
                if sender_id != receiver_id
            ]
            expected_counts = np.asarray([
                np.count_nonzero(task_env.nearby_gus_of_uavs[sender_id] != -1)
                for sender_id in sender_ids
            ])
            np.testing.assert_allclose(
                task_messages[receiver_id, :, 2], expected_counts
            )
            np.testing.assert_allclose(
                task_messages[receiver_id, :, :2],
                task_env.uav_positions[sender_ids, :2],
            )
            for slot, sender_id in enumerate(sender_ids):
                local_ids = task_env.nearby_gus_of_uavs[
                    sender_id, :task_env.max_GUs_in_range
                ]
                local_ids = local_ids[local_ids != -1].astype(np.int64)
                if not local_ids.size:
                    np.testing.assert_array_equal(
                        task_messages[receiver_id, slot, 3:9], 0.0
                    )
                    continue
                local_positions = task_env.gu_positions[local_ids, :2]
                centroid = np.mean(local_positions, axis=0)
                velocity_vectors = np.column_stack((
                    task_env.gu_velocities[local_ids]
                    * np.cos(task_env.gu_directions[local_ids]),
                    task_env.gu_velocities[local_ids]
                    * np.sin(task_env.gu_directions[local_ids]),
                ))
                spread = np.sqrt(np.mean(np.sum(
                    (local_positions - centroid) ** 2, axis=1
                )))
                hard_fraction = np.mean(
                    task_env.gu_tasks[local_ids, 1] / task_env.F_n
                    > task_env.gu_tasks[local_ids, 2]
                )
                np.testing.assert_allclose(
                    task_messages[receiver_id, slot, 3:5], centroid
                )
                np.testing.assert_allclose(
                    task_messages[receiver_id, slot, 5:7],
                    np.mean(velocity_vectors, axis=0),
                )
                np.testing.assert_allclose(
                    task_messages[receiver_id, slot, 7], spread
                )
                np.testing.assert_allclose(
                    task_messages[receiver_id, slot, 8], hard_fraction
                )

        observations_after_reset = task_env.actor_message_observation_count
        task_env.get_actor_message_block()
        self.assertEqual(
            task_env.actor_message_observation_count,
            observations_after_reset + 1,
        )
        task_env.reset()
        self.assertEqual(
            task_env.actor_message_observation_count,
            1,
        )

    def test_message_encoder_is_permutation_invariant_and_flight_only(self):
        torch.manual_seed(17)
        args = self.make_args(
            actor_message_mode="task_summary",
            neighbor_distance=1000.0,
        )
        env = MEC(args)
        env.seed(2)
        obs, _, available, _, _ = env.reset()
        actor = R_Actor(args, env.observation_space, env.action_space)
        obs_a = torch.as_tensor(obs, dtype=torch.float32)
        obs_b = obs_a.clone()
        start = 2
        messages = obs_b[:, start:start + 40].reshape(5, 4, 10)
        obs_b[:, start:start + 40] = messages[:, [2, 0, 3, 1]].reshape(5, 40)
        available = torch.as_tensor(available, dtype=torch.float32)
        rnn = torch.zeros(args.n_UAVs, args.recurrent_N, args.hidden_size)
        masks = torch.ones(args.n_UAVs, 1)

        with torch.no_grad():
            flight_a = actor.flight_base(obs_a, available)
            flight_b = actor.flight_base(obs_b, available)
            actions_a, _, _ = actor(
                obs_a, rnn, masks, available.clone(), deterministic=True
            )
            actions_b, _, _ = actor(
                obs_b, rnn, masks, available.clone(), deterministic=True
            )
            obs_c = obs_a.clone()
            obs_c[:, start:start + 40] = 0.0
            flight_c = actor.flight_base(obs_c, available)
            actions_c, _, _ = actor(
                obs_c, rnn, masks, available.clone(), deterministic=True
            )
        torch.testing.assert_close(flight_a, flight_b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actions_a[:, 2:], actions_b[:, 2:])
        self.assertFalse(torch.allclose(flight_a, flight_c))
        torch.testing.assert_close(actions_a[:, 2:], actions_c[:, 2:])

    def test_receiver_gated_message_pool_is_permutation_invariant(self):
        torch.manual_seed(19)
        args = self.make_args(
            actor_message_mode="task_summary",
            actor_message_pool="receiver_gated_sum",
            neighbor_distance=1000.0,
        )
        env = MEC(args)
        env.seed(2)
        obs, _, available, _, _ = env.reset()
        actor = R_Actor(args, env.observation_space, env.action_space)
        obs_a = torch.as_tensor(obs, dtype=torch.float32)
        obs_b = obs_a.clone()
        start = 2
        messages = obs_b[:, start:start + 40].reshape(5, 4, 10)
        obs_b[:, start:start + 40] = messages[:, [2, 0, 3, 1]].reshape(5, 40)
        available = torch.as_tensor(available, dtype=torch.float32)

        with torch.no_grad():
            flight_a = actor.flight_base(obs_a, available)
            flight_b = actor.flight_base(obs_b, available)

        torch.testing.assert_close(flight_a, flight_b, rtol=1e-5, atol=1e-6)

    def test_receiver_gated_r0_matches_mean_shared_initialization_and_actions(self):
        mean_args = self.make_args(
            actor_message_mode="task_summary",
            actor_message_pool="mean",
            neighbor_distance=0.0,
        )
        gated_args = self.make_args(
            actor_message_mode="task_summary",
            actor_message_pool="receiver_gated_sum",
            neighbor_distance=0.0,
        )
        env = MEC(mean_args)
        env.seed(31)
        obs, _, available, _, _ = env.reset()
        torch.manual_seed(31)
        mean_actor = R_Actor(mean_args, env.observation_space, env.action_space)
        mean_rng_after = torch.get_rng_state()
        torch.manual_seed(31)
        gated_actor = R_Actor(gated_args, env.observation_space, env.action_space)
        gated_rng_after = torch.get_rng_state()

        torch.testing.assert_close(mean_rng_after, gated_rng_after, rtol=0, atol=0)

        mean_state = mean_actor.state_dict()
        gated_state = gated_actor.state_dict()
        extra_keys = {
            key for key in gated_state if key.startswith("flight_base.message_gate.")
        }
        self.assertTrue(extra_keys)
        self.assertEqual(set(mean_state), set(gated_state) - extra_keys)
        for key, value in mean_state.items():
            torch.testing.assert_close(value, gated_state[key], rtol=0, atol=0)

        obs = torch.as_tensor(obs, dtype=torch.float32)
        available = torch.as_tensor(available, dtype=torch.float32)
        rnn = torch.zeros(mean_args.n_UAVs, mean_args.recurrent_N, mean_args.hidden_size)
        masks = torch.ones(mean_args.n_UAVs, 1)
        with torch.no_grad():
            mean_actions, _, _ = mean_actor(
                obs, rnn, masks, available.clone(), deterministic=True
            )
            gated_actions, _, _ = gated_actor(
                obs, rnn, masks, available.clone(), deterministic=True
            )
        torch.testing.assert_close(mean_actions, gated_actions, rtol=0, atol=0)

    def test_receiver_gated_fixed_sum_does_not_dilute_existing_zero_encoded_neighbors(self):
        class FirstFeatureEncoder(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, values):
                return values[..., :1].expand(*values.shape[:-1], self.hidden_size)

        class OpenGate(torch.nn.Module):
            def forward(self, values):
                return torch.full(
                    (*values.shape[:-1], 1), 50.0,
                    dtype=values.dtype,
                    device=values.device,
                )

        mean_args = self.make_args(
            actor_message_mode="task_summary",
            actor_message_pool="mean",
        )
        gated_args = self.make_args(
            actor_message_mode="task_summary",
            actor_message_pool="receiver_gated_sum",
        )
        env = MEC(mean_args)
        mean_actor = R_Actor(mean_args, env.observation_space, env.action_space)
        gated_actor = R_Actor(gated_args, env.observation_space, env.action_space)
        mean_actor.flight_base.message_encoder = FirstFeatureEncoder(
            mean_args.hidden_size
        )
        gated_actor.flight_base.message_encoder = FirstFeatureEncoder(
            gated_args.hidden_size
        )
        gated_actor.flight_base.message_gate = OpenGate()

        one_neighbor = torch.zeros(1, 4, 10)
        one_neighbor[0, 0, 0] = 4.0
        one_neighbor[0, 0, 9] = 1.0
        four_neighbors = one_neighbor.clone()
        four_neighbors[0, 1:, 9] = 1.0
        descriptor = torch.zeros(1, mean_args.hidden_size + 3)

        mean_one, _ = mean_actor.flight_base._pool_messages(
            one_neighbor, descriptor
        )
        mean_four, _ = mean_actor.flight_base._pool_messages(
            four_neighbors, descriptor
        )
        gated_one, _ = gated_actor.flight_base._pool_messages(
            one_neighbor, descriptor
        )
        gated_four, _ = gated_actor.flight_base._pool_messages(
            four_neighbors, descriptor
        )

        torch.testing.assert_close(mean_one, torch.full_like(mean_one, 4.0))
        torch.testing.assert_close(mean_four, torch.full_like(mean_four, 1.0))
        torch.testing.assert_close(gated_one, gated_four, rtol=1e-6, atol=1e-6)

    def test_absolute_raw_v2_normalizes_message_content_but_preserves_masks(self):
        args = self.make_args(
            actor_message_mode="task_summary",
            actor_message_contract="absolute_raw_v2",
            neighbor_distance=1000.0,
            hotspot_layout_mode="episode_template12_600_200",
            md_arrivals_min=5,
            md_arrivals_max=5,
            md_arrivals_per_region=[1, 4],
            episode_layout_context=True,
            episode_layout_context_units="meters_v2",
            ob_norm=True,
            ret_norm=False,
        )
        env = MEC(args)
        env.seed(2)
        obs, _, _, _, _ = env.reset()
        normer = Normer(
            args=args,
            obs_space=env.observation_space.shape,
            states_space=env.state_space.shape,
        )
        raw = obs.copy()
        normalized = normer._obfilt(obs.copy())
        start, end = normer.actor_message_slices[0]
        mask_indices = np.asarray([
            start + 10 * slot + 9 for slot in range(args.n_UAVs - 1)
        ])
        content_indices = np.asarray([
            index for index in range(start, end) if index not in mask_indices
        ])
        np.testing.assert_array_equal(
            normalized[:, mask_indices], raw[:, mask_indices]
        )
        self.assertFalse(np.allclose(
            normalized[:, content_indices], raw[:, content_indices]
        ))
        context_end = end + 8
        self.assertFalse(np.allclose(
            normalized[:, end:context_end], raw[:, end:context_end]
        ))
        self.assertFalse(np.allclose(
            normalized[:, context_end:], raw[:, context_end:]
        ))

        normers = [
            Normer(
                args=args,
                obs_space=env.observation_space.shape,
                states_space=env.state_space.shape,
            )
            for _ in range(args.n_UAVs)
        ]
        _, state, _, _, _ = env.reset()
        obs_batch = np.stack((raw, raw), axis=0)
        state_batch = np.stack((state, state), axis=0)
        normalized_batch, _, _ = normalize_batch(
            normers, obs_batch.copy(), state_batch.copy()
        )
        np.testing.assert_array_equal(
            normalized_batch[..., mask_indices], obs_batch[..., mask_indices]
        )
        self.assertFalse(np.allclose(
            normalized_batch[..., content_indices],
            obs_batch[..., content_indices],
        ))

    def test_relative_scaled_v1_still_preserves_the_full_message_block(self):
        args = self.make_args(
            actor_message_mode="task_summary",
            actor_message_contract="relative_scaled_v1",
            neighbor_distance=1000.0,
            ob_norm=True,
            ret_norm=False,
        )
        env = MEC(args)
        env.seed(2)
        obs, _, _, _, _ = env.reset()
        normer = Normer(
            args=args,
            obs_space=env.observation_space.shape,
            states_space=env.state_space.shape,
        )
        normalized = normer._obfilt(obs.copy())
        self.assertEqual(normer.obs_preserve_slices, normer.actor_message_slices)
        start, end = normer.actor_message_slices[0]
        np.testing.assert_array_equal(normalized[:, start:end], obs[:, start:end])

    def test_actor_messages_do_not_change_critic_state_contract(self):
        disabled_args = self.make_args(
            actor_message_mode="disabled",
            neighbor_distance=1000.0,
        )
        message_args = self.make_args(
            actor_message_mode="task_summary",
            neighbor_distance=1000.0,
        )
        disabled_env = MEC(disabled_args)
        message_env = MEC(message_args)
        disabled_env.seed(23)
        disabled_obs, disabled_state, _, _, _ = disabled_env.reset()
        message_env.seed(23)
        message_obs, message_state, _, _, _ = message_env.reset()

        self.assertEqual(message_obs.shape[-1] - disabled_obs.shape[-1], 40)
        np.testing.assert_allclose(message_state, disabled_state)
        np.testing.assert_allclose(
            message_env.get_critic_local_obs(message_obs),
            disabled_obs,
        )


if __name__ == "__main__":
    unittest.main()
