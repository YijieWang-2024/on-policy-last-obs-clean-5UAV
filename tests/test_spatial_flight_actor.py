import unittest

import numpy as np
import torch

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.config import get_config
from onpolicy.envs.mec.mec import MEC
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


if __name__ == "__main__":
    unittest.main()
