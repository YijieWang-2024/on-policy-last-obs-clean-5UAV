import unittest

import numpy as np
import torch

from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.config import get_config
from onpolicy.envs.mec.mec import MEC, _cartesian_flight_velocity
from onpolicy.scripts.train.train_mec import parse_args


class CartesianFlightTest(unittest.TestCase):
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
        }
        settings.update(overrides)
        for name, value in settings.items():
            setattr(args, name, value)
        return args

    def test_velocity_projection_is_symmetric(self):
        proposals = np.array([
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [0.0, 0.0], [3.0, 4.0],
        ])
        velocities = _cartesian_flight_velocity(proposals, 30.0)
        np.testing.assert_allclose(velocities[:5], 30.0 * proposals[:5])
        np.testing.assert_allclose(velocities[5], [18.0, 24.0])
        self.assertTrue(np.all(np.linalg.norm(velocities, axis=1) <= 30.0 + 1e-12))

    def test_environment_does_not_clip_negative_flight_proposals(self):
        env = MEC(self.make_args())
        env.reset()
        action_dim = 2 + 3 * env.n_GUs
        actions = np.full((env.n_UAVs, action_dim), 0.5, dtype=np.float64)
        actions[0, :2] = [-0.75, -0.25]
        processed = env.process_actions(actions)
        np.testing.assert_allclose(processed[0, :2], [-0.75, -0.25])
        self.assertTrue(np.all((processed[:, 2:] >= 0.0) & (processed[:, 2:] <= 1.0)))

    def test_initial_policy_is_unbiased_and_log_prob_is_reproducible(self):
        torch.manual_seed(7)
        args = self.make_args()
        env = MEC(args)
        layer = ACTLayer(env.action_space, 16, True, 0.01, args)
        observations = torch.zeros(100_000, 16)
        flight_dist = layer.action_outs[0](observations)
        proposals = flight_dist.sample()
        quadrant_counts = torch.stack([
            ((proposals[:, 0] >= 0) & (proposals[:, 1] >= 0)).sum(),
            ((proposals[:, 0] < 0) & (proposals[:, 1] >= 0)).sum(),
            ((proposals[:, 0] < 0) & (proposals[:, 1] < 0)).sum(),
            ((proposals[:, 0] >= 0) & (proposals[:, 1] < 0)).sum(),
        ]).float() / len(proposals)
        self.assertTrue(torch.all((quadrant_counts > 0.24) & (quadrant_counts < 0.26)))

        speeds = np.linalg.norm(
            _cartesian_flight_velocity(proposals.detach().numpy(), 30.0), axis=1
        )
        self.assertTrue(14.5 < speeds.mean() < 15.5)
        self.assertTrue(0.035 < (torch.linalg.norm(proposals, dim=1) > 1).float().mean() < 0.055)
        torch.testing.assert_close(flight_dist.mode(), torch.zeros_like(proposals))
        torch.testing.assert_close(
            flight_dist.log_probs(proposals),
            layer.action_outs[0](observations).log_probs(proposals),
        )

    def test_neighbor_observation_uses_relative_fixed_id_slots(self):
        env = MEC(self.make_args(actor_neighbor_obs=True))
        _, state, _, _, _ = env.reset()
        self.assertEqual(env.obs_dim, 215)
        self.assertEqual(env.state_dim, 1015)
        self.assertEqual(state.shape, (5, 1015))
        env.uav_positions[:, :2] = np.array([
            [0, 0], [130, 0], [300, 0], [0, 300], [400, 400],
        ])
        env._update_distance_matrices()
        obs = env.get_local_obs()
        np.testing.assert_allclose(obs[0, 2:5], [0.5, 0.0, 1.0])
        np.testing.assert_allclose(obs[0, 5:14], 0.0)

    def test_neighbor_observation_is_actor_only(self):
        baseline = MEC(self.make_args(distance_only_user_sort=True))
        neighbor = MEC(self.make_args(
            actor_neighbor_obs=True,
            distance_only_user_sort=True,
        ))
        baseline.seed(23)
        baseline_obs, baseline_state, _, _, _ = baseline.reset()
        neighbor.seed(23)
        neighbor_obs, neighbor_state, _, _, _ = neighbor.reset()

        self.assertEqual(baseline_obs.shape, (5, 203))
        self.assertEqual(neighbor_obs.shape, (5, 215))
        self.assertEqual(baseline_state.shape, neighbor_state.shape)
        np.testing.assert_allclose(baseline_state, neighbor_state)

    def test_distance_only_sort_is_available_to_standard_actor(self):
        env = MEC(self.make_args(distance_only_user_sort=True))
        env.seed(29)
        env.reset()
        before = env.nearby_gus_of_uavs.copy()
        env.gu_tasks[:, 1] = np.linspace(env.C_min, env.C_max, env.n_GUs)
        env.gu_tasks[:, 2] = np.where(
            np.arange(env.n_GUs) % 2, env.delay_min, env.delay_max
        )
        np.testing.assert_array_equal(before, env.get_nearby_users_sorted_all())

    def test_policy_to_environment_full_episode(self):
        torch.manual_seed(11)
        np.random.seed(11)
        args = self.make_args(episode_length=400)
        env = MEC(args)
        obs, _, available_actions, _, _ = env.reset()
        layer = ACTLayer(env.action_space, env.obs_dim, True, 0.01, args)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        available_tensor = torch.as_tensor(available_actions, dtype=torch.float32)
        actions, rollout_log_prob = layer(obs_tensor, available_tensor)
        evaluated_log_prob, _ = layer.evaluate_actions(
            obs_tensor, actions, available_tensor.clone()
        )
        torch.testing.assert_close(rollout_log_prob, evaluated_log_prob)

        for _ in range(args.episode_length):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            available_tensor = torch.as_tensor(available_actions, dtype=torch.float32)
            with torch.no_grad():
                actions, _ = layer(obs_tensor, available_tensor)
            obs, rewards, _, _, available_actions, info, _, _ = env.step(
                actions.numpy()
            )
            self.assertTrue(np.all(np.isfinite(rewards)))
        self.assertEqual(info["candidate_md_arrivals"], 2400)
        self.assertTrue(np.all(np.isfinite(env.uav_positions)))


if __name__ == "__main__":
    unittest.main()
