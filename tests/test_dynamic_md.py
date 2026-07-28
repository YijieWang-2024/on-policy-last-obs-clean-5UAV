import unittest

import numpy as np

from onpolicy.config import get_config
from onpolicy.envs.mec.mec import MEC
from onpolicy.scripts.train.train_mec import parse_args


class DynamicMDTest(unittest.TestCase):
    @staticmethod
    def make_args(**overrides):
        args = parse_args([], get_config())
        settings = {
            "n_UAVs": 5,
            "n_GUs": 60,
            "max_GUs_in_range": 20,
            "dynamic_md": True,
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
        }
        settings.update(overrides)
        for name, value in settings.items():
            setattr(args, name, value)
        return args

    def test_population_invariants(self):
        args = self.make_args(
            md_arrivals_min=3,
            md_arrivals_max=3,
            md_lifetime_min=20,
            md_lifetime_max=20,
            episode_length=20,
        )

        env = MEC(args)
        env.seed(2)
        env.reset()
        action_dim = sum(int(np.prod(space.shape)) for space in env.action_space.spaces)
        self.assertEqual(env.obs_dim, 203)
        self.assertEqual(action_dim, 62)

        reflected_slot = np.flatnonzero(env.active_md_mask)[0]
        env.gu_positions[reflected_slot] = [174.9, 70.0, env.H_GU]
        env.gu_velocities[reflected_slot] = env.mean_velocity
        env.gu_directions[reflected_slot] = 0.0
        env.gu_directions_0[reflected_slot] = 0.0
        env.md_remaining_lifetime[reflected_slot] = args.md_lifetime_max
        env.x_min_all_gus[reflected_slot] = 0
        env.x_max_all_gus[reflected_slot] = 175
        env.y_min_all_gus[reflected_slot] = 0
        env.y_max_all_gus[reflected_slot] = 175

        for step in range(args.episode_length):
            action = np.random.uniform(0, 1, (args.n_UAVs, action_dim)).astype(np.float32)
            action[:, :2] = 0
            _, rewards, _, _, _, info, _, _ = env.step(action)
            active_mask = env.active_md_mask
            active_ids = env.md_session_ids[active_mask]
            self.assertLessEqual(len(active_ids), args.n_GUs)
            self.assertEqual(len(active_ids), len(np.unique(active_ids)))
            self.assertTrue(np.all(env.md_remaining_lifetime[active_mask] > 0))
            self.assertTrue(np.all(np.any(env.coverage_mask[:, active_mask], axis=0)))
            self.assertTrue(np.all(env.coverage_mask[:, ~active_mask] == 0))
            self.assertTrue(np.all(env.gu_tasks[active_mask] > 0))
            self.assertTrue(np.all(env.gu_tasks[~active_mask] == 0))
            self.assertTrue(np.all(np.isfinite(rewards)))

            positions = env.gu_positions[active_mask]
            self.assertTrue(np.all(positions[:, 0] >= env.x_min_all_gus[active_mask]))
            self.assertTrue(np.all(positions[:, 0] <= env.x_max_all_gus[active_mask]))
            self.assertTrue(np.all(positions[:, 1] >= env.y_min_all_gus[active_mask]))
            self.assertTrue(np.all(positions[:, 1] <= env.y_max_all_gus[active_mask]))
            lower = (
                (env.x_min_all_gus[active_mask] == 0)
                & (env.x_max_all_gus[active_mask] == 175)
                & (env.y_min_all_gus[active_mask] == 0)
                & (env.y_max_all_gus[active_mask] == 175)
            )
            upper = (
                (env.x_min_all_gus[active_mask] == 200)
                & (env.x_max_all_gus[active_mask] == 600)
                & (env.y_min_all_gus[active_mask] == 200)
                & (env.y_max_all_gus[active_mask] == 600)
            )
            self.assertTrue(np.all(lower | upper))

            if step == 0:
                self.assertTrue(active_mask[reflected_slot])
                self.assertLessEqual(env.gu_positions[reflected_slot, 0], 175)
                self.assertLess(np.cos(env.gu_directions[reflected_slot]), 0)

        expected_full_performance = (
            info['system_performance_true_all_GUs']
            + (args.n_GUs * args.episode_length - env.dynamic_md_active_sum)
            * env.expected_inactive_md_local_reward
        )
        np.testing.assert_allclose(
            info['system_performance_equivalent_full_GUs'],
            expected_full_performance,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_regional_arrivals_start_empty_and_only_admit_covered_users(self):
        args = self.make_args(
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
        )
        env = MEC(args)
        env.seed(2)
        env.reset()

        np.testing.assert_array_equal(env.uav_positions[0, :2], [200, 525])
        self.assertEqual(env.dynamic_md_candidates_by_region.tolist(), [1, 5])
        self.assertLessEqual(np.sum(env.active_md_mask), 6)
        self.assertTrue(np.all(np.any(env.coverage_mask[:, env.active_md_mask], axis=0)))
        self.assertTrue(np.all(env.md_remaining_lifetime[env.active_md_mask] == 10))

        departing_slot = np.flatnonzero(env.active_md_mask)[0]
        departing_session = env.md_session_ids[departing_slot]
        env.gu_positions[departing_slot, :2] = [0, 175]
        env.gu_velocities[departing_slot] = 0
        env.gu_directions[departing_slot] = 0
        env.gu_directions_0[departing_slot] = 0

        action_dim = sum(int(np.prod(space.shape)) for space in env.action_space.spaces)
        action = np.zeros((args.n_UAVs, action_dim), dtype=np.float32)
        env.step(action)
        self.assertEqual(env.dynamic_md_candidates_by_region.tolist(), [2, 10])
        self.assertEqual(env.dynamic_md_candidates, 12)
        self.assertNotIn(departing_session, env.md_session_ids[env.active_md_mask])
        self.assertTrue(np.all(np.any(env.coverage_mask[:, env.active_md_mask], axis=0)))


if __name__ == "__main__":
    unittest.main()
