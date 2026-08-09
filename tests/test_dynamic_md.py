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

    def test_curriculum_schedule_and_random_id_assignment(self):
        args = self.make_args(
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=10,
            n_rollout_threads=1,
            num_env_steps=1000,
            uav_reset_curriculum=True,
            uav_reset_curriculum_training=True,
        )
        env = MEC(args)

        for reset_count, expected_probability in (
            (0, 0.5),
            (20, 0.5),
            (35, 0.25),
            (50, 0.0),
        ):
            env.curriculum_reset_count = reset_count
            self.assertAlmostEqual(
                env._curriculum_random_reset_probability(),
                expected_probability,
            )

        class FixedRNG:
            def __init__(self):
                self.values = iter([
                    10, 10, 150, 10, 290, 10, 430, 10, 570, 10,
                ])

            def uniform(self, _low, _high):
                return next(self.values)

            @staticmethod
            def permutation(size):
                return np.arange(size - 1, -1, -1)

        env.curriculum_reset_rng = FixedRNG()
        positions = env._sample_curriculum_uav_positions()
        np.testing.assert_array_equal(
            positions[:, :2],
            [[570, 10], [430, 10], [290, 10], [150, 10], [10, 10]],
        )

    def test_episode_template4_layouts_are_seeded_and_area_preserving(self):
        settings = dict(
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            hotspot_layout_mode="episode_template4",
        )
        env_a = MEC(self.make_args(**settings))
        env_b = MEC(self.make_args(**settings))
        env_a.seed(17)
        env_b.seed(17)

        sequence_a = []
        sequence_b = []
        seen = set()
        for _ in range(40):
            env_a.reset()
            env_b.reset()
            sequence_a.append(env_a.hotspot_layout_index)
            sequence_b.append(env_b.hotspot_layout_index)
            seen.add(env_a.hotspot_layout_index)
            np.testing.assert_array_equal(
                env_a.episode_hotspot_bounds,
                env_b.episode_hotspot_bounds,
            )
            widths = env_a.episode_hotspot_bounds[:, 1] - env_a.episode_hotspot_bounds[:, 0]
            heights = env_a.episode_hotspot_bounds[:, 3] - env_a.episode_hotspot_bounds[:, 2]
            np.testing.assert_array_equal(widths * heights, [175 ** 2, 400 ** 2])
            self.assertTrue(np.all(env_a.episode_hotspot_bounds >= 0))
            self.assertTrue(np.all(env_a.episode_hotspot_bounds <= 600))

        self.assertEqual(sequence_a, sequence_b)
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_p07_curriculum_schedule_uses_absolute_budget_steps(self):
        args = self.make_args(
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=1,
            n_rollout_threads=1,
            num_env_steps=50_000_000,
            uav_reset_curriculum=True,
            uav_reset_curriculum_training=True,
            uav_reset_curriculum_schedule="p0p7_10m_25m",
        )
        env = MEC(args)
        for reset_count, expected_probability in (
            (0, 0.70),
            (10_000_000, 0.70),
            (17_500_000, 0.35),
            (25_000_000, 0.0),
            (50_000_000, 0.0),
        ):
            env.curriculum_reset_count = reset_count
            self.assertAlmostEqual(
                env._curriculum_random_reset_probability(),
                expected_probability,
            )

    def test_bounded_md_velocity_profile_applies_on_birth_and_update(self):
        args = self.make_args(
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            mean_velocity=3.0,
            md_velocity_init_std=0.6,
            md_velocity_init_min_factor=0.4,
            md_velocity_init_max_factor=1.6,
            md_velocity_update_clip_min=0.0,
            md_velocity_update_clip_max=5.0,
        )
        env = MEC(args)
        env.seed(8)
        env.reset()
        active = env.active_md_mask.copy()
        self.assertTrue(np.all(env.gu_velocities[active] >= 1.2))
        self.assertTrue(np.all(env.gu_velocities[active] <= 4.8))

        env.gu_velocities[active] = 100.0
        action_dim = sum(int(np.prod(space.shape)) for space in env.action_space.spaces)
        action = np.zeros((args.n_UAVs, action_dim), dtype=np.float32)
        env.step(action)
        active_after = env.active_md_mask
        self.assertTrue(np.all(env.gu_velocities[active_after] >= 0.0))
        self.assertTrue(np.all(env.gu_velocities[active_after] <= 5.0))

    def test_random_layout_region_ids_do_not_depend_on_lower_left_coordinates(self):
        args = self.make_args(
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            hotspot_layout_mode="episode_template4",
        )
        env = MEC(args)
        env.seed(2)
        env.reset()
        env._set_episode_hotspot_layout(3)
        env.active_md_mask[:] = False
        env.md_region_ids[:] = -1
        env.dynamic_md_candidates = 0
        env.dynamic_md_admitted = 0
        env.dynamic_md_candidates_by_region[:] = 0
        env.dynamic_md_admitted_by_region[:] = 0
        env.uav_positions[:, :2] = np.array([
            [512.5, 512.5], [100, 100], [220, 100], [100, 220], [220, 220]
        ])
        env._admit_dynamic_mds()

        np.testing.assert_array_equal(env.dynamic_md_candidates_by_region, [1, 5])
        active = env.active_md_mask
        for region_id in (0, 1):
            slots = active & (env.md_region_ids == region_id)
            bounds = env.episode_hotspot_bounds[region_id]
            positions = env.gu_positions[slots, :2]
            self.assertTrue(np.all(positions[:, 0] >= bounds[0]))
            self.assertTrue(np.all(positions[:, 0] <= bounds[1]))
            self.assertTrue(np.all(positions[:, 1] >= bounds[2]))
            self.assertTrue(np.all(positions[:, 1] <= bounds[3]))

    def test_random_layout_is_constant_within_episode_and_hidden_from_shapes(self):
        args = self.make_args(
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=12,
            hotspot_layout_mode="episode_template4",
        )
        env = MEC(args)
        env.seed(23)
        obs, state, _, _, _ = env.reset()
        layout_index = env.hotspot_layout_index
        bounds = env.episode_hotspot_bounds.copy()
        action_dim = sum(int(np.prod(space.shape)) for space in env.action_space.spaces)
        action = np.zeros((args.n_UAVs, action_dim), dtype=np.float32)

        self.assertEqual(obs.shape, (args.n_UAVs, env.obs_dim))
        self.assertEqual(state.shape, (args.n_UAVs, env.state_dim))
        for _ in range(args.episode_length):
            env.step(action)
            self.assertEqual(env.hotspot_layout_index, layout_index)
            np.testing.assert_array_equal(env.episode_hotspot_bounds, bounds)
            for slot in np.flatnonzero(env.active_md_mask):
                region_bounds = bounds[env.md_region_ids[slot]]
                x, y = env.gu_positions[slot, :2]
                self.assertLessEqual(region_bounds[0], x)
                self.assertLessEqual(x, region_bounds[1])
                self.assertLessEqual(region_bounds[2], y)
                self.assertLessEqual(y, region_bounds[3])

    def test_700m_random_layout_is_episode_static_and_rotation_neutral(self):
        args = self.make_args(
            x_max_uav=700,
            y_max_uav=700,
            x_max_gu=700,
            y_max_gu=700,
            x_max=700,
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=12,
            hotspot_layout_mode="episode_template4",
        )
        env = MEC(args)
        env.seed(37)
        _, _, _, _, _ = env.reset()
        expected_positions = np.array([
            [350, 350], [150, 350], [550, 350], [350, 150], [350, 550]
        ])
        np.testing.assert_array_equal(env.uav_positions[:, :2], expected_positions)
        distances = np.linalg.norm(
            env.uav_positions[:, None, :2] - env.uav_positions[None, :, :2],
            axis=2,
        )
        self.assertTrue(np.all(distances <= 1000))

        layout_index = env.hotspot_layout_index
        bounds = env.episode_hotspot_bounds.copy()
        widths = bounds[:, 1] - bounds[:, 0]
        heights = bounds[:, 3] - bounds[:, 2]
        np.testing.assert_array_equal(widths * heights, [175 ** 2, 400 ** 2])
        self.assertTrue(np.all(bounds >= 0))
        self.assertTrue(np.all(bounds <= 700))

        action_dim = sum(int(np.prod(space.shape)) for space in env.action_space.spaces)
        action = np.zeros((args.n_UAVs, action_dim), dtype=np.float32)
        for _ in range(args.episode_length):
            env.step(action)
            self.assertEqual(env.hotspot_layout_index, layout_index)
            np.testing.assert_array_equal(env.episode_hotspot_bounds, bounds)

    def test_700m_template12_samples_all_ordered_corner_pairs(self):
        settings = dict(
            x_max_uav=700,
            y_max_uav=700,
            x_max_gu=700,
            y_max_gu=700,
            x_max=700,
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            hotspot_layout_mode="episode_template12",
        )
        env_a = MEC(self.make_args(**settings))
        env_b = MEC(self.make_args(**settings))
        env_a.seed(17)
        env_b.seed(17)
        seen = set()
        for _ in range(120):
            env_a.reset()
            env_b.reset()
            self.assertEqual(env_a.hotspot_layout_index, env_b.hotspot_layout_index)
            np.testing.assert_array_equal(
                env_a.episode_hotspot_bounds,
                env_b.episode_hotspot_bounds,
            )
            seen.add(env_a.hotspot_layout_index)
            widths = env_a.episode_hotspot_bounds[:, 1] - env_a.episode_hotspot_bounds[:, 0]
            heights = env_a.episode_hotspot_bounds[:, 3] - env_a.episode_hotspot_bounds[:, 2]
            np.testing.assert_array_equal(widths * heights, [175 ** 2, 400 ** 2])
            self.assertTrue(np.all(env_a.episode_hotspot_bounds >= 0))
            self.assertTrue(np.all(env_a.episode_hotspot_bounds <= 700))
        self.assertEqual(seen, set(range(12)))

    def test_600m_template12_uses_all_ordered_200m_400m_corner_pairs(self):
        settings = dict(
            md_arrivals_min=5,
            md_arrivals_max=5,
            md_arrivals_per_region=[1, 4],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            hotspot_layout_mode="episode_template12_600_200",
        )
        env_a = MEC(self.make_args(**settings))
        env_b = MEC(self.make_args(**settings))
        env_a.seed(31)
        env_b.seed(31)
        seen = set()
        corner_ids = {
            (0.0, 0.0): 0,
            (1.0, 0.0): 1,
            (0.0, 1.0): 2,
            (1.0, 1.0): 3,
        }
        for _ in range(120):
            env_a.reset()
            env_b.reset()
            self.assertEqual(env_a.hotspot_layout_index, env_b.hotspot_layout_index)
            np.testing.assert_array_equal(
                env_a.episode_hotspot_bounds,
                env_b.episode_hotspot_bounds,
            )
            seen.add(env_a.hotspot_layout_index)
            bounds = env_a.episode_hotspot_bounds
            widths = bounds[:, 1] - bounds[:, 0]
            heights = bounds[:, 3] - bounds[:, 2]
            np.testing.assert_array_equal(widths * heights, [200 ** 2, 400 ** 2])
            self.assertTrue(np.all(bounds >= 0))
            self.assertTrue(np.all(bounds <= 600))
            small_corner = corner_ids[(bounds[0, 0] / 400, bounds[0, 2] / 400)]
            large_corner = corner_ids[(bounds[1, 0] / 200, bounds[1, 2] / 200)]
            self.assertNotEqual(small_corner, large_corner)
        self.assertEqual(seen, set(range(12)))

    def test_700m_template12_start_layouts_have_expected_topology(self):
        common = dict(
            x_max_uav=700,
            y_max_uav=700,
            x_max_gu=700,
            y_max_gu=700,
            x_max=700,
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            hotspot_layout_mode="episode_template12",
        )
        for start_layout, expected in (
            (
                "line",
                np.array([
                    [100, 70], [200, 70], [300, 70], [400, 70], [450, 550]
                ], dtype=np.float32),
            ),
            (
                "staggered",
                np.array([
                    [100, 70], [200, 70], [300, 140], [400, 140], [450, 550]
                ], dtype=np.float32),
            ),
        ):
            env = MEC(self.make_args(
                **common, five_uav_start_layout=start_layout
            ))
            env.seed(2)
            env.reset()
            np.testing.assert_array_equal(env.uav_positions[:, :2], expected)
            distances = np.linalg.norm(
                expected[:, None] - expected[None, :], axis=2
            )
            upper = expected[-1]
            self.assertEqual(int(np.sum(distances[-1, :-1] <= 520)), 2)
            self.assertEqual(int(np.sum(distances[-1, :-1] <= 600)), 4)
            self.assertTrue(np.all(distances[np.triu_indices(5, 1)] >= 0))

    def test_episode_layout_context_is_shared_critic_visible_and_reward_invariant(self):
        settings = dict(
            x_max_uav=700,
            y_max_uav=700,
            x_max_gu=700,
            y_max_gu=700,
            x_max=700,
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            hotspot_layout_mode="episode_template12",
            actor_message_mode="geometry",
            spatial_flight_actor=True,
            neighbor_distance=260,
        )
        context_args = self.make_args(
            **settings,
            episode_layout_context=True,
            episode_layout_context_units="meters_v2",
        )
        baseline_args = self.make_args(
            **settings, episode_layout_context=False
        )
        context_env = MEC(context_args)
        baseline_env = MEC(baseline_args)
        context_env.seed(29)
        context_obs, context_state, _, _, _ = context_env.reset()
        # Reset the global NumPy stream before the baseline reset so the two
        # environments receive the same exogenous MD/task realization.
        baseline_env.seed(29)
        baseline_obs, baseline_state, _, _, _ = baseline_env.reset()

        self.assertEqual(context_env.layout_context_dim, 8)
        self.assertEqual(context_obs.shape[1], baseline_obs.shape[1] + 8)
        # The k-hop critic state contains one local token per UAV, so the
        # common 8-value context is repeated once per token.
        self.assertEqual(
            context_state.shape[1],
            baseline_state.shape[1] + 8 * context_env.n_UAVs,
        )
        np.testing.assert_allclose(
            context_env.episode_layout_context.reshape(2, 4),
            context_env.episode_hotspot_bounds,
            rtol=0.0,
            atol=1e-7,
        )

        prefix_dim = 2  # own x/y; timestep and agent-id are disabled here
        actor_only_end = prefix_dim + context_env.actor_only_obs_dim
        np.testing.assert_allclose(
            context_obs[:, actor_only_end:actor_only_end + 8],
            np.tile(context_env.episode_layout_context, (context_env.n_UAVs, 1)),
            rtol=0.0,
            atol=1e-7,
        )
        critic_obs = context_env.get_critic_local_obs(context_obs)
        np.testing.assert_allclose(
            critic_obs[:, prefix_dim:prefix_dim + 8],
            np.tile(context_env.episode_layout_context, (context_env.n_UAVs, 1)),
            rtol=0.0,
            atol=1e-7,
        )

        action_dim = sum(
            int(np.prod(space.shape)) for space in context_env.action_space.spaces
        )
        action = np.zeros((context_args.n_UAVs, action_dim), dtype=np.float32)
        np.random.seed(101)
        _, context_rewards, _, _, _, _, _, _ = context_env.step(action)
        np.random.seed(101)
        _, baseline_rewards, _, _, _, _, _, _ = baseline_env.step(action)
        np.testing.assert_allclose(
            context_rewards, baseline_rewards, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            context_env.system_performance_true_all_GUs,
            baseline_env.system_performance_true_all_GUs,
            rtol=0.0,
            atol=1e-12,
        )

    def test_normalized_v1_layout_context_remains_checkpoint_compatible(self):
        env = MEC(self.make_args(
            x_max_uav=700,
            y_max_uav=700,
            x_max_gu=700,
            y_max_gu=700,
            x_max=700,
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            hotspot_layout_mode="episode_template12",
            episode_layout_context=True,
            episode_layout_context_units="normalized_v1",
        ))
        env.seed(37)
        env.reset()
        np.testing.assert_allclose(
            env.episode_layout_context.reshape(2, 4),
            env.episode_hotspot_bounds / 700.0,
            rtol=0.0,
            atol=1e-7,
        )

    def test_random_layout_candidate_birth_stream_is_admission_independent(self):
        settings = dict(
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=2,
            hotspot_layout_mode="episode_template4",
        )
        covered_env = MEC(self.make_args(**settings))
        uncovered_env = MEC(self.make_args(**settings))
        for env in (covered_env, uncovered_env):
            env.seed(31)
            env.reset()
            env._set_episode_hotspot_layout(0)
            env.active_md_mask[:] = False

        for _ in range(4):
            covered_env.uav_positions[:, :2] = [100, 100]
            uncovered_env.uav_positions[:, :2] = [599, 1]
            covered_env._admit_dynamic_mds()
            uncovered_env._admit_dynamic_mds()
            np.testing.assert_array_equal(
                covered_env.last_candidate_positions,
                uncovered_env.last_candidate_positions,
            )
            np.testing.assert_array_equal(
                covered_env.last_candidate_region_ids,
                uncovered_env.last_candidate_region_ids,
            )

    def test_700m_moving_hotspots_are_continuous_hidden_birth_regions(self):
        args = self.make_args(
            x_max_uav=700,
            y_max_uav=700,
            x_max_gu=700,
            y_max_gu=700,
            x_max=700,
            md_arrivals_min=6,
            md_arrivals_max=6,
            md_arrivals_per_region=[1, 5],
            md_lifetime_min=10,
            md_lifetime_max=10,
            episode_length=400,
            hotspot_layout_mode="episode_moving_template4",
        )
        env = MEC(args)
        env.seed(41)
        obs, state, _, _, _ = env.reset()
        initial_bounds = env.episode_hotspot_bounds.copy()
        initial_obs_shape = obs.shape
        initial_state_shape = state.shape

        # A newly admitted user's reflection bounds are a birth-time snapshot,
        # not a pointer to the subsequently moving hotspot.
        env.uav_positions[:, :2] = np.array([[350.0, 350.0]] * args.n_UAVs)
        env._admit_dynamic_mds()
        active_slots = np.flatnonzero(env.active_md_mask)
        frozen_bounds = np.column_stack((
            env.x_min_all_gus[active_slots], env.x_max_all_gus[active_slots],
            env.y_min_all_gus[active_slots], env.y_max_all_gus[active_slots],
        )).copy()

        env._update_moving_hotspot_layout(1)
        step_one_bounds = env.episode_hotspot_bounds.copy()
        initial_centers = np.column_stack((
            initial_bounds[:, :2].mean(axis=1), initial_bounds[:, 2:].mean(axis=1)
        ))
        step_one_centers = np.column_stack((
            step_one_bounds[:, :2].mean(axis=1), step_one_bounds[:, 2:].mean(axis=1)
        ))
        center_shifts = np.linalg.norm(step_one_centers - initial_centers, axis=1)
        self.assertTrue(np.all(center_shifts < 4.0))
        self.assertTrue(np.all(center_shifts > 0.0))
        self.assertTrue(np.all(step_one_bounds >= 0.0))
        self.assertTrue(np.all(step_one_bounds <= 700.0))

        env._update_moving_hotspot_layout(200)
        midpoint_bounds = env.episode_hotspot_bounds.copy()
        env._update_moving_hotspot_layout(400)
        final_bounds = env.episode_hotspot_bounds.copy()
        self.assertFalse(np.array_equal(initial_bounds, midpoint_bounds))
        self.assertFalse(np.array_equal(midpoint_bounds, final_bounds))
        np.testing.assert_array_equal(
            np.column_stack((
                env.x_min_all_gus[active_slots], env.x_max_all_gus[active_slots],
                env.y_min_all_gus[active_slots], env.y_max_all_gus[active_slots],
            )),
            frozen_bounds,
        )
        self.assertEqual(initial_obs_shape, (args.n_UAVs, env.obs_dim))
        self.assertEqual(initial_state_shape, (args.n_UAVs, env.state_dim))


if __name__ == "__main__":
    unittest.main()
