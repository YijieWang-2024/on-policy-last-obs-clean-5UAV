import json
import numpy as np
import torch
from types import SimpleNamespace

from onpolicy.scripts.eval.compare_fixed_evaluations import paired_stats
from onpolicy.scripts.eval.evaluate_dynamic_mappo_fixed import (
    SUMMARY_METRICS,
    aggregate_rows,
    config_sha256,
    final_position_stability_slot,
    matched_config,
)
from onpolicy.scripts.eval.render_dynamic_mappo_episode import (
    mask_actor_message_observations,
    mask_actor_messages_beyond_distance,
    snapshot_checkpoint,
)
from onpolicy.runner.separated.mec_runner import MECRunner
from onpolicy.envs.mec.vec_normalize import normalize_batch
from onpolicy.utils.checkpoint_manifest import (
    build_checkpoint_manifest,
    write_checkpoint_manifest_atomic,
)
from scipy.stats import t as student_t


class _EvalPolicy:
    def act(
        self, obs, rnn_states, masks, available_actions=None,
        deterministic=False, attention_active_mask=None,
    ):
        assert deterministic
        assert attention_active_mask is not None
        return (
            torch.zeros((len(obs), 1), dtype=torch.float32),
            torch.as_tensor(rnn_states),
        )


class _EvalTrainer:
    def __init__(self):
        self.policy = _EvalPolicy()
        self.prepared = 0

    def prep_rollout(self):
        self.prepared += 1


class _EightFieldEvalEnv:
    def __init__(self):
        self.actions = None

    @staticmethod
    def reset():
        return (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 4), dtype=np.float32),
            np.ones((1, 2, 1), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
            np.ones((1, 2, 2), dtype=np.float32),
        )

    def step(self, actions):
        self.actions = actions.copy()
        info = {
            "cumulative_reward": np.asarray([1.0, 2.0]),
            "uav_m_toal_energy_consumption": np.asarray([3.0, 4.0]),
            "n_GUs_per_uav_served": np.asarray([5.0, 6.0]),
            "user_in_m_average_delay": np.asarray([7.0, 9.0]),
        }
        return (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 4), dtype=np.float32),
            np.zeros((1, 2, 1), dtype=np.float32),
            np.ones((1, 2), dtype=bool),
            [info],
            np.ones((1, 2, 1), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
            np.ones((1, 2, 2), dtype=np.float32),
        )


def test_separated_eval_uses_per_agent_trainers_and_eight_field_env_contract():
    runner = SimpleNamespace(
        num_agents=2,
        eval_envs=_EightFieldEvalEnv(),
        normer=[],
        trainer=[_EvalTrainer(), _EvalTrainer()],
        n_eval_rollout_threads=1,
        recurrent_N=1,
        hidden_size=3,
        all_args=SimpleNamespace(eval_episodes=1),
    )

    MECRunner.eval(runner, total_num_steps=0)

    assert runner.eval_envs.actions.shape == (1, 2, 1)
    assert [trainer.prepared for trainer in runner.trainer] == [1, 1]


class _FrozenRms:
    def __init__(self, width):
        self.mean = np.zeros(width, dtype=np.float64)
        self.var = np.ones(width, dtype=np.float64)

    @staticmethod
    def update(_values):
        raise AssertionError("evaluation must not update normalization statistics")


def test_normalize_batch_can_freeze_eval_statistics():
    normers = []
    for _ in range(2):
        normers.append(SimpleNamespace(
            ob_norm=True,
            ret_norm=True,
            shared_ret_norm=True,
            not_norm=0,
            obs_preserve_slices=(),
            clipob=10.0,
            cliprew=10.0,
            gamma=0.99,
            epsilon=1e-8,
            ob_rms=_FrozenRms(3),
            state_rms=_FrozenRms(4),
            ret_rms=_FrozenRms(1),
        ))
    obs = np.ones((1, 2, 3), dtype=np.float32)
    states = np.ones((1, 2, 4), dtype=np.float32)
    rewards = np.ones((1, 2, 1), dtype=np.float32)

    normalized_obs, normalized_states, normalized_rewards = normalize_batch(
        normers,
        obs.copy(),
        states.copy(),
        rewards.copy(),
        np.ones((1, 2), dtype=bool),
        update=False,
    )

    np.testing.assert_allclose(normalized_obs, obs, atol=1e-6)
    np.testing.assert_allclose(normalized_states, states, atol=1e-6)
    np.testing.assert_allclose(normalized_rewards, rewards, atol=1e-6)


def test_settling_slot_finds_suffix_near_final_deployment():
    positions = np.array(
        [
            [[0.0, 0.0], [100.0, 0.0]],
            [[20.0, 0.0], [90.0, 0.0]],
            [[39.0, 0.0], [81.0, 0.0]],
            [[41.0, 0.0], [79.0, 0.0]],
            [[40.0, 0.0], [80.0, 0.0]],
        ]
    )

    assert final_position_stability_slot(positions, tolerance=2.0) == 2


def test_aggregate_rows_reports_sample_std_and_confidence_width():
    rows = []
    for value in (1.0, 3.0, 5.0):
        row = {metric: value for metric in SUMMARY_METRICS}
        rows.append(row)

    aggregate = aggregate_rows(rows)

    for metric in SUMMARY_METRICS:
        assert aggregate[metric]["count"] == 3
        assert aggregate[metric]["mean"] == 3.0
        np.testing.assert_allclose(aggregate[metric]["std"], 2.0)
        np.testing.assert_allclose(
            aggregate[metric]["ci95_half_width"],
            student_t.ppf(0.975, 2) * 2.0 / np.sqrt(3),
        )


def test_paired_stats_uses_within_seed_differences():
    baseline = np.array([100.0, 200.0, 300.0, 400.0])
    candidate = baseline + np.array([5.0, 7.0, 9.0, 11.0])

    stats = paired_stats(baseline, candidate)

    assert stats["count"] == 4
    assert stats["baseline_mean"] == 250.0
    assert stats["candidate_mean"] == 258.0
    assert stats["mean_difference"] == 8.0
    assert stats["win_fraction"] == 1.0
    assert stats["ci95_low"] > 0.0


def test_paired_stats_respects_lower_is_better_direction():
    baseline = np.array([10.0, 12.0, 14.0, 16.0])
    candidate = baseline - 2.0

    stats = paired_stats(baseline, candidate, direction=-1)

    assert stats["mean_difference"] == -2.0
    assert stats["direction_adjusted_mean_improvement"] == 2.0
    assert stats["win_fraction"] == 1.0


def test_matched_config_ignores_only_predeclared_radius_and_run_fields():
    base = {
        "neighbor_distance": 0.0,
        "externality_beta": 0.025,
        "experiment_name": "r0",
        "seed": 2,
        "md_lifetime_max": 10,
        "mean_velocity": 0.5,
    }
    candidate = dict(
        base,
        neighbor_distance=1000.0,
        externality_beta=0.1,
        experiment_name="r1000",
    )

    assert config_sha256(matched_config(base)) == config_sha256(
        matched_config(candidate)
    )
    candidate["mean_velocity"] = 2.0
    assert config_sha256(matched_config(base)) != config_sha256(
        matched_config(candidate)
    )


def test_actor_message_masking_zeros_only_declared_preserved_slice():
    obs = np.arange(2 * 12, dtype=np.float32).reshape(2, 12)
    normers = [
        SimpleNamespace(
            actor_message_slices=[(3, 7)], obs_preserve_slices=[(6, 7)]
        ),
        SimpleNamespace(
            actor_message_slices=[(3, 7)], obs_preserve_slices=[(6, 7)]
        ),
    ]

    masked = mask_actor_message_observations(normers, obs)

    np.testing.assert_array_equal(masked[:, :3], obs[:, :3])
    np.testing.assert_array_equal(masked[:, 3:7], 0.0)
    np.testing.assert_array_equal(masked[:, 7:], obs[:, 7:])
    np.testing.assert_array_equal(obs, np.arange(24, dtype=np.float32).reshape(2, 12))


def test_actor_message_masking_rejects_non_message_checkpoint():
    normers = [SimpleNamespace(obs_preserve_slices=[]) for _ in range(2)]
    with np.testing.assert_raises_regex(
        ValueError, "message-capable checkpoint"
    ):
        mask_actor_message_observations(normers, np.ones((2, 8)))


def test_actor_message_distance_mask_retains_near_and_zeros_far_packets():
    obs = np.arange(2 * 46, dtype=np.float32).reshape(2, 46)
    normers = [
        SimpleNamespace(obs_preserve_slices=[(3, 43)]),
        SimpleNamespace(obs_preserve_slices=[(3, 43)]),
    ]
    packets = np.zeros((2, 4, 10), dtype=np.float32)
    packets[:, 0, :2] = [0.2, 0.0]       # 120 m
    packets[:, 1, :2] = [0.5, 0.0]       # 300 m
    packets[:, 2, :2] = [0.3, 0.4]       # 300 m
    packets[:, 3, :2] = [260.0 / 600.0, 0.0]  # boundary retained
    packets[:, :, 2:9] = 7.0
    packets[:, :, 9] = 1.0
    obs[:, 3:43] = packets.reshape(2, 40)

    masked = mask_actor_messages_beyond_distance(
        normers,
        obs,
        distance_limit=260.0,
        map_scale=600.0,
    )
    masked_packets = masked[:, 3:43].reshape(2, 4, 10)

    np.testing.assert_array_equal(masked_packets[:, 0], packets[:, 0])
    np.testing.assert_array_equal(masked_packets[:, 1:3], 0.0)
    np.testing.assert_allclose(masked_packets[:, 3], packets[:, 3])
    np.testing.assert_array_equal(masked[:, :3], obs[:, :3])
    np.testing.assert_array_equal(masked[:, 43:], obs[:, 43:])
    np.testing.assert_array_equal(obs[:, 3:43], packets.reshape(2, 40))


def test_absolute_message_distance_mask_uses_receiver_and_sender_positions():
    obs = np.zeros((2, 46), dtype=np.float32)
    normers = [
        SimpleNamespace(
            actor_message_slices=[(3, 43)],
            obs_preserve_slices=[(12, 13), (22, 23), (32, 33), (42, 43)],
            actor_message_contract="absolute_raw_v2",
            not_norm=1,
        )
        for _ in range(2)
    ]
    receiver_positions = np.asarray([[100.0, 100.0], [200.0, 200.0]])
    obs[:, 1:3] = receiver_positions
    packets = np.zeros((2, 4, 10), dtype=np.float32)
    offsets = np.asarray([
        [120.0, 0.0], [300.0, 0.0], [180.0, 240.0], [260.0, 0.0]
    ])
    packets[:, :, :2] = receiver_positions[:, None, :] + offsets[None, :, :]
    packets[:, :, 2:9] = 7.0
    packets[:, :, 9] = 1.0
    obs[:, 3:43] = packets.reshape(2, 40)

    masked = mask_actor_messages_beyond_distance(
        normers, obs, distance_limit=260.0
    )
    masked_packets = masked[:, 3:43].reshape(2, 4, 10)

    np.testing.assert_array_equal(masked_packets[:, 0], packets[:, 0])
    np.testing.assert_array_equal(masked_packets[:, 1:3], 0.0)
    np.testing.assert_allclose(masked_packets[:, 3], packets[:, 3])
    np.testing.assert_array_equal(masked[:, :3], obs[:, :3])
    np.testing.assert_array_equal(masked[:, 43:], obs[:, 43:])


def test_actor_message_distance_mask_validates_contract():
    normers = [SimpleNamespace(obs_preserve_slices=[(2, 11)])]
    with np.testing.assert_raises_regex(ValueError, "10-value packets"):
        mask_actor_messages_beyond_distance(
            normers,
            np.ones((1, 12), dtype=np.float32),
            distance_limit=260.0,
            map_scale=600.0,
        )
    for invalid_limit in (0.0, -1.0, np.nan, np.inf):
        with np.testing.assert_raises_regex(ValueError, "distance limit"):
            mask_actor_messages_beyond_distance(
                [SimpleNamespace(obs_preserve_slices=[(2, 12)])],
                np.ones((1, 12), dtype=np.float32),
                distance_limit=invalid_limit,
                map_scale=600.0,
            )


def _write_checkpoint_files(models_dir, num_agents=5):
    for agent_id in range(num_agents):
        (models_dir / f"actor_agent{agent_id}.pt").write_bytes(b"actor")
        (models_dir / f"critic_agent{agent_id}.pt").write_bytes(b"critic")
        (models_dir / f"normer{agent_id}.pkl").write_bytes(b"normer")


def _write_manifest(models_dir, args_path, session_step, source_step=0, num_agents=5):
    manifest = build_checkpoint_manifest(
        models_dir,
        num_agents=num_agents,
        episode_index=10,
        session_total_num_steps=session_step,
        source_total_num_steps=source_step,
        save_interval_episodes=5,
        cumulative_step_provenance=(
            "verified_chain" if source_step == 0 else "legacy_declared_base"
        ),
        args_path=args_path,
    )
    write_checkpoint_manifest_atomic(models_dir, manifest)
    return manifest


def test_snapshot_checkpoint_verifies_and_copies_manifest(tmp_path):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text('{"n_UAVs": 5}', encoding="utf-8")
    _write_checkpoint_files(models_dir)
    _write_manifest(models_dir, args_path, 123456)

    checkpoint_dir, _ = snapshot_checkpoint(run_dir, tmp_path / "evaluation")

    copied = json.loads(
        (checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert copied["total_num_steps"] == 123456


def test_snapshot_checkpoint_reads_six_uav_count_from_args(tmp_path):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text('{"n_UAVs": 6}', encoding="utf-8")
    _write_checkpoint_files(models_dir, num_agents=6)
    _write_manifest(models_dir, args_path, 654321, num_agents=6)

    checkpoint_dir, _ = snapshot_checkpoint(run_dir, tmp_path / "evaluation")

    assert (checkpoint_dir / "actor_agent5.pt").is_file()
    assert (checkpoint_dir / "critic_agent5.pt").is_file()
    assert (checkpoint_dir / "normer5.pkl").is_file()


def test_snapshot_checkpoint_rejects_file_changed_after_manifest(tmp_path):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text('{"n_UAVs": 5}', encoding="utf-8")
    _write_checkpoint_files(models_dir)
    _write_manifest(models_dir, args_path, 123456)
    changed = models_dir / "actor_agent0.pt"
    changed.write_bytes(b"changed-after-manifest")

    with np.testing.assert_raises_regex(RuntimeError, "does not match manifest"):
        snapshot_checkpoint(run_dir, tmp_path / "evaluation")


def test_snapshot_checkpoint_rejects_partial_manifest(tmp_path):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text('{"n_UAVs": 5}', encoding="utf-8")
    _write_checkpoint_files(models_dir)
    manifest = _write_manifest(models_dir, args_path, 123456)
    del manifest["files"]["critic_agent4.pt"]
    with (models_dir / "checkpoint_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle)

    with np.testing.assert_raises_regex(ValueError, "file set mismatch"):
        snapshot_checkpoint(run_dir, tmp_path / "evaluation")


def test_snapshot_checkpoint_rejects_args_changed_after_manifest(tmp_path):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text('{"n_UAVs": 5}', encoding="utf-8")
    _write_checkpoint_files(models_dir)
    _write_manifest(models_dir, args_path, 123456)
    args_path.write_text('{"n_UAVs": 5, "changed": true}', encoding="utf-8")

    with np.testing.assert_raises_regex(RuntimeError, "args.json does not match"):
        snapshot_checkpoint(run_dir, tmp_path / "evaluation")


def test_manifest_tracks_cumulative_warm_start_steps(tmp_path):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text('{"n_UAVs": 5}', encoding="utf-8")
    _write_checkpoint_files(models_dir)

    manifest = _write_manifest(
        models_dir, args_path, session_step=50_000_000, source_step=50_000_000
    )

    assert manifest["session_total_num_steps"] == 50_000_000
    assert manifest["source_total_num_steps"] == 50_000_000
    assert manifest["total_num_steps"] == 100_000_000


def test_snapshot_checkpoint_keeps_optional_md_gru_state(tmp_path):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text('{"n_UAVs": 5}', encoding="utf-8")
    _write_checkpoint_files(models_dir)
    (models_dir / "md_gru_shared.pt").write_bytes(b"shared-gru")
    manifest = build_checkpoint_manifest(
        models_dir,
        num_agents=5,
        episode_index=10,
        session_total_num_steps=123456,
        source_total_num_steps=0,
        save_interval_episodes=5,
        cumulative_step_provenance="verified_chain",
        args_path=args_path,
        extra_checkpoint_names=("md_gru_shared.pt",),
    )
    write_checkpoint_manifest_atomic(models_dir, manifest)

    checkpoint_dir, _ = snapshot_checkpoint(run_dir, tmp_path / "evaluation")

    assert (checkpoint_dir / "md_gru_shared.pt").read_bytes() == b"shared-gru"


def test_manifest_rejects_unknown_optional_checkpoint_file(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_checkpoint_files(models_dir)
    (models_dir / "unexpected.pt").write_bytes(b"unexpected")

    with np.testing.assert_raises_regex(ValueError, "unsupported extra"):
        build_checkpoint_manifest(
            models_dir,
            num_agents=5,
            episode_index=0,
            session_total_num_steps=1,
            source_total_num_steps=0,
            save_interval_episodes=1,
            cumulative_step_provenance="verified_chain",
            extra_checkpoint_names=("unexpected.pt",),
        )


class _LoadTarget:
    def load_state_dict(self, state):
        assert state == {}


class _NormerTarget:
    def load(self, path):
        assert path.endswith(".pkl")


def _restore_only_runner(
    models_dir,
    checkpoint_base_steps=None,
    confirm_legacy_checkpoint_frozen=False,
):
    runner = MECRunner.__new__(MECRunner)
    runner.model_dir = str(models_dir)
    runner.num_agents = 5
    runner.all_args = SimpleNamespace(
        checkpoint_base_steps=checkpoint_base_steps,
        confirm_legacy_checkpoint_frozen=confirm_legacy_checkpoint_frozen,
    )
    runner.policy = [
        SimpleNamespace(actor=_LoadTarget(), critic=_LoadTarget())
        for _ in range(5)
    ]
    runner.normer = [_NormerTarget() for _ in range(5)]
    return runner


def test_restore_verifies_manifest_and_inherits_cumulative_steps(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text("{}", encoding="utf-8")
    _write_checkpoint_files(models_dir)
    _write_manifest(
        models_dir, args_path, session_step=50_000_000, source_step=50_000_000
    )
    monkeypatch.setattr("onpolicy.runner.separated.mec_runner.torch.load", lambda *args, **kwargs: {})
    runner = _restore_only_runner(models_dir)

    runner.restore()

    assert runner.checkpoint_source_steps == 100_000_000


def test_restore_requires_explicit_steps_for_legacy_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_checkpoint_files(models_dir)
    runner = _restore_only_runner(models_dir)

    with np.testing.assert_raises_regex(ValueError, "checkpoint_base_steps"):
        runner.restore()


def test_restore_requires_frozen_confirmation_for_legacy_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_checkpoint_files(models_dir)
    runner = _restore_only_runner(models_dir, checkpoint_base_steps=50_000_000)

    with np.testing.assert_raises_regex(
        ValueError, "confirm_legacy_checkpoint_frozen"
    ):
        runner.restore()


def test_restore_rejects_negative_legacy_base_steps(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_checkpoint_files(models_dir)
    runner = _restore_only_runner(
        models_dir,
        checkpoint_base_steps=-1,
        confirm_legacy_checkpoint_frozen=True,
    )

    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        runner.restore()


def test_restore_rejects_checkpoint_changed_while_loading(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    args_path = run_dir / "args.json"
    args_path.write_text("{}", encoding="utf-8")
    _write_checkpoint_files(models_dir)
    _write_manifest(models_dir, args_path, 123456)
    changed = False

    def load_and_change(*args, **kwargs):
        nonlocal changed
        if not changed:
            (models_dir / "critic_agent4.pt").write_bytes(b"changed-during-load")
            changed = True
        return {}

    monkeypatch.setattr(
        "onpolicy.runner.separated.mec_runner.torch.load", load_and_change
    )
    runner = _restore_only_runner(models_dir)

    with np.testing.assert_raises_regex(RuntimeError, "does not match manifest"):
        runner.restore()


def test_snapshot_legacy_requires_explicit_frozen_confirmation(tmp_path):
    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    (run_dir / "args.json").write_text('{"n_UAVs": 5}', encoding="utf-8")
    _write_checkpoint_files(models_dir)

    with np.testing.assert_raises_regex(ValueError, "legacy checkpoint"):
        snapshot_checkpoint(run_dir, tmp_path / "blocked")

    checkpoint_dir, _ = snapshot_checkpoint(
        run_dir,
        tmp_path / "allowed",
        allow_frozen_legacy=True,
    )
    assert (checkpoint_dir / "actor_agent0.pt").is_file()
