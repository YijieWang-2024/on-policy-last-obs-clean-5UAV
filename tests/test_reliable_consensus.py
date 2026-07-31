from types import SimpleNamespace

import numpy as np

from onpolicy.runner.separated.mec_runner import MECRunner


def test_finite_consensus_approaches_component_mean():
    runner = SimpleNamespace(num_agents=3, n_rollout_threads=1, episode_length=2)
    weights = np.array(
        [
            [2 / 3, 1 / 3, 0],
            [1 / 3, 1 / 3, 1 / 3],
            [0, 1 / 3, 2 / 3],
        ],
        dtype=np.float32,
    )
    runner.buffer = [
        SimpleNamespace(Metropolis_weights=np.array([[weights[i]]], dtype=np.float32))
        for i in range(3)
    ]
    local_advantages = np.array(
        [
            [[[0.0]], [[2.0]]],
            [[[6.0]], [[4.0]]],
            [[[0.0]], [[8.0]]],
        ],
        dtype=np.float32,
    )

    after_one = MECRunner.run_consensus_algorithm(runner, local_advantages, 1)
    after_fifty = MECRunner.run_consensus_algorithm(runner, local_advantages, 50)
    exact_mean = np.mean(local_advantages, axis=0, keepdims=True)

    assert np.mean(np.abs(after_fifty - exact_mean)) < np.mean(
        np.abs(after_one - exact_mean)
    )
    np.testing.assert_allclose(
        after_fifty, np.broadcast_to(exact_mean, after_fifty.shape), atol=1e-5
    )
