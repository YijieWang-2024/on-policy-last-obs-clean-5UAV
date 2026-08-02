from .running_mean_std import RunningMeanStd
import numpy as np
import pickle


def normalize_batch(normers, obs, states, rews=None, dones=None):
    """Normalize all agents, optionally sharing only the return scale."""
    if not normers:
        return obs, states, rews

    first = normers[0]
    if any(
        (normer.ob_norm, normer.ret_norm, normer.shared_ret_norm,
         normer.not_norm, normer.clipob, normer.cliprew, normer.gamma,
         normer.epsilon)
        != (first.ob_norm, first.ret_norm, first.shared_ret_norm,
            first.not_norm, first.clipob, first.cliprew, first.gamma,
            first.epsilon)
        for normer in normers[1:]
    ):
        raise ValueError("batched normalization requires identical Normer settings")

    discounted_returns = []
    for agent_id, normer in enumerate(normers):
        if normer.ob_rms:
            normer.ob_rms.update(obs[:, agent_id])
            normer.state_rms.update(states[:, agent_id])

        if rews is not None and normer.ret_rms:
            agent_rews = rews[:, agent_id]
            if not hasattr(normer, 'returns') or normer.returns is None:
                normer.returns = np.zeros_like(agent_rews)
            normer.returns = normer.returns * normer.gamma + agent_rews
            if dones is not None:
                agent_dones = dones[:, agent_id]
                if agent_dones.shape != normer.returns.shape:
                    agent_dones = agent_dones[..., np.newaxis]
                normer.returns = normer.returns * (1.0 - agent_dones)
            discounted_returns.append(normer.returns.reshape(-1, 1))
            if not first.shared_ret_norm:
                normer.ret_rms.update(discounted_returns[-1])

    if rews is not None and first.ret_rms and first.shared_ret_norm:
        # Every agent observes the same pooled return distribution.  The RMS
        # objects remain separate so existing per-agent checkpoint files stay
        # self-contained, but receive identical samples and therefore retain
        # identical statistics.
        pooled_returns = np.concatenate(discounted_returns, axis=0)
        for normer in normers:
            normer.ret_rms.update(pooled_returns)

    if first.ob_rms:
        start = first.not_norm
        obs_mean = np.stack([normer.ob_rms.mean for normer in normers])
        obs_var = np.stack([normer.ob_rms.var for normer in normers])
        state_mean = np.stack([normer.state_rms.mean for normer in normers])
        state_var = np.stack([normer.state_rms.var for normer in normers])
        obs[..., start:] = np.clip(
            (obs[..., start:] - obs_mean[None, :, start:])
            / np.sqrt(obs_var[None, :, start:] + first.epsilon),
            -first.clipob,
            first.clipob,
        )
        states[..., start:] = np.clip(
            (states[..., start:] - state_mean[None, :, start:])
            / np.sqrt(state_var[None, :, start:] + first.epsilon),
            -first.clipob,
            first.clipob,
        )

    if rews is not None and first.ret_rms:
        scales = np.asarray([
            np.sqrt(np.asarray(normer.ret_rms.var) + normer.epsilon).reshape(-1)[0]
            for normer in normers
        ])
        rews[...] = np.clip(
            rews / scales[None, :, None],
            -first.cliprew,
            first.cliprew,
        )

    return obs, states, rews


class Normer():
    def __init__(self, args=None, obs_space=None, states_space=None, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.ob_norm = args.ob_norm
        self.ret_norm = args.ret_norm
        self.shared_ret_norm = getattr(args, 'shared_ret_norm', False)
        self.ob_state_with_timestep = getattr(args, 'ob_state_with_timestep', False)
        self.ob_state_with_id = getattr(args, 'ob_state_with_id', False)
        self.n_UAVs = getattr(args, 'n_UAVs', 1)
        self.not_norm = 0
        if self.ob_state_with_timestep:
            self.not_norm += 1
        if self.ob_state_with_id:
            self.not_norm += self.n_UAVs

        if self.ob_norm or self.ret_norm:
            self.ob_rms = RunningMeanStd(shape=obs_space) if self.ob_norm else None
            self.state_rms = RunningMeanStd(shape=states_space) if self.ob_norm else None
            self.ret_rms = RunningMeanStd(shape=()) if self.ret_norm else None
            self.clipob = clipob
            self.cliprew = cliprew
            # self.ret = 0
            self.gamma = gamma
            self.epsilon = epsilon

    def _rewsfilt(self, rews, dones=None):
        # if self.ret_rms:
        #     rews_mean = np.mean(rews)
        #     self.ret = self.ret * self.gamma + rews_mean
        #     self.ret_rms.update(np.array([[self.ret]]).reshape(1, 1))
        #     rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        #
        # return rews

        if self.ret_rms:
            # Initialize return tracking if needed
            if not hasattr(self, 'returns') or self.returns is None:
                self.returns = np.zeros_like(rews)
            # Update returns
            self.returns = self.returns * self.gamma + rews
            # Reset returns at episode boundaries
            if dones is not None:
                if dones.shape != self.returns.shape:
                    dones = dones[..., np.newaxis]
                self.returns = self.returns * (1.0 - dones)
            # Update running stats with flattened returns
            self.ret_rms.update(self.returns.reshape(-1, 1))
            # Scale rewards by standard deviation only (no mean subtraction)
            scaled_rews = rews / np.sqrt(self.ret_rms.var + self.epsilon)
            return np.clip(scaled_rews, -self.cliprew, self.cliprew)
        return rews


    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs[..., self.not_norm:] = np.clip((obs[..., self.not_norm:] - self.ob_rms.mean[..., self.not_norm:]) / np.sqrt(self.ob_rms.var[..., self.not_norm:] + self.epsilon), -self.clipob, self.clipob)
        return obs

    def _statefilt(self, states):
        if self.state_rms:
            self.state_rms.update(states)
            states[..., self.not_norm:] = np.clip((states[..., self.not_norm:] - self.state_rms.mean[..., self.not_norm:]) / np.sqrt(self.state_rms.var[..., self.not_norm:] + self.epsilon), -self.clipob, self.clipob)
        return states

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            state_dict = self.__dict__.copy()
            if 'returns' in state_dict:
                del state_dict['returns']
            pickle.dump(state_dict, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            state_dict = pickle.load(f)
            if 'returns' in state_dict:
                del state_dict['returns']
            self.__dict__.update(state_dict)
