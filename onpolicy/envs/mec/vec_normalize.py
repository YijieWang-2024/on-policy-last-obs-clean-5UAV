from .running_mean_std import RunningMeanStd
import numpy as np
import pickle

class Normer():
    def __init__(self, args=None, obs_space=None, states_space=None, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.ob_norm = args.ob_norm
        self.ret_norm = args.ret_norm
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
