from abc import ABC
try:
    import gym
    from gym.spaces import Box, Discrete
except ImportError:
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete
import numpy as np
from .mec import MEC

class WrappedMECEnv(gym.Env, ABC):
    def __init__(self,
                 args=None,
                 ):
        assert args is not None
        self.args =args
        self.not_process_action = args.not_process_action

        self.env = MEC(args=args)
        env_info = self.env.get_info()

        self.nagent = env_info['n_agents']

        # self.action_space = [self.env.action_space]
        # self.observation_space = [self.env.observation_space.shape]
        # self.share_observation_space = [self.env.state_space.shape]
        self.available_actions_space = self.env.GUs_in_action_dim

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.nagent):
            self.action_space.append(self.env.action_space)
            self.observation_space.append(self.env.observation_space.shape)
            self.share_observation_space.append(self.env.state_space.shape)

    def seed(self, seed=None):
        self.env.seed(seed)

    def process_actions(self, actions):
        if self.not_process_action:
            return actions
        else:
            processed_actions = self.env.process_local_actions(actions)
            return processed_actions

    def reset(self):
        obs, state, avail_actions, Metropolis_weights, attention_active_mask = self.env.reset()
        obs, states = self.format_obs_state(obs, state)
        return obs, states, avail_actions, Metropolis_weights, attention_active_mask

    def step(self, actions):
        obs, reward, dones, state, avail_actions, env_info, Metropolis_weights, attention_active_mask = self.env.step(actions)
        obs, states = self.format_obs_state(obs, state)
        # return obs, reward, dones, states, avail_actions, env_info
        # 换了return的顺序
        return obs, states, reward[..., np.newaxis], dones, env_info, avail_actions, Metropolis_weights, attention_active_mask

    def format_obs_state(self, obs, state):
        return obs, state

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)
