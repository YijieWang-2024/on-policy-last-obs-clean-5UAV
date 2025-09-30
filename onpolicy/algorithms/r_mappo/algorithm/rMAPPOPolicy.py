import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_attention import R_Actor_Attention, R_Critic_Attention
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.use_atten_actor = args.use_atten_actor
        self.use_atten_critic = args.use_atten_critic
        # self.concat_neighbor_obs = args.concat_neighbor_obs
        self.state_is_k_hops = args.state_is_k_hops
        self.perform_with_local_state = args.perform_with_local_state
        # 0803，obs和state一样。CTCE。
        if self.use_atten_actor:
            self.actor = R_Actor_Attention(args, self.obs_space, self.act_space, self.device)
        else:
            self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        # self.actor = R_Actor(args, self.obs_space, self.act_space, self.device) # actor固定了s_{i,t}。没用atten必要。

        if self.use_atten_critic:
            # assert self.concat_neighbor_obs is True
            assert (self.state_is_k_hops is True) or ((self.perform_with_local_state or self.state_is_k_hops) is False)
            # 利用s_{M_i^k}的信息，才要atten的critic。 或者默认到了全局拼接state，也可以要atten的critic。
            self.critic = R_Critic_Attention(args, self.share_obs_space, self.device)
        else:
            self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False,attention_active_mask=None):
        # # 测试
        # deterministic = True
        # 在mec_runner里调用了这个，运行得到动作。只用了return的 _, actions, _, rnn_states_actor, rnn_states_critic
        if self.use_atten_actor:
            actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                     rnn_states_actor,
                                                                     masks,
                                                                     available_actions,
                                                                     deterministic = deterministic,
                                                                     attention_active_mask = attention_active_mask)
        else:
            actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                     rnn_states_actor,
                                                                     masks,
                                                                     available_actions,
                                                                     deterministic=deterministic)
        if self.use_atten_critic:
            values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, attention_active_mask=attention_active_mask)
        else:
            values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks, attention_active_mask=None):
        if self.use_atten_critic:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks, attention_active_mask=attention_active_mask)
        else:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, attention_active_mask=None):
        if self.use_atten_actor:
            action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                         rnn_states_actor,
                                                                         action,
                                                                         masks,
                                                                         available_actions,
                                                                         active_masks=active_masks,
                                                                         attention_active_mask=attention_active_mask)
        else:
            action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                         rnn_states_actor,
                                                                         action,
                                                                         masks,
                                                                         available_actions,
                                                                         active_masks=active_masks)
        if self.use_atten_critic:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks, attention_active_mask=attention_active_mask)
        else:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, attention_active_mask=None):
        if self.use_atten_actor:
            actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions,
                                                      deterministic=deterministic,
                                                      attention_active_mask=attention_active_mask)
        else:
            actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
