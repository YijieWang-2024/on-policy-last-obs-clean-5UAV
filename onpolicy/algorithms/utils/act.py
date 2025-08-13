from .distributions import Bernoulli, DiagGaussian, DiagBeta, DiagDirichlet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.mujoco_box = False
        self.action_type = action_space.__class__.__name__
        # self.discrete_associate = args.discrete_associate
        self.continuous_associate = args.continuous_associate
        self.ave_resource = args.ave_resource
        self.ave_bandwidth = args.ave_bandwidth
        self.fix_uav_pos = args.fix_uav_pos
        print("在act.py中使用到了连续连接动作的0.5阈值。")
        # self.nearest_associate = args.nearest_associate
        self.not_process_action = args.not_process_action
        if action_space.__class__.__name__ == "Box":
            self.mujoco_box = True
            self.action_dim = action_space.shape[0]
            # self.action_out = DiagBeta(inputs_dim, action_dim, use_orthogonal, gain)
            self.action_out = DiagGaussian(inputs_dim, self.action_dim, use_orthogonal, gain)
        else:  # discrete + continous
            self.mixed_action = True
            self.action_outs = nn.ModuleList()
            self.action_dims = []
            for i, action_space_i in enumerate(action_space):
                self.action_dims.append(action_space_i.shape[0])
                if action_space_i.__class__.__name__ == "Box":
                    # 后边资源分配选择狄利克雷分布
                    if self.continuous_associate:
                        if self.fix_uav_pos:
                            if i >= 1:
                                self.action_outs.append(DiagDirichlet(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
                            else:
                                self.action_outs.append(DiagGaussian(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
                        else:
                            if i >= 2:
                                self.action_outs.append(DiagDirichlet(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
                            else:
                                self.action_outs.append(DiagGaussian(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
                    else:
                        self.action_outs.append(DiagGaussian(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
                    # # 全都选择高斯分布。
                    # self.action_outs.append(DiagGaussian(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
                else:   # multibonulli
                    self.action_outs.append(Bernoulli(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
    
    def forward(self, x, available_actions=None, deterministic=False):
        if self.mujoco_box:
            if self.action_dim != available_actions.shape[-1]:
                available_actions = None
            dist = self.action_out(x, available_actions)
            actions = dist.mode() if deterministic else dist.sample()  # Sample the action according to the probability distribution
            action_log_probs = dist.log_prob(actions)  # The log probability density of the action
            action_log_probs = torch.sum(action_log_probs, -1, keepdim=True)
            # action_log_probs = (action_log_probs * available_actions).sum(-1, keepdim=True) / available_actions.sum(-1, keepdim=True) # 高斯里边用avail_actions计算过logp了
        else:   # discrete + continous
            actions = []
            action_log_probs = []
            for i, action_out in enumerate(self.action_outs):
                if available_actions is not None:
                    if self.action_dims[i] != available_actions.shape[-1]:
                        available_action = None
                    else:
                        available_action = available_actions
                else:
                    available_action = None
                dist = action_out(x, available_action)
                if action_out.__class__.__name__ != "Bernoulli":
                    action = dist.mode() if deterministic else dist.sample()  # Sample the action according to the probability distribution
                else:
                    action = dist.sample()  # 伯努利没有.mean()
                action_log_prob = dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
                # 这个是用来修改分配B和F_m的avail_actions的。所以分配B和F_m的动作一定要在allocation link之后。
                if action_out.__class__.__name__ == "Bernoulli" and available_actions is not None:
                    available_actions = available_actions * action
                if self.continuous_associate and available_actions is not None:    # 连接动作的0.5阈值。
                    if self.fix_uav_pos:
                        if i == 0:
                            available_actions = available_actions * (action>=0.5)
                    else:
                        if i == 1:
                            available_actions = available_actions * (action>=0.5)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        return actions, action_log_probs

    # def get_probs(self, x, available_actions=None):
    #     if self.mixed_action
    #         action_probs = []
    #         for action_out in self.action_outs:
    #             action_logit = action_out(x)
    #             action_prob = action_logit.probs
    #             action_probs.append(action_prob)
    #         action_probs = torch.cat(action_probs, -1)
    #     else:
    #         action_logits = self.action_out(x, available_actions)
    #         action_probs = action_logits.probs
    #
    #     return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        if self.mujoco_box:
            if self.action_dim != available_actions.shape[-1]:
                available_actions = None
            dist = self.action_out(x, available_actions)
            action_log_probs = dist.log_prob(action)  # The log probability density of the action
            action_log_probs = torch.sum(action_log_probs, -1, keepdim=True)
            # action_log_probs = (action_log_probs * available_actions).sum(-1, keepdim=True) / available_actions.sum(-1, keepdim=True)
            dist_entropy = dist.entropy()
            if active_masks is not None:
                dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = dist_entropy.mean()
        else:   # discrete + continous
            action = action.split(self.action_dims, dim=-1)
            action_log_probs = [] 
            dist_entropy = []
            for i, action_out in enumerate(self.action_outs):
                if self.action_dims[i] != available_actions.shape[-1]:
                    available_action = None
                else:
                    available_action = available_actions
                dist = action_out(x, available_action)
                log_prob = dist.log_probs(action[i])
                entropy = dist.entropy()
                action_log_probs.append(log_prob)
                dist_entropy.append(entropy)
                # 这个是用来修改分配B和F_m的avail_actions的。所以分配B和F_m的动作一定要在allocation link之后。
                if action_out.__class__.__name__=="Bernoulli" and available_actions is not None:
                    available_actions = available_actions * action[i]
                if self.continuous_associate and available_actions is not None:    # 连接动作的0.5阈值。
                    if self.fix_uav_pos:
                        if i == 0:
                            available_actions = available_actions * (action[i]>=0.5)
                    else:
                        if i == 1:
                            available_actions = available_actions * (action[i]>=0.5)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            if active_masks is not None:
                dist_entropy = (torch.sum(torch.cat(dist_entropy, -1), -1) * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = torch.sum(torch.cat(dist_entropy, -1), -1).mean()
        return action_log_probs, dist_entropy


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
