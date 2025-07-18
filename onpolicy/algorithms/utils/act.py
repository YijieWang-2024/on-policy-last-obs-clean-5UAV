from .distributions import Bernoulli, DiagGaussian, DiagBeta
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
        # self.continuous_associate = args.continuous_associate
        # self.nearest_associate = args.nearest_associate
        self.not_process_action = args.not_process_action
        self.tanh_gaussian = args.tanh_gaussian
        if action_space.__class__.__name__ == "Box":
            self.mujoco_box = True
            self.action_dim = action_space.shape[0]
            # self.action_out = DiagBeta(inputs_dim, action_dim, use_orthogonal, gain)
            self.action_out = DiagGaussian(inputs_dim, self.action_dim, use_orthogonal, gain)
        else:  # discrete + continous
            self.mixed_action = True
            self.action_outs = nn.ModuleList()
            self.action_dims = []
            for action_space_i in action_space:
                self.action_dims.append(action_space_i.shape[0])
                if action_space_i.__class__.__name__ == "Box":
                    # self.action_outs.append(DiagBeta(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
                    self.action_outs.append(DiagGaussian(inputs_dim, action_space_i.shape[0], use_orthogonal, gain))
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
                    # 伯努利没有.mean()
                    action = dist.sample()
                # # 不额外处理动作，这里输出时对动作归一化。这里的if条件也是很固定的。必须i位于后两个带宽资源和计算资源的动作。     0710这是干嘛的？没懂
                # if self.not_process_action:
                #     if action_out.__class__.__name__ == "DiagGaussian" and i in [len(self.action_dims)-1, len(self.action_dims)-2]:
                #         action = torch.nn.functional.normalize(action, p=1, dim=1)

                action_log_prob = dist.log_probs(action)
                if self.tanh_gaussian:
                    if dist.avail_actions is None:
                        action_log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action)) - np.log(2)).sum(-1, keepdim=True)
                        action = (torch.tanh(action) + 1) / 2
                    else:
                        rect = (2 * (np.log(2) - action - F.softplus(-2 * action)) - np.log(2)) * (dist.avail_actions > 0).float()
                        action_log_prob -= rect.sum(-1, keepdim=True)
                        action = (torch.tanh(action)+1)/2 * dist.avail_actions
                actions.append(action)
                action_log_probs.append(action_log_prob)
                # 这个是用来修改分配B和F_m的avail_actions的。所以分配B和F_m的动作一定要在allocation link之后。
                if action_out.__class__.__name__ == "Bernoulli" and available_actions is not None:
                    available_actions = available_actions * action
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
                if self.tanh_gaussian:
                    if dist.avail_actions is None:
                        log_prob -= (torch.log(1 - action[i] ** 2) - np.log(2)).sum(-1, keepdim=True)
                        jacobian_log = (torch.log(1 - action[i] ** 2) - np.log(2)).sum(-1, keepdim=True)
                        entropy += jacobian_log
                    else:
                        rect = (torch.log(1 - action[i] ** 2) - np.log(2)) * (dist.avail_actions > 0).float()
                        log_prob -= rect.sum(-1, keepdim=True)
                        jacobian_log = (torch.log(1 - action[i] ** 2) - np.log(2)) * (dist.avail_actions > 0).float()
                        entropy += jacobian_log.sum(-1, keepdim=True)
                action_log_probs.append(log_prob)
                dist_entropy.append(entropy)
                # 这个是用来修改分配B和F_m的avail_actions的。所以分配B和F_m的动作一定要在allocation link之后。
                if action_out.__class__.__name__=="Bernoulli":
                    available_actions = available_actions * action[i]
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
