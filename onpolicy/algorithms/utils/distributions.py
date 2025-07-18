import torch
import torch.nn as nn
from .util import init
import torch.nn.functional as F
import numpy as np


class FixedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, avail_actions=None):
        super().__init__(loc, scale)
        self.avail_actions = avail_actions

    def log_probs(self, actions):

        # raw_actions = torch.log(actions+1e-8) - torch.log(1-actions+1e-8)
        # log_prob_values = super().log_prob(raw_actions) + torch.log(actions+1e-8)+torch.log(1-actions+1e-8)

        log_prob_values = super().log_prob(actions)
        if self.avail_actions is not None:
            # 将不可用动作的log_prob设为0（不参与计算）
            log_prob_values = log_prob_values * (self.avail_actions > 0).float()
        return log_prob_values.sum(-1, keepdim=True)

    def entropy(self):
        entropy_values = super().entropy()
        if self.avail_actions is not None:
            # 只考虑可用动作的熵
            entropy_values = entropy_values * (self.avail_actions > 0).float()
        return entropy_values.sum(-1, keepdim=True)

    def mode(self):
        mode_values = self.mean
        if self.avail_actions is not None:
            # 对不可用的动作，将mode设为0
            mode_values = mode_values * self.avail_actions
        return mode_values

    def sample(self):
        samples = super().sample()
        # samples = torch.sigmoid(samples)
        if self.avail_actions is not None:
            # 对不可用的动作，将采样值设为0
            samples = samples * self.avail_actions
        return samples


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def __init__(self, logits=None, probs=None, avail_actions=None):
        self.avail_actions = avail_actions
        # 如果有可用动作掩码，调整logits或probs
        if avail_actions is not None:
            if logits is not None:
                # 对不可用的动作设置极小的logits值（相当于概率为0）
                masked_logits = logits.clone()
                masked_logits = masked_logits.masked_fill_(avail_actions == 0, -1e10)
                super().__init__(logits=masked_logits)
            elif probs is not None:
                # 对不可用的动作设置概率为0
                masked_probs = probs.clone() * avail_actions
                super().__init__(probs=masked_probs)
        else:
            super().__init__(logits=logits, probs=probs)

    def log_probs(self, actions):
        # 只考虑可用动作的log_prob
        log_prob_values = super().log_prob(actions)
        if self.avail_actions is not None:
            # 将不可用动作的log_prob设为0（因为它们不应影响总体结果）
            log_prob_values = log_prob_values * (self.avail_actions > 0).float()
        return log_prob_values.view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        entropy_values = super().entropy()
        if self.avail_actions is not None:
            # 只考虑可用动作的熵
            entropy_values = entropy_values * (self.avail_actions > 0).float()
        return entropy_values.sum(-1, keepdim=True)

    def mode(self):
        mode_values = torch.gt(self.probs, 0.5).float()
        if self.avail_actions is not None:
            # 确保不可用动作的mode为0
            mode_values = mode_values * self.avail_actions
        return mode_values

    def sample(self):
        samples = super().sample()
        if self.avail_actions is not None:
            # 确保不可用动作的采样结果为0
            samples = samples * self.avail_actions
        return samples


class FixedBeta(torch.distributions.Beta):
    def __init__(self, concentration1, concentration0, avail_actions=None):
        super().__init__(concentration1, concentration0)
        self.avail_actions = avail_actions

    def log_probs(self, actions):
        log_prob_values = super().log_prob(actions)
        if self.avail_actions is not None:
            # 将不可用动作的log_prob设为0（不参与计算）
            log_prob_values = log_prob_values * (self.avail_actions > 0).float()
        return log_prob_values.view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        entropy_values = super().entropy()
        if self.avail_actions is not None:
            # 只考虑可用动作的熵
            entropy_values = entropy_values * (self.avail_actions > 0).float()
        return entropy_values.sum(-1)

    def sample(self):
        samples = super().sample()
        if self.avail_actions is not None:
            # 对不可用的动作，将采样值设为默认值（0.5）
            default_value = torch.ones_like(samples) * 0.5
            samples = torch.where(self.avail_actions > 0, samples, default_value)
        return samples

    def mode(self):
        # Beta分布的众数：(alpha - 1)/(alpha + beta - 2)，当alpha和beta都大于1时
        mode_vals = (self.concentration1 - 1) / (self.concentration1 + self.concentration0 - 2)
        # 处理alpha或beta小于等于1的情况
        alpha_le_1 = self.concentration1 <= 1
        beta_le_1 = self.concentration0 <= 1
        mode_vals = torch.where(alpha_le_1 & beta_le_1, 0.5, mode_vals)
        mode_vals = torch.where(alpha_le_1 & ~beta_le_1, 0.0, mode_vals)
        mode_vals = torch.where(~alpha_le_1 & beta_le_1, 1.0, mode_vals)

        if self.avail_actions is not None:
            # 对不可用的动作，将mode值设为默认值（0.5）
            default_value = torch.ones_like(mode_vals) * 0.5
            mode_vals = torch.where(self.avail_actions > 0, mode_vals, default_value)
        return mode_vals


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.ones(num_outputs) * np.log(0.4))

    def forward(self, x, avail_actions=None):
        action_mean = self.fc_mean(x)
        action_mean = torch.sigmoid(action_mean)

        # An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda(device=x.device.index)
        action_logstd = self.logstd(zeros)

        return FixedNormal(action_mean, action_logstd.exp(), avail_actions)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, avail_actions=None):
        x = self.linear(x)
        return FixedBernoulli(logits=x, avail_actions=avail_actions)


class DiagBeta(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagBeta, self).__init__()
        self.alpha_layer = nn.Linear(num_inputs, num_outputs)
        self.beta_layer = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.alpha_layer.weight, gain=gain)
        nn.init.constant_(self.alpha_layer.bias, 0)
        nn.init.orthogonal_(self.beta_layer.weight, gain=gain)
        nn.init.constant_(self.beta_layer.bias, 0)

    def forward(self, s, avail_actions=None):
        # alpha and beta need to be larger than 1, so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0

        # 创建Beta分布，添加avail_actions支持
        dist = FixedBeta(alpha, beta, avail_actions)
        return dist

    def get_dist(self, s, avail_actions=None):
        # 为了兼容性保留此方法
        return self.forward(s, avail_actions)

    def mean(self, s, avail_actions=None):
        dist = self.forward(s, avail_actions)
        mean_vals = dist.mean
        if avail_actions is not None:
            # 对不可用的动作，将均值设为默认值（0.5）
            default_value = torch.ones_like(mean_vals) * 0.5
            mean_vals = torch.where(avail_actions > 0, mean_vals, default_value)
        return mean_vals


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
