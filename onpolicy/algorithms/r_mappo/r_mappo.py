import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.algorithms.utils.util import check
import math
import warnings


class R_MAPPO():
    """
    改进的MAPPO训练器，增加了数值稳定性和异常检测
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.average_local_advantage = args.average_local_advantage
        self.average_local_advantage_timely = args.average_local_advantage_timely
        self.average_neighbor_advantage = args.average_neighbor_advantage
        self.num_updates = 0
        self.continue_training = True

        # 新增参数
        self.action_dim = 63
        # 新增参数与阈值设置
        # 高维空间下使用更小的阈值，防止训练不稳定
        base_threshold = 5.0  # 基础阈值
        dim_factor = max(1.0, np.log(self.action_dim) / 3.0)  # 维度调整因子
        # 阈值随着维度增加而减小
        self.log_ratio_clip = base_threshold / dim_factor  # 对数比率裁剪值
        self.max_ratio = min(20.0, np.exp(self.log_ratio_clip))  # 确保与log裁剪一致
        self.ratio_warning = self.max_ratio * 0.75  # 警告阈值为最大值的75%
        self.ratio_stat_warning = 2.0  # 统计警告阈值，更保守
        self.eps = 1e-8  # 数值稳定性epsilon
        self.use_ratio_clipping = getattr(args, "use_ratio_clipping", True) # 这个只在这里局部用了，实际并没有设为参数
        self.use_kl_threshold = getattr(args, "use_kl_threshold", False)    # 在train_mec.py中添加了这个参数
        if self.use_kl_threshold:
            self.max_kl_threshold = getattr(args, 'max_kl_threshold', 0.02) # 这个只在这里局部用了，实际并没有设为参数
        print(f"动作空间维度: {self.action_dim}")
        print(f"PPO阈值设置 - log_ratio裁剪: {self.log_ratio_clip:.4f}, "
              f"max_ratio: {self.max_ratio:.4f}, "
              f"警告阈值: {self.ratio_warning:.4f}, "
              f"统计警告阈值: {self.ratio_stat_warning:.4f}")

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        # 添加跟踪统计信息的变量
        self.kl_divergence_history = []
        self.ratio_max_history = []
        self.ratio_mean_history = []

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        计算价值函数损失(保持不变)
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def calculate_kl_divergence(self, old_action_log_probs_batch, action_log_probs):
        """
        计算新旧策略之间的KL散度
        """
        kl_div = torch.exp(old_action_log_probs_batch) * (old_action_log_probs_batch - action_log_probs)
        if self._use_policy_active_masks:
            kl_div = kl_div.mean()
        return kl_div.mean()

    def ppo_update(self, sample, update_actor=True):
        """
        更新actor和critic网络，增加了数值稳定性措施
        """
        attention_active_mask_batch = None
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, attention_active_mask_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch,
                                                                              attention_active_mask=attention_active_mask_batch)
        if self.use_kl_threshold:
            # # Calculate KL divergence
            # kl_divergence = self.calculate_kl_divergence(old_action_log_probs_batch, action_log_probs)
            # # Max KL threshold - adaptive based on recent performance

            # with torch.no_grad():
            #     log_ratio = action_log_probs - old_action_log_probs_batch
            #     kl_divergence = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

            with torch.no_grad():
                kl_divergence = (torch.exp(old_action_log_probs_batch) * (old_action_log_probs_batch - action_log_probs)).mean()
            # Skip policy update if KL is too high
            if kl_divergence > self.max_kl_threshold and update_actor:
                warnings.warn(
                    f"KL divergence too high: {kl_divergence:.4f} > {self.max_kl_threshold:.4f}, skipping policy update")
                update_actor = False
                self.continue_training = False
                return [],[],[],[],[],[]

        # 重要性权重计算前，防止数值问题
        log_ratio = action_log_probs - old_action_log_probs_batch

        # 检测并处理log_ratio异常值，使用self.log_ratio_clip
        if torch.isnan(log_ratio).any() or torch.isinf(log_ratio).any():
            warnings.warn(f"检测到log_ratio中有NaN或Inf值，已裁剪到[-{self.log_ratio_clip:.4f}, {self.log_ratio_clip:.4f}]范围内")
        log_ratio = torch.clamp(log_ratio, -self.log_ratio_clip, self.log_ratio_clip)

        # 计算重要性权重，使用exp(log_ratio)而不是直接除法，提高数值稳定性
        ratio = torch.exp(log_ratio)

        # 直接裁剪比率，防止极端值，使用self.max_ratio
        if self.use_ratio_clipping:
            ratio = torch.clamp(ratio, 0.0, self.max_ratio)
        # 记录最大和平均重要性权重，用于监控训练稳定性
        ratio_max = ratio.max().item()
        ratio_mean = ratio.mean().item()
        self.ratio_max_history.append(ratio_max)
        self.ratio_mean_history.append(ratio_mean)
        # 使用self.ratio_warning作为警告阈值
        if ratio_max > self.ratio_warning:
            warnings.warn(f"重要性权重过大: max={ratio_max:.4f}, mean={ratio_mean:.4f}, 阈值={self.ratio_warning:.4f}")

        # 计算PPO目标函数
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        policy_loss = policy_action_loss
        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            if actor_grad_norm > 100.0:
                warnings.warn(f"Actor梯度范数异常: {actor_grad_norm}")
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        # 更新Critic网络
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            if critic_grad_norm > 100.0:
                warnings.warn(f"Critic梯度范数异常: {critic_grad_norm}")
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()
        self.num_updates += 1
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio

    def train(self, buffer, update_actor=True):
        """
        使用小批量梯度下降进行训练更新，增加了训练稳定性监控
        """
        if self.average_local_advantage_timely and self.average_local_advantage:
            advantages = buffer.advantages
        elif (not self.average_local_advantage_timely) and (self.average_local_advantage or self.average_neighbor_advantage):
            advantages = buffer.advantages
        else:
            if self._use_popart or self._use_valuenorm:
                advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
            else:
                advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        # if (not self.average_local_advantage) or (not self.average_neighbor_advantage):
        #     if self._use_popart or self._use_valuenorm:
        #         advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        #     else:
        #         advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        # else:
        #     advantages = buffer.advantages
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + self.eps)
        # 可选：额外裁剪极端优势值
        advantages = np.clip(advantages, -10.0, 10.0)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['ratio_max'] = 0  # 添加最大重要性权重追踪

        self.num_updates = 0
        self.continue_training = True
        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                if not self.continue_training:
                    break

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm

                # 收集重要性权重统计信息
                ratio_mean = imp_weights.mean().item()
                ratio_max = imp_weights.max().item()
                train_info['ratio'] += ratio_mean
                train_info['ratio_max'] = max(train_info['ratio_max'], ratio_max)

            if not self.continue_training:
                break
        # num_updates = self.ppo_epoch * self.num_mini_batch

        num_updates = self.num_updates

        for k in train_info.keys():
            if k != 'ratio_max':  # 最大值不需要平均
                train_info[k] /= num_updates

        # 在训练结束时打印统计信息
        if len(self.ratio_max_history) > 0:
            recent_ratio_max = np.mean(self.ratio_max_history[-10:]) if len(self.ratio_max_history) >= 10 else np.mean(self.ratio_max_history)
            # 使用self.ratio_stat_warning作为统计警告阈值
            if recent_ratio_max > self.ratio_stat_warning:
                warnings.warn(
                    f"重要性权重持续偏大，平均最大值: {recent_ratio_max:.4f}，阈值: {self.ratio_stat_warning:.4f}，"
                    f"建议减小学习率或增加PPO剪裁参数")

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
