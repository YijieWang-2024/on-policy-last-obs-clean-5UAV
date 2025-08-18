import time
# import wandb
import numpy as np
import torch
from onpolicy.runner.separated.base_runner import Runner
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.utils.util import get_shape_from_obs_space

def _t2n(x):
    return x.detach().cpu().numpy()

class MECRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(MECRunner, self).__init__(config)
        self.normer = []
        self.n_UAVs = config['all_args'].n_UAVs
        for i in range(self.n_UAVs):
            self.normer.append(Normer(args=self.all_args, obs_space=get_shape_from_obs_space(self.envs.observation_space[i]), states_space=get_shape_from_obs_space(self.envs.share_observation_space[i])))
        # self.normer = Normer(args=self.all_args, obs_space=get_shape_from_obs_space(self.envs.observation_space[0]), states_space=get_shape_from_obs_space(self.envs.share_observation_space[0]))
        if self.model_dir is not None:
            self.restore()
        self.average_local_advantage = config['all_args'].average_local_advantage
        self.average_local_advantage_timely = config['all_args'].average_local_advantage_timely
        self.whether_local_add_ave_adadvantage = config['all_args'].whether_local_add_ave_adadvantage
        self.whether_local_add_direct_ave_adv = config['all_args'].whether_local_add_direct_ave_adv
        self.local_add_T_ave_adv = config['all_args'].local_add_T_ave_adv
        self.average_neighbor_advantage = config['all_args'].average_neighbor_advantage
        self.whether_average_network_parameters = config['all_args'].whether_average_network_parameters
        self.average_network_parameters_interval = config['all_args'].average_network_parameters_interval
        if self.whether_average_network_parameters:
            self.average_network_parameters()
        # self.n_UAVs = config['all_args'].n_UAVs
        # self.uav_positions = np.zeros((self.n_rollout_threads, self.n_UAVs, 2))
        # self.local_advantages = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        # # self.uav_positions[0] = np.array(
        # #             [[90, 90], [270, 90], [450, 90], [630, 90], [810, 90],
        # #              [90, 270], [270, 270], [450, 270], [630, 270], [810, 270],
        # #              [90, 450], [270, 450], [450, 450], [630, 450], [810, 450],
        # #              [90, 630], [270, 630], [450, 630], [630, 630], [810, 630],
        # #              [90, 810], [270, 810], [450, 810], [630, 810], [810, 810]], dtype=np.float)
        # self.use_errors_noise = config['all_args'].use_errors_noise
        self.use_errors_noise = getattr(config['all_args'], "use_errors_noise", False)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions    (在collect里边，调用env处理了动作。输出处理后的动作，和对应的log_p)
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions, Metropolis_weights, attention_active_mask = self.envs.step(actions)
                for i in range(self.n_UAVs):
                    obs[:, i] = self.normer[i]._obfilt(obs[:, i])
                    share_obs[:, i] = self.normer[i]._statefilt(share_obs[:, i])
                    rewards[:, i] = self.normer[i]._rewsfilt(rewards[:, i], dones[:, i])
                # obs = self.normer._obfilt(obs)
                # share_obs = self.normer._statefilt(share_obs)
                # rewards = self.normer._rewsfilt(rewards, dones)
                # if len(infos) > 0:
                #     for i, info in enumerate(infos):
                #         self.uav_positions[step, i] = info['uav_positions']

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, Metropolis_weights, attention_active_mask
                
                # insert data into buffer
                self.insert(data, step)

            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     for i, info in enumerate(infos):
            #         self.uav_positions[i] = info['uav_positions']
            #     np.save(str(self.run_dir) + '/uav_positions_' + str(episode) + '.npy', self.uav_positions)
            #     for agent_id in range(self.num_agents):
            #         self.local_advantages[agent_id] = self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1]
            #     np.save(str(self.run_dir) + '/local_advantages_' + str(episode) + '.npy', self.local_advantages)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            if self.whether_average_network_parameters and (episode % self.average_network_parameters_interval == 0):
                self.average_network_parameters()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == 'mec':
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update({'cumulative_reward': np.mean(np.mean([info['cumulative_reward'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'cumulative_individual_reward_wo_cover': np.mean([info['cumulative_individual_reward_wo_cover'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'cumulative_reward_wo_cover': np.mean(np.mean([info['cumulative_reward_wo_cover'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'system_performance': np.mean(np.mean([info['system_performance'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'system_performance_true_all_GUs': np.mean(np.mean([info['system_performance_true_all_GUs'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'system_performance_individual': np.mean([info['system_performance_individual'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'cumulative_individual_reward': np.mean([info['cumulative_individual_reward'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'delay_true_all_GUs': np.mean([info['delay_true_all_GUs'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'delay_true_coverd_GUs': np.mean([info['delay_true_coverd_GUs'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'energy_true_all_GUs': np.mean([info['energy_true_all_GUs'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'energy_all_GUs_UAVs': np.mean([info['energy_all_GUs_UAVs'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'n_GUs_by_coverd': np.mean([info['n_GUs_by_coverd'] for info in infos]).round(5)})
                        train_infos[agent_id].update({'complete_task_ratio': np.mean([info['complete_task_ratio'] for info in infos]).round(5)})

                print('cumulative_reward is ', np.mean([info['cumulative_reward'] for info in infos], axis=0).round(5))
                print('system_performance is ', np.mean([info['system_performance'] for info in infos], axis=0).round(5))
                print('system_performance_true_all_GUs is ', np.mean([info['system_performance_true_all_GUs'] for info in infos], axis=0).round(5))
                print('system_performance_individual is ', np.mean([info['system_performance_individual'] for info in infos], axis=0).round(5))
                print('cumulative_individual_reward is ', np.mean([info['cumulative_individual_reward'] for info in infos], axis=0).round(5))
                print('complete_task_ratio is ', np.mean([info['complete_task_ratio'] for info in infos], axis=0).round(5))
                print('delay_true_all_GUs is ', np.mean([info['delay_true_all_GUs'] for info in infos], axis=0).round(5))
                print('UAV 0 value_loss is {}, policy_loss is {}, dist_entropy is {}, actor_grad_norm is {}, critic_grad_norm is {}, ratio is {}'.format(train_infos[0]['value_loss'],
                                                                                                                                                train_infos[0]['policy_loss'],
                                                                                                                                                train_infos[0]['dist_entropy'],
                                                                                                                                                train_infos[0]['actor_grad_norm'],
                                                                                                                                                train_infos[0]['critic_grad_norm'],
                                                                                                                                                train_infos[0]['ratio']))

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions, Metropolis_weights, attention_active_mask = self.envs.reset()
        for i in range(self.n_UAVs):
            obs[:, i] = self.normer[i]._obfilt(obs[:, i])
            share_obs[:, i] = self.normer[i]._statefilt(share_obs[:, i])
        # obs = self.normer._obfilt(obs)
        # share_obs = self.normer._statefilt(share_obs)


        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()
            self.buffer[agent_id].Metropolis_weights[0] = Metropolis_weights[:, agent_id].copy()
            self.buffer[agent_id].attention_active_mask[0] = attention_active_mask[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []

        raw_action_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            # 这里的action_log_prob不能用来计算policy_loss，因为这个还没有经过处理，不是真正用到环境的动作。
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            self.buffer[agent_id].available_actions[step],
                                                            attention_active_mask = self.buffer[agent_id].attention_active_mask[step])
            raw_action_collector.append(_t2n(action))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
            value_collector.append(_t2n(value))
            action_log_prob_collector.append(_t2n(action_log_prob))
        # # [self.envs, agents, dim]
        # raw_actions = np.array(raw_actions).transpose(1, 0, 2)
        # processed_actions = self.envs.process_actions(raw_actions)
        # for agent_id in range(self.num_agents):
        #     value, action_log_prob, _ = self.trainer[agent_id].policy.evaluate_actions(self.buffer[agent_id].share_obs[step],
        #                                                                                self.buffer[agent_id].obs[step],
        #                                                                                self.buffer[agent_id].rnn_states[step],
        #                                                                                self.buffer[agent_id].rnn_states_critic[step],
        #                                                                                processed_actions[:, agent_id],
        #                                                                                self.buffer[agent_id].masks[step],
        #                                                                                self.buffer[agent_id].available_actions[step],
        #                                                                                attention_active_mask = self.buffer[agent_id].attention_active_mask[step])
        #     value_collector.append(_t2n(value))
        #     action_log_prob_collector.append(_t2n(action_log_prob))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        raw_actions = np.array(raw_action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)
        
        # return values, processed_actions, action_log_probs, rnn_states, rnn_states_critic
        return values, raw_actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data, step):
        buffer_step = step
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, Metropolis_weights, attention_active_mask = data

        dones_env = np.all(dones, axis=1)
        # rnn狀態重置
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states.shape[2:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)
        # 數據有沒有用的mask。因爲有的episode會提前結束，每個episode長度不同。所以有這個來mask不同長度。是用来重置rnn_state的
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # 添加可用的动作空间小于2时，记录该智能体死亡。不计算梯度。为了利用狄利克雷分布。
        covering_GUs = np.sum(available_actions, axis=-1)
        active_masks[covering_GUs < 1] = np.zeros(((covering_GUs < 1).sum(), 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id], action_log_probs[:, agent_id], values[:, agent_id],
                                         rewards[:, agent_id], masks[:, agent_id], active_masks = active_masks[:, agent_id],
                                         available_actions=available_actions[:, agent_id],
                                         Metropolis_weights=Metropolis_weights[:, agent_id], attention_active_mask=attention_active_mask[:, agent_id])
        #正常流程是在buffer里添加一个delta的属性，存每个step的delta。然后将delta按照下边的方法更新。
        # # 这个对v的平均，有问题呐。推测是会影响Critic的更新。
        # if self.average_local_advantage_timely:
        #     # 如果所有之前时刻的values都按照当前时刻的邻居权重来做平均
        #     assert self.average_local_advantage
        #     all_values = np.zeros((self.num_agents, step+1, self.n_rollout_threads), dtype=np.float32)
        #     all_Metropolis_weights = np.zeros((self.num_agents, step+1, self.n_rollout_threads, self.num_agents), dtype=np.float32)
        #     for agent_id in range(self.num_agents):
        #         all_values[agent_id] = self.buffer[agent_id].value_preds[:step+1].squeeze(-1)
        #         # 所有的values都按照当前时刻的邻居权重来做平均
        #         all_Metropolis_weights[agent_id] = np.tile(self.buffer[agent_id].Metropolis_weights[[step]], (step+1, 1, 1))
        #     updated_all_values = np.einsum('istj,jst->ist', all_Metropolis_weights, all_values)
        #     for agent_id in range(self.num_agents):
        #         self.buffer[agent_id].value_preds[:step+1] = updated_all_values[agent_id, :, :, np.newaxis].copy()

    def train(self):
        train_infos = []
        if self.average_local_advantage_timely and self.average_local_advantage:
            if self.whether_local_add_direct_ave_adv:
                # 直接local+True_mean
                local_advantage = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages = self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1]
                    local_advantage[agent_id] = self.buffer[agent_id].advantages.copy()
                mean_advantage = np.mean(local_advantage, axis=0)
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages += mean_advantage
            elif self.whether_local_add_ave_adadvantage:
                # buffer更新。local+updated_mean
                local_advantage = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages = self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1]
                    local_advantage[agent_id] = self.buffer[agent_id].advantages.copy()
                mean_advantage = local_advantage.copy()
                for step in range(self.episode_length):
                    all_advantages = mean_advantage[:, :step+1].squeeze(-1)
                    all_Metropolis_weights = np.zeros((self.num_agents, step+1, self.n_rollout_threads, self.num_agents), dtype=np.float32)
                    for agent_id in range(self.num_agents):
                        all_Metropolis_weights[agent_id] = np.tile(self.buffer[agent_id].Metropolis_weights[[step]], (step+1, 1, 1))
                    updated_advantages = np.einsum('istj,jst->ist', all_Metropolis_weights, all_advantages)
                    mean_advantage[:, :step+1] = updated_advantages[..., np.newaxis].copy()
                    # if step == 0:
                    #     advantages_0_mean = np.mean(all_advantages[:, 0], axis=0)
                    #     error_0 = np.mean(np.linalg.norm(all_advantages[:, 0] - advantages_0_mean, axis=0))
                    # # 记录下平均的进度的代码。
                    # error_step = np.mean(np.linalg.norm(updated_advantages[:,0] - advantages_0_mean, axis=0))
                    # print('第',step,'步:', error_step/error_0)
                if self.use_errors_noise:
                    True_mean_advantage = np.mean(local_advantage, axis=0, keepdims=True)
                    errors = np.abs(mean_advantage - True_mean_advantage) / np.abs(local_advantage - True_mean_advantage)
                else:
                    for agent_id in range(self.num_agents):
                        self.buffer[agent_id].advantages = mean_advantage[agent_id] + local_advantage[agent_id]
            elif self.local_add_T_ave_adv != 0:
                # 按时间步的T步更新。 t=0的不会更新buffer_size步数了。
                local_advantage = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages = self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1]
                    local_advantage[agent_id] = self.buffer[agent_id].advantages.copy()
                mean_advantage = local_advantage.copy()
                for step in range(self.episode_length):
                    all_advantages = mean_advantage[:, max(step + 1 - self.local_add_T_ave_adv, 0):step + 1].squeeze(-1)
                    all_Metropolis_weights = np.zeros((self.num_agents, min(self.local_add_T_ave_adv, step + 1), self.n_rollout_threads, self.num_agents), dtype=np.float32)
                    for agent_id in range(self.num_agents):
                        all_Metropolis_weights[agent_id] = np.tile(self.buffer[agent_id].Metropolis_weights[[step]], (min(self.local_add_T_ave_adv, step + 1), 1, 1))
                    updated_advantages = np.einsum('istj,jst->ist', all_Metropolis_weights, all_advantages)
                    mean_advantage[:, max(step + 1 - self.local_add_T_ave_adv, 0):step + 1] = updated_advantages[..., np.newaxis].copy()
                if self.use_errors_noise:
                    True_mean_advantage = np.mean(local_advantage, axis=0, keepdims=True)
                    errors = np.abs(mean_advantage - True_mean_advantage) / np.abs(local_advantage - True_mean_advantage)
                else:
                    for agent_id in range(self.num_agents):
                        self.buffer[agent_id].advantages = mean_advantage[agent_id] + local_advantage[agent_id]
                # # 调试到这里，然后使用下边的代码去debug
                # for T in [1, 2, 4, 6, 8, 10, 15, 20]:
                #     # T = 20
                #     mean_advantage = local_advantage.copy()
                #     for step in range(self.episode_length):
                #         all_advantages = mean_advantage[:, max(step + 1 - T, 0):step + 1].squeeze(-1)
                #         all_Metropolis_weights = np.zeros(
                #             (self.num_agents, min(T, step + 1), self.n_rollout_threads, self.num_agents),
                #             dtype=np.float32)
                #         for agent_id in range(self.num_agents):
                #             all_Metropolis_weights[agent_id] = np.tile(self.buffer[agent_id].Metropolis_weights[[step]],
                #                                                        (min(T, step + 1), 1, 1))
                #         updated_advantages = np.einsum('istj,jst->ist', all_Metropolis_weights, all_advantages)
                #         mean_advantage[:, max(step + 1 - T, 0):step + 1] = updated_advantages[..., np.newaxis].copy()
                #     # advantages_0_mean = np.mean(local_advantage[:, :self.episode_length - T], axis=0)
                #     advantages_0_mean = np.mean(local_advantage[:, :], axis=0)
                #     error_0 = np.mean(np.linalg.norm(local_advantage[:, :] - advantages_0_mean, axis=0))
                #     error_step = np.mean(np.linalg.norm(mean_advantage[:, :] - advantages_0_mean, axis=0))
                #     print('第', T, '步:', error_step / error_0)
            else:
                # buffer更新。local
                local_advantage = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages = self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1]
                    local_advantage[agent_id] = self.buffer[agent_id].advantages.copy()
                mean_advantage = local_advantage.copy()
                for step in range(self.episode_length):
                    all_advantages = mean_advantage[:, :step+1].squeeze(-1)
                    all_Metropolis_weights = np.zeros((self.num_agents, step + 1, self.n_rollout_threads, self.num_agents), dtype=np.float32)
                    for agent_id in range(self.num_agents):
                        all_Metropolis_weights[agent_id] = np.tile(self.buffer[agent_id].Metropolis_weights[[step]], (step + 1, 1, 1))
                    updated_advantages = np.einsum('istj,jst->ist', all_Metropolis_weights, all_advantages)
                    mean_advantage[:, :step + 1] = updated_advantages[..., np.newaxis].copy()
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages = mean_advantage[agent_id]
        elif (not self.average_local_advantage_timely) and (self.average_local_advantage or self.average_neighbor_advantage):
            # 没有timely，就是每一步的adv固定迭代更新n_iterations次。（这样好像也不对？前几步明明可以更新的更多的啊）
            n_iterations = self.all_args.n_iterations
            if self.average_neighbor_advantage:
                assert n_iterations==1
            all_advantages = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads), dtype=np.float32)
            all_Metropolis_weights = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, self.num_agents), dtype=np.float32)
            for agent_id in range(self.num_agents):
                advantages_i = self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1]
                all_advantages[agent_id] = advantages_i.squeeze(-1)
                all_Metropolis_weights[agent_id] = self.buffer[agent_id].Metropolis_weights[:-1]
            updated_advantages = all_advantages.copy()
            for _ in range(n_iterations):
                # This single line replaces the triple nested loop
                # 'istj,jst->ist' means: sum over j dimension (agent weights)
                updated_advantages = np.einsum('istj,jst->ist', all_Metropolis_weights, updated_advantages)
                # # 记录下平均的进度的代码。
                # mean_per_step_thread = np.mean(updated_advantages, axis=0)
                # abs_deviations = np.abs(updated_advantages - mean_per_step_thread)
                # avg_abs_deviation = np.mean(abs_deviations)
                # print(_, ':', avg_abs_deviation)
            # updated_advantages = all_advantages - updated_advantages
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].advantages = updated_advantages[agent_id, :, :, np.newaxis].copy()

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def average_network_parameters(self):
        """
        Average network parameters across all trainers.
        Each trainer in self.trainer is an instance of R_MAPPO.
        """
        if len(self.trainer) <= 1:
            return
        # Use the first trainer as a reference for parameter keys
        reference_trainer = self.trainer[0]
        # Get actor and critic models from the reference trainer
        actor_state = reference_trainer.policy.actor.state_dict()
        critic_state = reference_trainer.policy.critic.state_dict()
        # Initialize dictionaries to store summed parameters
        summed_actor_params = {key: torch.zeros_like(param) for key, param in actor_state.items()}
        summed_critic_params = {key: torch.zeros_like(param) for key, param in critic_state.items()}
        # Sum parameters from all trainers
        for trainer in self.trainer:
            actor_params = trainer.policy.actor.state_dict()
            critic_params = trainer.policy.critic.state_dict()
            for key in summed_actor_params:
                summed_actor_params[key] += actor_params[key]
            for key in summed_critic_params:
                summed_critic_params[key] += critic_params[key]
        # Compute average
        n_trainers = len(self.trainer)
        for key in summed_actor_params:
            summed_actor_params[key] = summed_actor_params[key] / n_trainers
        for key in summed_critic_params:
            summed_critic_params[key] = summed_critic_params[key] / n_trainers
        # Update all trainers with the averaged parameters
        for trainer in self.trainer:
            actor_state_dict = trainer.policy.actor.state_dict()
            critic_state_dict = trainer.policy.critic.state_dict()
            # Update with averaged parameters
            for key in actor_state_dict:
                actor_state_dict[key].copy_(summed_actor_params[key])
            for key in critic_state_dict:
                critic_state_dict[key].copy_(summed_critic_params[key])
            # Make sure the changes are applied
            trainer.policy.actor.load_state_dict(actor_state_dict)
            trainer.policy.critic.load_state_dict(critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        # train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                if agent_id==0 or k in ['cumulative_individual_reward', 'system_performance_individual', 'cumulative_individual_reward_wo_cover']:
                    agent_k = "agent%i/" % agent_id + k
                    if self.use_wandb:
                        wandb.log({agent_k: v}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        all_infos = {'cumulative_reward': [], 'uav_m_toal_energy_consumption': [], 'n_GUs_per_uav_served': [], 'user_in_m_average_delay': []}

        # eval_battles_won = 0
        eval_episode = 0

        # eval_episode_rewards = []
        # one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = self.normer._obfilt(eval_obs)
        eval_share_obs = self.normer._statefilt(eval_share_obs)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            else:
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            eval_obs = self.normer._obfilt(eval_obs)
            eval_share_obs = self.normer._statefilt(eval_share_obs)
            eval_rewards = self.normer._rewsfilt(eval_rewards, eval_dones)

            # one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, *eval_rnn_states.shape[2:]), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    all_infos['cumulative_reward'].append(np.sum(eval_infos[eval_i]['cumulative_reward']).round(5))
                    all_infos['uav_m_toal_energy_consumption'].append(np.sum(eval_infos[eval_i]['uav_m_toal_energy_consumption']).round(5))
                    all_infos['n_GUs_per_uav_served'].append(np.sum(eval_infos[eval_i]['n_GUs_per_uav_served']).round(5))
                    all_infos['user_in_m_average_delay'].append(np.mean(eval_infos[eval_i]['user_in_m_average_delay']).round(5))
                    eval_episode += 1
                    # eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    # one_episode_rewards = []
                    # if eval_infos[eval_i][0]['won']:
                    #     eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                print('cumulative_reward is {}, uav_m_toal_energy_consumption is {}, n_GUs_per_uav_served is {}, user_in_m_average_delay is {}'.format(np.mean(all_infos['cumulative_reward']),
                                                                                                                     np.mean(all_infos['uav_m_toal_energy_consumption']),
                                                                                                                     np.mean(all_infos['n_GUs_per_uav_served']),
                                                                                                                     np.mean(all_infos['user_in_m_average_delay'])))
                print('eval over')
                # eval_episode_rewards = np.array(eval_episode_rewards)
                # eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}
                # self.log_env(eval_env_infos, total_num_steps)
                # eval_win_rate = eval_battles_won/eval_episode
                # print("eval win rate is {}.".format(eval_win_rate))
                # if self.use_wandb:
                #     wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                # else:
                #     self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

            # 保存 normer 对象
            normer_path = str(self.save_dir) + "/normer"+ str(agent_id) +".pkl"
            self.normer[agent_id].save(normer_path)
        # # 保存 normer 对象
        # normer_path = str(self.save_dir)+ "/normer.pkl"
        # self.normer.save(normer_path)

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt', map_location='cpu')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt', map_location='cpu')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            # 恢复 normer 对象
            normer_path = str(self.model_dir) + "/normer"+ str(agent_id) +".pkl"
            self.normer[agent_id].load(normer_path)
        # # 恢复 normer 对象
        # normer_path = str(self.model_dir) + "/normer.pkl"
        # self.normer.load(normer_path)

    def run_consensus_algorithm(self, local_observations, max_iterations):
        """
            Run the consensus algorithm to share local observations among UAVs.
            Args:
                positions: UAV positions with shape (n_rollout_threads, n_UAVs, 2)
                local_observations: Local observations with shape (n_UAVs, episode_length, n_rollout_threads, 1)
                neighbor_distance: Maximum distance for two UAVs to be neighbors
                max_iterations: Maximum number of consensus iterations
            Returns:
                Updated observations after consensus with shape (n_UAVs, episode_length, n_rollout_threads, 1)
            """
        # 重新组织观测数据，使计算更直接
        # 新形状: (n_rollout_threads, n_UAVs, episode_length, 1)
        obs = np.transpose(local_observations, (2, 0, 1, 3))
        # 初始化共识估计值
        consensus_estimates = np.copy(obs)

        weight_matrices = np.zeros((self.num_agents, self.n_rollout_threads, self.num_agents))
        for agent_id in range(self.num_agents):
            weight_matrices[agent_id] = self.buffer[agent_id].Metropolis_weights[-1]
        weight_matrices = np.transpose(weight_matrices, (1,0,2))

        # Run consensus iterations
        for iteration in range(max_iterations):
            for env_idx in range(self.n_rollout_threads):
                weights = weight_matrices[env_idx]  # (n_UAVs, n_UAVs)
                # 使用矩阵乘法执行共识迭代
                thread_estimates = consensus_estimates[env_idx]  # (n_UAVs, episode_length, 1)
                # 使用矩阵乘法进行更新 (n_UAVs, n_UAVs) @ (n_UAVs, episode_length, 1)
                # 重塑为2D进行矩阵乘法，然后恢复原始形状
                reshaped_estimates = thread_estimates.reshape(self.num_agents, -1)  # (n_UAVs, episode_length*1)
                updated_estimates = weights @ reshaped_estimates  # (n_UAVs, episode_length*1)
                thread_estimates = updated_estimates.reshape(self.num_agents, self.episode_length, 1)
                consensus_estimates[env_idx] = thread_estimates

        # 转置回原始格式: (n_UAVs, episode_length, n_rollout_threads, 1)
        result = np.transpose(consensus_estimates, (1, 2, 0, 3))
        return result
