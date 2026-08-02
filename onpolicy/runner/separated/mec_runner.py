import time
# import wandb
import numpy as np
import torch
from onpolicy.runner.separated.base_runner import Runner
from onpolicy.envs.mec.vec_normalize import Normer, normalize_batch
from onpolicy.utils.util import get_shape_from_obs_space
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

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
        self.advantage_mode = getattr(config['all_args'], "advantage_mode", "default")
        self.consensus_alpha = getattr(config['all_args'], "consensus_alpha", 0.5)
        if self.whether_average_network_parameters:
            self.average_network_parameters()
        # self.n_UAVs = config['all_args'].n_UAVs
        self.uav_positions = np.zeros((self.n_rollout_threads, self.num_agents, 2))
        self.neighbor_distance = config['all_args'].neighbor_distance
        # 测评、记录效果的。 把训练的注释掉，这个打开。
        self.system_gain = np.zeros(15)
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
                normalize_batch(self.normer, obs, share_obs, rewards, dones)
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
            # self.system_gain[episode] = np.mean(np.mean([info['system_performance_true_all_GUs'] for info in infos], axis=0)).round(5)
            # if episode == 14:
            #     # np.savetxt('./plot_data/system_gain_7UAVs.txt', self.system_gain)
            #     np.savetxt('./plot_data/system_gain_run316.txt', self.system_gain)
            #     print(np.mean(self.system_gain))
            #     print('over')

            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     for i, info in enumerate(infos):
            #         self.uav_positions[i] = info['uav_positions']
            #     np.save(str(self.run_dir) + '/uav_positions_' + str(episode) + '.npy', self.uav_positions)
            #     for agent_id in range(self.num_agents):
            #         self.local_advantages[agent_id] = self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1]
            #     np.save(str(self.run_dir) + '/local_advantages_' + str(episode) + '.npy', self.local_advantages)
            for i, info in enumerate(infos):
                self.uav_positions[i] = info['uav_positions']

            # compute return and update network
            self.compute()
            train_infos = self.train()

            if self.whether_average_network_parameters and (episode % self.average_network_parameters_interval == 0):
                self.average_network_parameters()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

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
                    curriculum_infos = {}
                    if infos and 'curriculum_random_reset' in infos[0]:
                        random_resets = np.asarray([
                            bool(info['curriculum_random_reset']) for info in infos
                        ])
                        for reset_type, reset_mask in (
                            ('random_reset', random_resets),
                            ('fixed_reset', ~random_resets),
                        ):
                            curriculum_infos[f'curriculum_{reset_type}_samples'] = int(
                                np.sum(reset_mask)
                            )
                            if np.any(reset_mask):
                                curriculum_infos[
                                    f'system_performance_true_all_GUs_{reset_type}'
                                ] = np.mean([
                                    info['system_performance_true_all_GUs']
                                    for info, selected in zip(infos, reset_mask)
                                    if selected
                                ]).round(5)
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update(curriculum_infos)
                        train_infos[agent_id].update({'cumulative_reward': np.mean(np.mean([info['cumulative_reward'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'cumulative_individual_reward_wo_cover': np.mean([info['cumulative_individual_reward_wo_cover'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'cumulative_reward_wo_cover': np.mean(np.mean([info['cumulative_reward_wo_cover'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'system_performance': np.mean(np.mean([info['system_performance'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'system_performance_true_all_GUs': np.mean(np.mean([info['system_performance_true_all_GUs'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'system_performance_equivalent_full_GUs': np.mean(np.mean([info['system_performance_equivalent_full_GUs'] for info in infos], axis=0)).round(5)})
                        train_infos[agent_id].update({'system_performance_individual': np.mean([info['system_performance_individual'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'cumulative_individual_reward': np.mean([info['cumulative_individual_reward'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'delay_true_all_GUs': np.mean([info['delay_true_all_GUs'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'delay_true_coverd_GUs': np.mean([info['delay_true_coverd_GUs'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'energy_true_all_GUs': np.mean([info['energy_true_all_GUs'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'energy_all_GUs_UAVs': np.mean([info['energy_all_GUs_UAVs'] for info in infos], axis=0)[agent_id].round(5)})
                        train_infos[agent_id].update({'n_GUs_by_coverd': np.mean([info['n_GUs_by_coverd'] for info in infos]).round(5)})
                        train_infos[agent_id].update({'complete_task_ratio': np.mean([info['complete_task_ratio'] for info in infos]).round(5)})
                        for metric in (
                            'candidate_md_arrivals',
                            'admitted_md_arrivals',
                            'candidate_md_arrivals_lower_left',
                            'candidate_md_arrivals_upper_right',
                            'admitted_md_arrivals_lower_left',
                            'admitted_md_arrivals_upper_right',
                            'md_admission_ratio',
                            'md_admission_ratio_lower_left',
                            'md_admission_ratio_upper_right',
                            'expired_md_accesses',
                            'coverage_departed_md_accesses',
                            'average_active_mds',
                            'average_active_mds_second_half',
                            'flight_proposal_norm_mean',
                            'flight_disk_projection_ratio',
                            'curriculum_random_reset',
                            'curriculum_random_probability',
                        ):
                            if metric in infos[0]:
                                train_infos[agent_id][metric] = np.mean(
                                    [info[metric] for info in infos]
                                ).round(5)

                # print(episode)
                print('cumulative_reward is ', np.mean([info['cumulative_reward'] for info in infos], axis=0).round(5))
                print('system_performance is ', np.mean([info['system_performance'] for info in infos], axis=0).round(5))
                print('system_performance_true_all_GUs is ', np.mean([info['system_performance_true_all_GUs'] for info in infos], axis=0).round(5))
                print('system_performance_equivalent_full_GUs is ', np.mean([info['system_performance_equivalent_full_GUs'] for info in infos], axis=0).round(5))
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
        normalize_batch(self.normer, obs, share_obs)
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

        # A UAV with exactly one local user is still a live agent: its flight,
        # association and critic samples remain valid. Dirichlet degeneracy is
        # handled inside the resource head instead of masking the whole agent.

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
        consensus_infos = {}
        if self.advantage_mode != "default":
            local_advantage = self.collect_local_advantages()

            if self.advantage_mode == "local":
                training_advantage = local_advantage
            elif self.advantage_mode == "legacy_noise":
                exact_mean = np.mean(local_advantage, axis=0, keepdims=True)
                raw_consensus = self.run_consensus_algorithm(
                    local_advantage, self.all_args.n_iterations
                )
                consensus_residual = self.normalized_consensus_residual(
                    local_advantage, raw_consensus
                )
                std_advantage = np.std(local_advantage, axis=0, keepdims=True)
                training_advantage = local_advantage + exact_mean
                training_advantage += (
                    consensus_residual
                    * 0.12
                    * std_advantage
                    * np.random.randn(*local_advantage.shape)
                )
                consensus_infos = {
                    'consensus_residual_mean': float(np.mean(consensus_residual)),
                    'consensus_residual_max': float(np.max(consensus_residual)),
                }
            else:
                consensus_advantage, graph_stats = self.run_consensus_algorithm(
                    local_advantage,
                    self.all_args.n_iterations,
                    scale_by_component=True,
                    return_graph_stats=True,
                )
                if self.advantage_mode == "pure_consensus":
                    training_advantage = consensus_advantage
                elif self.advantage_mode == "mixed_consensus":
                    training_advantage = (
                        (1.0 - self.consensus_alpha) * local_advantage
                        + self.consensus_alpha * consensus_advantage
                    )
                else:
                    raise ValueError(
                        f"unsupported advantage_mode: {self.advantage_mode}"
                    )
                consensus_infos = self.consensus_diagnostics(
                    local_advantage,
                    consensus_advantage,
                    training_advantage,
                    graph_stats,
                )

            for agent_id in range(self.num_agents):
                self.buffer[agent_id].advantages = training_advantage[agent_id].copy()

        elif self.average_local_advantage_timely and self.average_local_advantage:
            if self.whether_local_add_direct_ave_adv:
                # 直接local+True_mean
                local_advantage = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages = self.buffer[agent_id].returns[:-1] - self.buffer[agent_id].value_preds[:-1]
                    local_advantage[agent_id] = self.buffer[agent_id].advantages.copy()
                # # L + Cluster_Mean
                # cluster_mean_advantage = self.compute_average_advantages_directed(self.uav_positions, local_advantage, self.neighbor_distance)
                # for agent_id in range(self.num_agents):
                #     self.buffer[agent_id].advantages += cluster_mean_advantage[agent_id]
                # L+M + /(Noi)
                mean_advantage = np.mean(local_advantage, axis=0)
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages += mean_advantage
                cluster_mean_advantage = self.run_consensus_algorithm(
                    local_advantage, self.all_args.n_iterations
                )
                consensus_residual = self.normalized_consensus_residual(
                    local_advantage, cluster_mean_advantage
                )
                consensus_infos = {
                    'consensus_residual_mean': float(np.mean(consensus_residual)),
                    'consensus_residual_max': float(np.max(consensus_residual)),
                }
                std_advantage = np.std(local_advantage, axis=0)
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].advantages += (
                        consensus_residual[0]
                        * 0.12
                        * std_advantage
                        * np.random.randn(*self.buffer[agent_id].advantages.shape)
                    )
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
            train_info.update(consensus_infos)
            if getattr(self.all_args, "cartesian_flight", False):
                actor_act = self.trainer[agent_id].policy.actor.act
                flight_head = actor_act.action_out if actor_act.mujoco_box else actor_act.action_outs[0]
                flight_std = torch.exp(flight_head.logstd._bias.detach()).cpu().numpy().reshape(-1)
                train_info["flight_std_x"] = float(flight_std[0])
                train_info["flight_std_y"] = float(flight_std[1])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def average_network_parameters(self):
        """
        Average network parameters across all trainers.
        Each trainer in self.trainer is an instance of R_MAPPO.
        """
        # def _reset_momentum(opt):
        #     if opt is None: return
        #     for st in opt.state.values():
        #         if 'exp_avg' in st: st['exp_avg'].zero_()
        #         if 'exp_avg_sq' in st: st['exp_avg_sq'].zero_()
        # tau = 0.05
        # if len(self.trainer) <= 1: return
        # # 1) 计算全局平均 actor
        # ref = self.trainer[0].policy.actor.state_dict()
        # avg = {k: sum(tr.policy.actor.state_dict()[k] for tr in self.trainer) / len(self.trainer) for k in ref}
        # # 2) Polyak 软同步 + 同步旧策略 + 清零动量
        # for tr in self.trainer:
        #     sd = tr.policy.actor.state_dict()
        #     for k in sd: sd[k].lerp_(avg[k], tau)
        #     tr.policy.actor.load_state_dict(sd)
        #     if hasattr(tr.policy, "actor_old"):
        #         tr.policy.actor_old.load_state_dict(sd)
        #     _reset_momentum(getattr(tr.policy, "actor_optimizer", None))
        # # critic 不平均；若要平均，用很小 tau 对 critic 做同样 lerp，并重置其动量

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
                if (agent_id == 0 or k.startswith('flight_std_') or
                        k in ['cumulative_individual_reward', 'system_performance_individual', 'cumulative_individual_reward_wo_cover']):
                    agent_k = "agent%i/" % agent_id + k
                    if self.use_wandb:
                        wandb.log({agent_k: v}, step=total_num_steps)
                    else:
                        self.writter.add_scalar(agent_k, v, total_num_steps)
    
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

    def save(self, episode):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

            # 保存 normer 对象
            normer_path = str(self.save_dir) + "/normer"+ str(agent_id) +".pkl"
            self.normer[agent_id].save(normer_path)
        # if episode % 100 == 0:
        #     for agent_id in range(self.num_agents):
        #         policy_actor = self.trainer[agent_id].policy.actor
        #         torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) +"_"+str(episode)+ ".pt")
        #         policy_critic = self.trainer[agent_id].policy.critic
        #         torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) +"_"+str(episode)+ ".pt")
        #         # 保存 normer 对象
        #         normer_path = str(self.save_dir) + "/normer" + str(agent_id) +"_"+str(episode)+ ".pkl"
        #         self.normer[agent_id].save(normer_path)
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

    def collect_local_advantages(self):
        """Collect per-UAV GAE advantages in a common array."""
        local_advantages = np.zeros(
            (self.num_agents, self.episode_length, self.n_rollout_threads, 1),
            dtype=np.float32,
        )
        for agent_id in range(self.num_agents):
            value_preds = self.buffer[agent_id].value_preds[:-1]
            trainer = self.trainer[agent_id]
            if trainer._use_popart or trainer._use_valuenorm:
                value_preds = trainer.value_normalizer.denormalize(value_preds)
            local_advantages[agent_id] = (
                self.buffer[agent_id].returns[:-1] - value_preds
            )
        return local_advantages

    @staticmethod
    def _safe_correlation(left, right, epsilon=1e-8):
        left = np.asarray(left).reshape(-1)
        right = np.asarray(right).reshape(-1)
        finite = np.isfinite(left) & np.isfinite(right)
        if np.count_nonzero(finite) < 2:
            return 0.0
        left = left[finite]
        right = right[finite]
        if np.std(left) <= epsilon or np.std(right) <= epsilon:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    def consensus_diagnostics(
        self, local_advantages, consensus_advantages, training_advantages,
        graph_stats, epsilon=1e-8,
    ):
        """Return low-cost scalar diagnostics without running eval episodes."""
        exact_mean = np.mean(local_advantages, axis=0, keepdims=True)
        initial_error = np.linalg.norm(local_advantages - exact_mean, axis=0)
        global_error = np.linalg.norm(consensus_advantages - exact_mean, axis=0)
        relative_global_error = np.divide(
            global_error,
            initial_error,
            out=np.zeros_like(global_error),
            where=initial_error > epsilon,
        )
        local_std = float(np.std(local_advantages))
        consensus_std = float(np.std(consensus_advantages))
        return {
            'consensus_global_error_mean': float(np.mean(relative_global_error)),
            'consensus_global_error_max': float(np.max(relative_global_error)),
            'consensus_component_count_mean': float(np.mean(graph_stats['component_counts'])),
            'consensus_component_size_mean': float(np.mean(graph_stats['component_sizes'])),
            'consensus_graph_connected_fraction': float(graph_stats['connected_fraction']),
            'consensus_local_corr': self._safe_correlation(
                local_advantages, consensus_advantages
            ),
            'training_local_corr': self._safe_correlation(
                local_advantages, training_advantages
            ),
            'consensus_to_local_std_ratio': consensus_std / (local_std + epsilon),
        }

    def run_consensus_algorithm(
        self, local_observations, max_iterations, scale_by_component=False,
        return_graph_stats=False,
    ):
        """Run finite consensus on the rollout-ending Metropolis graph.

        With ``scale_by_component``, each component estimate is multiplied by
        ``component_size / num_agents``.  Consequently R=0 produces A_i/M,
        while a connected converged graph produces the exact global mean.
        """
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        estimates = np.transpose(local_observations, (2, 0, 1, 3))
        estimate_shape = estimates.shape
        estimates = estimates.reshape(self.n_rollout_threads, self.num_agents, -1)
        weights = np.stack(
            [self.buffer[i].Metropolis_weights[-1] for i in range(self.num_agents)],
            axis=1,
        ).astype(local_observations.dtype, copy=False)

        for _ in range(max_iterations):
            estimates = weights @ estimates

        component_counts = np.zeros(self.n_rollout_threads, dtype=np.int32)
        component_sizes = np.ones(
            (self.n_rollout_threads, self.num_agents), dtype=local_observations.dtype
        )
        for env_id in range(self.n_rollout_threads):
            component_count, labels = connected_components(
                weights[env_id] > 0.0, directed=False, return_labels=True
            )
            component_counts[env_id] = component_count
            sizes = np.bincount(labels, minlength=component_count)
            component_sizes[env_id] = sizes[labels]

        if scale_by_component:
            estimates *= component_sizes[..., np.newaxis] / float(self.num_agents)

        estimates = estimates.reshape(estimate_shape)
        estimates = np.transpose(estimates, (1, 2, 0, 3))
        if not return_graph_stats:
            return estimates
        graph_stats = {
            'component_counts': component_counts,
            'component_sizes': component_sizes,
            'connected_fraction': np.mean(component_counts == 1),
        }
        return estimates, graph_stats

    @staticmethod
    def normalized_consensus_residual(
        local_advantages, consensus_advantages, epsilon=1e-8
    ):
        """Return one bounded L2 consensus-error ratio per rollout sample.

        The norm is taken across the UAV axis.  Consequently every UAV in a
        given (time, environment) sample receives the same topology-dependent
        noise magnitude, while retaining an independent Gaussian draw.
        """
        exact_mean = np.mean(local_advantages, axis=0, keepdims=True)
        initial_disagreement = local_advantages - exact_mean
        residual_disagreement = consensus_advantages - exact_mean
        denominator = np.linalg.norm(initial_disagreement, axis=0, keepdims=True)
        numerator = np.linalg.norm(residual_disagreement, axis=0, keepdims=True)
        residual = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > epsilon,
        )
        return np.clip(residual, 0.0, 1.0)

    def compute_average_advantages_directed(self, uav_positions, local_advantages, neighbor_distance):
        """
        Compute average advantages for UAVs based on connectivity.
        Parameters:
            uav_positions : Array of UAV positions, shape (n_rollout_threads, num_agents, 2)
            local_advantages : Array of local advantage estimates, shape (num_agents, episode_length, n_rollout_threads, 1)
            neighbor_distance : Distance threshold for UAVs to be considered neighbors
        Returns:
            Array of average advantages, shape (num_agents, episode_length, n_rollout_threads, 1)
        """
        # Reshape local_advantages for more efficient processing
        # From (num_agents, episode_length, n_rollout_threads, 1)
        # to (n_rollout_threads, episode_length, num_agents)
        reshaped_advantages = np.transpose(local_advantages, (2, 1, 0, 3)).squeeze(-1)

        # Initialize output array
        average_advantages = np.zeros_like(reshaped_advantages)

        # Process all environments at once using adjacency matrices
        for env_idx in range(self.n_rollout_threads):
            # Calculate adjacency matrix - UAVs are neighbors if distance <= neighbor_distance
            positions = uav_positions[env_idx]
            distances = cdist(positions, positions)  # More efficient than pdist + squareform
            adjacency_matrix = distances <= neighbor_distance

            # Find connected components
            n_components, labels = connected_components(adjacency_matrix, directed=False)

            # Pre-compute component indices for all agents
            component_indices = [np.where(labels == i)[0] for i in range(n_components)]

            # Create a mapping from agent to its component average for all time steps
            for comp_idx, agent_indices in enumerate(component_indices):
                cluster_advantages = reshaped_advantages[env_idx, :, agent_indices]
                assert cluster_advantages.shape[1] == self.episode_length
                # Calculate average advantages for all time steps at once
                component_mean = np.mean(cluster_advantages, axis=0, keepdims=True)
                # Assign to all agents in this component (broadcasting over time steps)
                average_advantages[env_idx, :, agent_indices] = component_mean

        # Reshape back to original format (num_agents, episode_length, n_rollout_threads, 1)
        return np.transpose(average_advantages, (2, 1, 0))[:, :, :, np.newaxis]
