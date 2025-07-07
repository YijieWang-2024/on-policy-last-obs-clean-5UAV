import time
import wandb
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.utils.util import get_shape_from_obs_space

def _t2n(x):
    return x.detach().cpu().numpy()

class MECRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(MECRunner, self).__init__(config)
        self.normer = Normer(args=self.all_args, obs_space=get_shape_from_obs_space(self.envs.observation_space[0]), states_space=get_shape_from_obs_space(self.envs.share_observation_space[0]))
        if self.model_dir is not None:
            self.restore(self.model_dir)
        self.average_local_advantage = config['all_args'].average_local_advantage

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions    (在collect里边，调用env处理了动作。输出处理后的动作，和对应的log_p)
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions, Metropolis_weights, attention_active_mask = self.envs.step(actions)
                obs = self.normer._obfilt(obs)
                share_obs = self.normer._statefilt(share_obs)
                rewards = self.normer._rewsfilt(rewards, dones)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, Metropolis_weights, attention_active_mask
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
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
                    # for agent_id in range(self.num_agents):
                    train_infos.update({'cumulative_reward': np.mean(np.mean([info['cumulative_reward'] for info in infos], axis=0)).round(1)})
                    train_infos.update({'system_performance': np.mean(np.mean([info['system_performance'] for info in infos], axis=0)).round(1)})
                    train_infos.update({'system_performance_true_all_GUs': np.mean(np.mean([info['system_performance_true_all_GUs'] for info in infos], axis=0)).round(1)})
                    train_infos.update({'system_performance_individual': np.mean([info['system_performance_individual'] for info in infos], axis=0)[0].round(1)})
                    train_infos.update({'cumulative_individual_reward': np.mean([info['cumulative_individual_reward'] for info in infos], axis=0)[0].round(1)})
                    train_infos.update({'n_GUs_by_coverd': np.mean([info['n_GUs_by_coverd'] for info in infos]).round(1)})
                    train_infos.update({'cumulative_individual_reward': np.mean([info['cumulative_individual_reward'] for info in infos], axis=0)[0].round(5)})

                    print('cumulative_reward is ', np.mean([info['cumulative_reward'] for info in infos], axis=0).round(1))
                    print('system_performance is ', np.mean([info['system_performance'] for info in infos], axis=0).round(1))
                    print('system_performance_true_all_GUs is ', np.mean([info['system_performance_true_all_GUs'] for info in infos], axis=0).round(1))
                    print('system_performance_individual is ', np.mean([info['system_performance_individual'] for info in infos], axis=0).round(1))
                    print('cumulative_individual_reward is ', np.mean([info['cumulative_individual_reward'] for info in infos], axis=0).round(1))
                    print('value_loss is {}, policy_loss is {}, dist_entropy is {}, actor_grad_norm is {}, critic_grad_norm is {}, ratio is {}'.format(train_infos['value_loss'],
                                                                                                                                                    train_infos['policy_loss'],
                                                                                                                                                    train_infos['dist_entropy'],
                                                                                                                                                    train_infos['actor_grad_norm'],
                                                                                                                                                    train_infos['critic_grad_norm'],
                                                                                                                                                    train_infos['ratio']))

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions, Metropolis_weights, attention_active_mask = self.envs.reset()
        obs = self.normer._obfilt(obs)
        share_obs = self.normer._statefilt(share_obs)

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()
        self.buffer.Metropolis_weights[0] = Metropolis_weights.copy()
        self.buffer.attention_active_mask[0] = attention_active_mask.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # 这里的action_log_prob不能用来计算policy_loss，因为这个还没有经过处理，不是真正用到环境的动作。
        _, action, _, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]),
                                              attention_active_mask = np.concatenate(self.buffer.attention_active_mask[step]))
        # 处理动作！
        action = np.array(np.split(_t2n(action), self.n_rollout_threads))
        processed_action = self.envs.process_actions(action)
        # 下边evaluate_actions输出的value应该是和上边的value是一样的。用哪个都可以，value只和obs有关，和动作无关。
        value, action_log_prob, _ = self.trainer.policy.evaluate_actions(np.concatenate(self.buffer.share_obs[step]),
                                                                      np.concatenate(self.buffer.obs[step]),
                                                                      np.concatenate(self.buffer.rnn_states[step]),
                                                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                                                      np.concatenate(processed_action),
                                                                      np.concatenate(self.buffer.masks[step]),
                                                                      np.concatenate(self.buffer.available_actions[step]),
                                                                         attention_active_mask = self.buffer.attention_active_mask[step])
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        processed_actions = processed_action
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, processed_actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, Metropolis_weights, attention_active_mask = data

        dones_env = np.all(dones, axis=1)
        # rnn狀態重置
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states.shape[3:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        # 數據有沒有用的mask。因爲有的episode會提前結束，每個episode長度不同。所以有這個來mask不同長度。是用来重置rnn_state的
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # # 用来计算策略熵和动作logp时，求均值的范围，avtive里边求。
        # active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # # 用来表示由于截断，导致的环境结束。  mask只表示环境结束，这个相当于更加细分，有了由于truncated导致的终止。
        # bad_masks = np.ones_like(masks)
        # # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        # self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
        #                    actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks,
                           available_actions=available_actions, Metropolis_weights=Metropolis_weights,
                           attention_active_mask=attention_active_mask)

    def train(self):
        if self.average_local_advantage:
            all_advantages = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads), dtype=np.float32)
            all_Metropolis_weights = np.zeros((self.num_agents, self.episode_length, self.n_rollout_threads, self.num_agents), dtype=np.float32)
            for agent_id in range(self.num_agents):
                advantages_i = self.buffer.returns[:-1, :, agent_id] - self.buffer.value_preds[:-1, :, agent_id]

                all_advantages[agent_id] = advantages_i.squeeze(-1)
                all_Metropolis_weights[agent_id] = self.buffer.Metropolis_weights[:-1, :, agent_id]
            updated_advantages = all_advantages.copy()
            n_iterations = self.all_args.n_iterations
            for _ in range(n_iterations):
                # This single line replaces the triple nested loop
                # 'istj,jst->ist' means: sum over j dimension (agent weights)
                updated_advantages = np.einsum('istj,jst->ist', all_Metropolis_weights, updated_advantages)

            # updated_advantages = all_advantages - updated_advantages

            self.buffer.advantages = updated_advantages.transpose(1, 0, 2)
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def log_train(self, train_infos, total_num_steps):
        # train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
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

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(self.save_dir, episode)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

        # 保存 normer 对象
        normer_path = str(self.save_dir) + "/normer.pkl"
        self.normer.save(normer_path)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)

        # 恢复 normer 对象
        normer_path = str(self.model_dir) + "/normer.pkl"
        self.normer.load(normer_path)
