import copy
import numpy as np
import gym
import gym.spaces as spaces
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
import time

# 仿真参数
# 参数来自Joint Task Offloading, Resource Allocation, and Trajectory Design for Multi-UAV Cooperative Edge Computing With Task Priority
# 通信参数
p_t = 1  # 地面用户的发射功率，W
N_0 = 1e-16      # 系统噪声功率谱密度，W/Hz，也就是-130dBm/Hz。这个来自（Multi-Agent Deep Reinforcement Learning for Decentralized Multi-UAV Mobile Edge Computing Networks）
a = 9.61
b = 0.16
eta_LoS = 1     # dB，LoS信道额外的损失
eta_NLoS = 20   # dB，NLoS信道额外的损失
f_c = 2 * 10 ** 9     # 2GHz，无人机服务器和地面用户之间的信号频率

# 参数来自Evolutionary Multi-Objective Reinforcement Learning Based Trajectory Control and Task Offloading in UAV-Assisted Mobile Edge Computing
P1 = 79.86  # Blade profile power
P2 = 88.63  # Induced power
U_tip = 120   # (m/s) Tip speed of rotor blade
v0 = 4.03  # Meanrotor induced velocity in hover
d0 = 0.6  # 无量纲，Fuselage drag ratio
rho = 1.225  # (kg/m^3) Air density
g = 0.05  # 无量纲，Rotor solidity
A = 0.503  # (m^2)Rotor disc area
# 计算消耗能量参数来自Task Offloading and Resource Allocation Strategies Among Multiple Edge Servers
k_local = 1e-27  # J/(Hz^2 * cycle)Placeholder value
k_server = 1e-29  # J/(Hz^2 * cycle)Placeholder value

class MEC(gym.Env):
    def __init__(self, args=None):
        assert args is not None
        self.ob_state_with_timestep = args.ob_state_with_timestep
        self.n_UAVs = args.n_UAVs
        self.n_GUs = args.n_GUs
        self.max_GUs_in_range = args.max_GUs_in_range   # 无人机的服务范围内距离由近到远，保留信息的最大用户数目。
        self.max_UAVs_in_neighbor = args.max_UAVs_in_neighbor  # 只用到自己的观测s_{i,t}中的其他无人机数目。无人机的邻居范围内距离由近到远，保留信息的最大无人机数目。
        self.neighbor_distance = args.neighbor_distance  # 无人机之间定义为邻居的距离。 之前为d_cov*2=240。我的last-obs设为覆盖范围内的无人机数目。因此设置为120
        self.d_optimal = args.d_optimal
        self.perform_with_local_state = args.perform_with_local_state
        self.state_is_k_hops = args.state_is_k_hops
        assert not (self.perform_with_local_state and self.state_is_k_hops), "不能同时使用和obs一样的local_state，和k_hops state"
        # self.concat_neighbor_obs = args.concat_neighbor_obs
        self.max_UAVs_obs_concat = args.max_UAVs_obs_concat # 这个是s_{M_i^k}跳要拼接的无人机s_{i,t}的数目。 放在之前就是构造一跳观测的s_{M_i^1}拼接数目。
        # self.max_UAVs_obs_concat = args.n_UAVs # 这个是s_{M_i^k}跳要拼接的无人机s_{i,t}的数目。 放在之前就是构造一跳观测的s_{M_i^1}拼接数目。
        self.local_reward = args.local_reward
        self.n_agents = self.n_UAVs                 # 100架飞机。
        self.x_max = args.x_max                  # 15km*15km的范围
        self.alpha_r = args.alpha_r
        self.beta_r = args.beta_r
        self.gamma_r = args.gamma_r
        self.delta_r = args.delta_r
        self.lambda_r = args.lambda_r
        self.epsilon_r = args.epsilon_r
        self.mu_r = args.mu_r
        self.q1 = args.q1
        self.q2 = args.q2
        self.q3 = args.q3
        self.q4 = args.q4
        self.q5 = args.q5

        self.B = args.B
        self.H_UAV = args.H_UAV
        self.H_GU = args.H_GU
        self.F_m = args.F_m
        self.F_n = args.F_n
        self.D_min = args.D_min
        self.D_max = args.D_max
        self.C_min = args.C_min
        self.C_max = args.C_max
        self.delay_min = args.delay_min
        self.delay_max = args.delay_max
        self.Dis_min = args.Dis_min
        self.Cover_R = args.Cover_R
        self.Delta_t = args.Delta_t
        self.MAX_SIMULATION_TIME = args.episode_length
        self.w1 = args.w1
        self.w2 = args.w2
        self.p3 = args.p3
        self.v_max = args.v_max
        self.discrete_associate = args.discrete_associate
        self.continuous_associate = args.continuous_associate
        self.nearest_associate = args.nearest_associate
        assert self.discrete_associate + self.continuous_associate + self.nearest_associate == 1
        self.not_process_action = args.not_process_action
        self.fix_uav_pos = args.fix_uav_pos
        self.ave_resource = args.ave_resource
        if self.ave_resource:
            assert self.continuous_associate and not self.fix_uav_pos, "只写了在无人机飞行且连续associate条件下的平均分配资源"
        self.mean_velocity = args.mean_velocity
        # Define the UAV flight direction and distance action space (continuous)

        if self.perform_with_local_state:
            # [总无人机数目、总用户数目、区域总长度、
            # 自身无人机位置；
            # 与自身距离小于neighbor_distance的邻居无人机的数目；
            # 距离由近到远的前max_UAVs_in_neighbor架邻居无人机的位置；
            # 与自身距离小于Cover_R的地面用户的数目；
            # 距离由近到远的前self.max_GUs_in_range个地面用户的位置、信道增益和计算任务的信息(已经未被服务的时间+id。所以从7变成9)]
            self.GUs_in_action_dim = self.max_GUs_in_range
            self.obs_dim = 3+2 + 1 + 2*self.max_UAVs_in_neighbor +1 +  9*self.max_GUs_in_range
            self.state_dim = 3+2 + 1 + 2*self.max_UAVs_in_neighbor +1 +  9*self.max_GUs_in_range

            # # 不要邻居无人机的位置。
            # self.obs_dim = 3 + 2 + 1 + 7 * self.max_GUs_in_range
            # self.state_dim = 3 + 2 + 1 + 7 * self.max_GUs_in_range
        elif self.state_is_k_hops:  # last-obs的k跳。自己的s_{i,t}是包括覆盖范围内的无人机的。
            self.GUs_in_action_dim = self.max_GUs_in_range
            # 包括覆盖范围内d_cov的无人机信息。
            self.obs_dim = 3 + 2 + 1+2*self.max_UAVs_in_neighbor + 1+ 9*self.max_GUs_in_range
            self.state_dim = 3 + 2 + 1+2*self.max_UAVs_in_neighbor + 1+ 9*self.max_GUs_in_range
        else:
            # self.GUs_in_action_dim = self.n_GUs
            self.GUs_in_action_dim = self.max_GUs_in_range
            self.obs_dim = 3+2 + 1+2*self.max_UAVs_in_neighbor + 1+ 9*self.max_GUs_in_range   # 局部obs的dim
            # 不要邻居无人机的位置。
            # self.obs_dim = 3+2 + 1 + 7*self.max_GUs_in_range   # 局部obs的dim

            # self.state_dim = 1 + 3 +  2*self.n_UAVs+6*self.n_GUs +1
            # self.state_dim = self.n_UAVs * (self.obs_dim + int(self.ob_state_with_timestep))
            self.state_dim = (self.n_UAVs+1) * (self.obs_dim + int(self.ob_state_with_timestep))
            # self.state_dim = (self.obs_dim + int(self.ob_state_with_timestep)) + 1 + 3 +  2*self.n_UAVs+6*self.n_GUs +1

        if self.ob_state_with_timestep:
            self.obs_dim += 1
        if self.ob_state_with_timestep and (self.perform_with_local_state or self.state_is_k_hops):
            self.state_dim += 1

        # 不管怎么决策，obs是不变了。就是自己的s_{i,t}
        # if self.concat_neighbor_obs:    # obs也拼接？？？
        #     self.obs_dim *= self.max_UAVs_obs_concat
        # if self.concat_neighbor_obs and self.perform_with_local_state:
        #     pass
        # if self.concat_neighbor_obs and self.state_is_k_hops:
        #     self.state_dim *= self.max_UAVs_obs_concat
        # # else，如果都没有，就是mappo形式的state。直接设定好了。

        if self.state_is_k_hops:
            # if self.perform_with_local_state直接就不用管state了。
            # if self.state_is_k_hops:就是下边的乘一下最大数目，然后用atten。
            # else：其实就是mappo，state_dim也提前设定好了
            self.state_dim *= self.max_UAVs_obs_concat

        if self.fix_uav_pos:
            pass
        else:
            self.flight_action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)  # 方向和速度都是0-1之间的数
        if self.discrete_associate:
            self.task_offloading_space = spaces.MultiBinary(self.GUs_in_action_dim)
        elif self.continuous_associate:
            self.task_offloading_space = spaces.Box(low=0.0, high=1.0, shape=(self.GUs_in_action_dim,), dtype=np.float32)
        else:
            pass
        if self.ave_resource:
            pass
        else:
            self.bandwidth_allocation_space = spaces.Box(low=0.0, high=1.0, shape=(self.GUs_in_action_dim,), dtype=np.float32)
            self.computation_allocation_space = spaces.Box(low=0.0, high=1.0, shape=(self.GUs_in_action_dim,), dtype=np.float32)
        # 这里的动作空间设置的n_GUs，也影响这process_actions里边的actions的处理，所以要注意。
        # 组成完整动作空间
        if self.fix_uav_pos and self.nearest_associate:
            self.action_space = spaces.Tuple((self.bandwidth_allocation_space, self.computation_allocation_space))
        elif self.fix_uav_pos and not self.nearest_associate:
            self.action_space = spaces.Tuple((self.task_offloading_space, self.bandwidth_allocation_space,self.computation_allocation_space))
        elif not self.fix_uav_pos and self.nearest_associate:
            self.action_space = spaces.Tuple((self.flight_action_space, self.bandwidth_allocation_space, self.computation_allocation_space))
        elif not self.fix_uav_pos and self.ave_resource:
            assert self.continuous_associate
            self.action_space = spaces.Tuple((self.flight_action_space, self.task_offloading_space))
        else:
            self.action_space = spaces.Tuple((self.flight_action_space, self.task_offloading_space, self.bandwidth_allocation_space,self.computation_allocation_space))
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(self.obs_dim,), dtype=np.float32)
        self.state_space = spaces.Box(low=0.0, high=0.0, shape=(self.state_dim,), dtype=np.float32)

        self.state = None
        self.obs = None
        self.avail_actions = None
        self.time_step = None
        self.attention_active_mask = np.zeros((self.n_UAVs, self.max_UAVs_obs_concat), dtype=np.float32)

        # Initialize UAV positions randomly within the area
        self.uav_positions = np.random.uniform(0, self.x_max, (self.n_UAVs, 2))
        self.uav_positions = np.hstack((self.uav_positions, self.H_UAV * np.ones((self.n_UAVs, 1))))

        # Initialize ground user positions with fixed height of 1m
        self.gu_positions = np.random.uniform(0, self.x_max, (self.n_GUs, 2))
        self.gu_positions = np.hstack((self.gu_positions, self.H_GU * np.ones((self.n_GUs, 1))))
        # Gauss-Markov Model parameters
        # self.alpha_gaussian = 0.95
        self.alpha_gaussian = 0.7
        self.std_dev_gaussian = 1
        self.gu_velocities = np.random.normal(self.mean_velocity, 0.3*self.std_dev_gaussian, self.n_GUs)
        self.gu_velocities = np.clip(self.gu_velocities, 0.7* self.mean_velocity, 1.3 * self.mean_velocity)
        np.random.seed(0)
        self.gu_directions = np.random.uniform(0, 2 * np.pi, self.n_GUs)
        self.gu_directions_0 = self.gu_directions.copy()

        # Initialize ground user tasks
        self.gu_tasks = self.generate_tasks()
        self.cumulative_reward = np.zeros((self.n_agents,))
        self.system_performance = np.zeros((self.n_agents,))
        self.system_performance_individual = np.zeros((self.n_agents,))
        self.system_performance_true_all_GUs = np.zeros((self.n_agents,))
        self.system_performance_coverd_GUs = np.zeros((self.n_agents,))
        self.delay_true_all_GUs = np.zeros((self.n_agents,))
        self.delay_true_coverd_GUs = np.zeros((self.n_agents,))
        self.energy_true_all_GUs = np.zeros((self.n_agents,))
        self.energy_all_GUs_UAVs = np.zeros((self.n_agents,))
        self.n_GUs_by_coverd = 0
        self.cumulative_individual_reward = np.zeros((self.n_agents,))
        self.uav_energy_consumption = np.zeros((self.n_UAVs,))  # 无人机及其范围内用户每时刻能耗和，再对step求和
        self.user_average_delay = np.zeros((self.n_UAVs,))  # 每架无人机范围内用户平均时延，再对step求平均
        self.n_GUs_per_uav_served = np.zeros((self.n_UAVs,))  # 每架无人机每个时刻服务的用户数目，再对step求平均
        self.nearby_gus_of_uavs = -np.ones((self.n_UAVs, self.n_GUs))   # process_local_actions里用到了（类似transform_uav_actions的代码。需要处理和反处理）。
        # calculate_local_reward里先调用到了transform_uav_actions()。
        # get_local_obs()里边，是重新挨个计算的距离。
        self.average_neighbor_advantage = args.average_neighbor_advantage
        self.Metropolis_weights = None      # Metropolis_weights，用来对邻居的Adv进行加权求和。

    def generate_tasks(self):
        tasks = np.zeros((self.n_GUs, 4))
        tasks[:, 0] = np.random.uniform(self.D_min, self.D_max, self.n_GUs)  # Data size
        tasks[:, 1] = np.random.uniform(self.C_min, self.C_max, self.n_GUs)  # compute Resource demand
        tasks[:, 2] = np.random.uniform(self.delay_min, self.delay_max, self.n_GUs)  # Delay requirement
        tasks[:, 3] = 1 # 几个时刻未被服务了。
        return tasks

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, seed=None, *args, **kwargs):
        self.time_step = 0
        # Initialize UAV positions randomly within the area
        # np.random.seed(0)

        # # Initialize UAVs in a grid pattern starting from (50,50)
        # x_start = 100
        # y_start = 100
        # x_spacing = 100
        # y_spacing = 100
        # x_pos = x_start
        # y_pos = y_start
        # self.uav_positions = np.zeros((self.n_UAVs, 3))
        # for i in range(self.n_UAVs):
        #     self.uav_positions[i, 0] = x_pos
        #     self.uav_positions[i, 1] = y_pos
        #     self.uav_positions[i, 2] = self.H_UAV
        #     # Move to next position
        #     x_pos += x_spacing
        #     # If we reach the edge, move to next row
        #     if x_pos >= self.x_max:
        #         x_pos = x_start
        #         y_pos += y_spacing
        if self.n_UAVs == 10 and self.x_max == 600 and self.n_GUs == 80:
            self.uav_positions = np.array([[120, 120, self.H_UAV], [300,120, self.H_UAV], [480, 120, self.H_UAV],
                                           [120, 300, self.H_UAV], [300, 300, self.H_UAV], [480, 300, self.H_UAV],
                                           [120, 480, self.H_UAV],[300, 480, self.H_UAV], [480, 480, self.H_UAV],
                                           [210,390, self.H_UAV]], dtype=np.float)
        if self.n_UAVs == 4 and self.x_max == 400 and self.n_GUs == 40:
            self.uav_positions = np.array([[25, 25, self.H_UAV], [375, 25, self.H_UAV], [25, 375, self.H_UAV],
                                           [375, 375, self.H_UAV]], dtype=np.float)
        if self.n_UAVs == 9 and self.x_max == 600 and self.n_GUs == 80:
            self.uav_positions = np.array([[50, 500, self.H_UAV], [50, 400, self.H_UAV], [50, 300, self.H_UAV],
                                           [50, 200, self.H_UAV], [50, 100, self.H_UAV], [100, 50, self.H_UAV],
                                           [200, 50, self.H_UAV],
                                           [300, 50, self.H_UAV], [400, 50, self.H_UAV]],
                                          dtype=np.float)

        if self.n_UAVs in [5,10,15,20,25] and self.x_max==1000 and self.n_GUs==220:
            if self.n_UAVs == 5:
                self.uav_positions = np.array([[250, 250, self.H_UAV], [750, 250, self.H_UAV], [500, 500, self.H_UAV],
                                               [250, 750, self.H_UAV], [750, 750, self.H_UAV]], dtype=np.float)
            elif self.n_UAVs == 10:
                self.uav_positions = np.array([[190, 190, self.H_UAV], [500, 190, self.H_UAV], [810, 190, self.H_UAV],
                                               [128, 500, self.H_UAV], [376, 500, self.H_UAV], [628, 500, self.H_UAV], [872, 500, self.H_UAV],
                                               [190, 810, self.H_UAV], [500, 810, self.H_UAV], [810, 810, self.H_UAV]], dtype=np.float)
            elif self.n_UAVs == 15:
                self.uav_positions = np.array(
                    [[128, 128, self.H_UAV], [376, 128, self.H_UAV], [628, 128, self.H_UAV], [872, 128, self.H_UAV],
                     [128, 376, self.H_UAV], [376, 376, self.H_UAV], [628, 376, self.H_UAV], [872, 376, self.H_UAV],
                     [128, 628, self.H_UAV], [376, 628, self.H_UAV], [628, 628, self.H_UAV], [872, 628, self.H_UAV],
                     [190, 872, self.H_UAV],[500, 872, self.H_UAV], [810, 872, self.H_UAV]],
                    dtype=np.float)
            elif self.n_UAVs == 20:
                self.uav_positions = np.array([[100, 128, self.H_UAV], [300, 128, self.H_UAV], [500, 128, self.H_UAV], [700, 128, self.H_UAV], [900, 128, self.H_UAV],
                                               [100, 376, self.H_UAV], [300, 376, self.H_UAV],[500, 376, self.H_UAV],[700, 376, self.H_UAV],[900, 376, self.H_UAV],
                                               [100, 628, self.H_UAV], [300, 628, self.H_UAV], [500, 628, self.H_UAV], [700, 628, self.H_UAV], [900, 628, self.H_UAV],
                                               [100, 872, self.H_UAV], [300, 872, self.H_UAV], [500, 872, self.H_UAV], [700, 872, self.H_UAV], [900, 872, self.H_UAV]],
                                              dtype=np.float)
            elif self.n_UAVs == 25:
                self.uav_positions = np.array(
                    [[100, 100, self.H_UAV], [300, 100, self.H_UAV],[500, 100, self.H_UAV],[700, 100, self.H_UAV],[900, 100, self.H_UAV],
                     [100, 300, self.H_UAV], [300, 300, self.H_UAV], [500, 300, self.H_UAV], [700, 300, self.H_UAV], [900, 300, self.H_UAV],
                     [100, 500, self.H_UAV], [300, 500, self.H_UAV], [500, 500, self.H_UAV], [700, 500, self.H_UAV], [900, 500, self.H_UAV],
                     [100, 700, self.H_UAV], [300, 700, self.H_UAV], [500, 700, self.H_UAV], [700, 700, self.H_UAV], [900, 700, self.H_UAV],
                     [100, 900, self.H_UAV], [300, 900, self.H_UAV], [500, 900, self.H_UAV], [700, 900, self.H_UAV], [900, 900, self.H_UAV]],
                    dtype=np.float)


        # self.uav_positions = np.array([[120, 120, self.H_UAV], [480, 120, self.H_UAV], [120, 480, self.H_UAV],[480, 480, self.H_UAV]], dtype=np.float32)
        # assert self.uav_positions.shape[0] == self.n_UAVs

        # np.random.seed(0)
        # Initialize ground user positions with fixed height of 1m
        self.gu_positions = np.random.uniform(0, self.x_max, (self.n_GUs, 2))
        distances = np.sqrt(np.sum(self.gu_positions ** 2, axis=1))
        sorted_indices = np.argsort(distances)
        self.gu_positions = self.gu_positions[sorted_indices]

        self.gu_positions = np.hstack((self.gu_positions, self.H_GU * np.ones((self.n_GUs, 1))))
        self.gu_velocities = np.random.normal(self.mean_velocity, 0.3*self.std_dev_gaussian, self.n_GUs)
        self.gu_velocities = np.clip(self.gu_velocities, 0.7* self.mean_velocity, 1.3 * self.mean_velocity)
        self.gu_directions = np.random.uniform(0, 2 * np.pi, self.n_GUs)

        # np.random.seed(None)
        # Initialize tasks for ground users
        self.gu_tasks = self.generate_tasks()
        self.nearby_gus_of_uavs = self.get_nearby_users_sorted_all()

        # 在state_k_hops中，带自己的s_{i,t}，总共有max_UAVs_obs_concat个信息。 如果是全局拼接state的话，也不需要mask了。正好。
        self.attention_active_mask = np.zeros((self.n_UAVs, self.max_UAVs_obs_concat), dtype=np.float32)
        local_obs = self.get_local_obs()
        # # 自己的obs直接就不变了。
        self.obs = local_obs

        if self.perform_with_local_state:
            self.state = self.obs
        elif self.state_is_k_hops:
            uav_uav_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.uav_positions[:, :2].T[np.newaxis, :], axis=1)
            single_state_dim = self.state_dim // self.max_UAVs_obs_concat
            final_state = np.zeros((self.n_UAVs, self.state_dim))
            final_state[:, :single_state_dim] = local_obs
            for i in range(self.n_UAVs):
                neighbor_mask = (uav_uav_distances[i] <= self.neighbor_distance) & (np.arange(self.n_UAVs) != i)
                neighbors = np.where(neighbor_mask)[0]
                if len(neighbors) > 0:
                    sorted_neighbors = neighbors[np.argsort(uav_uav_distances[i, neighbors])]
                    closest = sorted_neighbors[:self.max_UAVs_obs_concat - 1]
                    for j, neighbor_idx in enumerate(closest):
                        start_pos = (j + 1) * single_state_dim
                        end_pos = (j + 2) * single_state_dim
                        final_state[i, start_pos:end_pos] = local_obs[neighbor_idx]
                self.attention_active_mask[i, :min(len(neighbors)+1, self.max_UAVs_obs_concat)] = 1
            self.state = final_state
        else:
            # self.state = self.get_state()
            # self.state = np.tile(local_obs.reshape((1, -1)), (self.n_UAVs, 1))
            self.state = np.concatenate((local_obs, np.tile(local_obs.reshape((1, -1)), (self.n_UAVs, 1))), axis=-1)
            # self.state = np.concatenate((local_obs, self.get_state()), axis=-1)

        self.avail_actions = self.get_local_avail_actions()

        if self.average_neighbor_advantage:
            self.Metropolis_weights = self.get_neighbor_weights()
        else:
            self.Metropolis_weights = self.get_Metropolis_weights()
        self.cumulative_reward = np.zeros((self.n_agents,))
        self.system_performance = np.zeros((self.n_agents,))
        self.system_performance_individual = np.zeros((self.n_agents,))
        self.system_performance_true_all_GUs = np.zeros((self.n_agents,))
        self.system_performance_coverd_GUs = np.zeros((self.n_agents,))
        self.delay_true_all_GUs = np.zeros((self.n_agents,))
        self.delay_true_coverd_GUs = np.zeros((self.n_agents,))
        self.energy_true_all_GUs = np.zeros((self.n_agents,))
        self.energy_all_GUs_UAVs = np.zeros((self.n_agents,))
        self.n_GUs_by_coverd = 0
        self.cumulative_individual_reward = np.zeros((self.n_agents,))
        self.uav_energy_consumption = np.zeros((self.n_UAVs,))  # 无人机及其范围内用户每时刻能耗和，再对step求和
        self.user_average_delay = np.zeros((self.n_UAVs,))  # 每架无人机范围内用户平均时延，再对step求平均
        self.n_GUs_per_uav_served = np.zeros((self.n_UAVs,))  # 每架无人机每个时刻服务的用户数目，再对step求平均
        return self.obs, self.state, self.avail_actions, self.Metropolis_weights, self.attention_active_mask

    def process_actions(self, action):
        action = np.clip(action, 0., 1.)
        # Convert action to a list with the specified slices
        if self.fix_uav_pos:
            if self.nearest_associate:
                actions = [
                    action[:, :self.n_GUs],
                    action[:, self.n_GUs:]
                ]
            else:
                actions = [
                    action[:, :self.n_GUs],
                    action[:, self.n_GUs:2 * self.n_GUs],
                    action[:, 2 * self.n_GUs:]
                ]
        else:
            if self.nearest_associate:
                actions = [
                    action[:, :2],
                    action[:, 2:2 + self.n_GUs],
                    action[:, 2 + self.n_GUs:]
                ]
            else:
                if self.ave_resource:
                    actions = [
                        action[:, :2],
                        action[:, 2:2 + self.n_GUs]
                    ]
                else:
                    actions = [
                        action[:, :2],
                        action[:, 2:2 + self.n_GUs],
                        action[:, 2 + self.n_GUs:2 + 2 * self.n_GUs],
                        action[:, 2 + 2 * self.n_GUs:]
                    ]

        if self.discrete_associate:
            if self.fix_uav_pos:
                mask_1 = actions[1] == 0
                mask_2 = actions[2] == 0
                # 合并 mask_2 和 mask_3，找到需要置 0 的位置
                combined_mask = mask_1 | mask_2
                # 将 actions[1] 中对应位置置为 0
                actions[0][combined_mask] = 0
                epsilon = 0
                # epsilon = 1e-10
                actions[1] = np.clip(actions[1], epsilon, 1)
                actions[2] = np.clip(actions[2], epsilon, 1)
                for n in range(self.n_GUs):
                    user_selection = actions[0][:, n].copy()
                    bandwidth_allocation = actions[1][:, n].copy()
                    computation_allocation = actions[2][:, n].copy()
                    if np.sum(user_selection) == 0:
                        actions[0][:, n] = 0
                        actions[1][:, n] = 0
                        actions[2][:, n] = 0
                    elif np.sum(user_selection) == 1:
                        best_uav = np.argmax(actions[0][:, n])  # UAV m is offloading task n
                        actions[0][:, n] = 0
                        actions[0][best_uav, n] = 1
                        actions[1][:, n] = 0
                        actions[1][best_uav, n] = bandwidth_allocation[best_uav]
                        actions[2][:, n] = 0
                        actions[2][best_uav, n] = computation_allocation[best_uav]
                    else:
                        best_uav = -1
                        best_reward = -float('inf')
                        for m in range(self.n_UAVs):
                            if actions[0][m, n] == 1:
                                bandwidth_allocation_m_n = actions[1][m, n] * self.B
                                computation_allocation_m_n = actions[2][m, n] * self.F_m

                                # Calculate total energy and delay for UAV m serving user n
                                uav_m_position = self.uav_positions[m]
                                gu_n_position = self.gu_positions[n]
                                gu_n_task = self.gu_tasks[n]

                                epsilon = 0  # 防止分母为0，其实分母为0计算的值没加和到奖励里边。
                                d_nm_3 = np.linalg.norm(uav_m_position - gu_n_position)
                                theta_nm = 180 / np.pi * np.arcsin((self.H_UAV - self.H_GU) / d_nm_3)
                                P_LoS = 1 / (1 + a * np.exp(-b * (theta_nm - a)))
                                P_NLoS = 1 - P_LoS
                                PL_LoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_LoS
                                PL_NLoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_NLoS
                                PL_nm = P_LoS * PL_LoS + P_NLoS * PL_NLoS
                                h_nm = 10 ** (-PL_nm / 10)
                                R_nm = bandwidth_allocation_m_n * np.log2(
                                    1 + p_t * h_nm / (N_0 * (bandwidth_allocation_m_n + epsilon)))

                                tau_trans = gu_n_task[0] / (R_nm + epsilon)
                                E_trans = p_t * tau_trans
                                tau_exe = gu_n_task[1] / (computation_allocation_m_n + epsilon)
                                E_exe = k_server * (computation_allocation_m_n ** 2) * gu_n_task[1]
                                total_energy = E_trans + E_exe
                                total_delay = tau_trans + tau_exe

                                reward = self.w1 * (self.gu_tasks[n, 2] - total_delay) - self.w2 * total_energy

                                if reward > best_reward:
                                    best_reward = reward
                                    best_uav = m
                        actions[0][:, n] = 0
                        actions[0][best_uav, n] = 1
                        actions[1][:, n] = 0
                        actions[1][best_uav, n] = bandwidth_allocation[best_uav]
                        actions[2][:, n] = 0
                        actions[2][best_uav, n] = computation_allocation[best_uav]
                for m in range(self.n_UAVs):
                    total = np.sum(actions[1][m])
                    # if total > 0:
                    if total > 1:
                        actions[1][m] = actions[1][m] / total
                for m in range(self.n_UAVs):
                    total = np.sum(actions[2][m])
                    # if total > 0:
                    if total > 1:
                        actions[2][m] = actions[2][m] / total
            else:
                mask_2 = actions[2] == 0
                mask_3 = actions[3] == 0
                # 合并 mask_2 和 mask_3，找到需要置 0 的位置
                combined_mask = mask_2 | mask_3
                # 将 actions[1] 中对应位置置为 0
                actions[1][combined_mask] = 0
                epsilon = 0
                # epsilon = 1e-10
                actions[2] = np.clip(actions[2], epsilon, 1)
                actions[3] = np.clip(actions[3], epsilon, 1)
                for n in range(self.n_GUs):
                    user_selection = actions[1][:, n].copy()
                    bandwidth_allocation = actions[2][:, n].copy()
                    computation_allocation = actions[3][:, n].copy()
                    if np.sum(user_selection) == 0:
                        actions[1][:, n] = 0
                        actions[2][:, n] = 0
                        actions[3][:, n] = 0
                    elif np.sum(user_selection) == 1:
                        best_uav = np.argmax(actions[1][:, n])  # UAV m is offloading task n
                        actions[1][:, n] = 0
                        actions[1][best_uav, n] = 1
                        actions[2][:, n] = 0
                        actions[2][best_uav, n] = bandwidth_allocation[best_uav]
                        actions[3][:, n] = 0
                        actions[3][best_uav, n] = computation_allocation[best_uav]
                    else:
                        best_uav = -1
                        best_reward = -float('inf')
                        for m in range(self.n_UAVs):
                            if actions[1][m, n] == 1:
                                bandwidth_allocation_m_n = actions[2][m, n] * self.B
                                computation_allocation_m_n = actions[3][m, n] * self.F_m

                                # Calculate total energy and delay for UAV m serving user n
                                uav_m_position = self.uav_positions[m]
                                gu_n_position = self.gu_positions[n]
                                gu_n_task = self.gu_tasks[n]

                                epsilon = 0  # 防止分母为0，其实分母为0计算的值没加和到奖励里边。
                                d_nm_3 = np.linalg.norm(uav_m_position - gu_n_position)
                                theta_nm = 180 / np.pi * np.arcsin((self.H_UAV - self.H_GU) / d_nm_3)
                                P_LoS = 1 / (1 + a * np.exp(-b * (theta_nm - a)))
                                P_NLoS = 1 - P_LoS
                                PL_LoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_LoS
                                PL_NLoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_NLoS
                                PL_nm = P_LoS * PL_LoS + P_NLoS * PL_NLoS
                                h_nm = 10 ** (-PL_nm / 10)
                                R_nm = bandwidth_allocation_m_n * np.log2(1 + p_t * h_nm / (N_0 * (bandwidth_allocation_m_n + epsilon)))

                                tau_trans = gu_n_task[0] / (R_nm + epsilon)
                                E_trans = p_t * tau_trans
                                tau_exe = gu_n_task[1] / (computation_allocation_m_n + epsilon)
                                E_exe = k_server * (computation_allocation_m_n ** 2) * gu_n_task[1]
                                total_energy = E_trans + E_exe
                                total_delay = tau_trans + tau_exe

                                reward = self.w1 * (self.gu_tasks[n, 2] - total_delay) - self.w2 * total_energy

                                if reward > best_reward:
                                    best_reward = reward
                                    best_uav = m
                        actions[1][:, n] = 0
                        actions[1][best_uav, n] = 1
                        actions[2][:, n] = 0
                        actions[2][best_uav, n] = bandwidth_allocation[best_uav]
                        actions[3][:, n] = 0
                        actions[3][best_uav, n] = computation_allocation[best_uav]
                for m in range(self.n_UAVs):
                    total = np.sum(actions[2][m])
                    # if total > 0:
                    if total > 1:
                        actions[2][m] = actions[2][m] / total
                for m in range(self.n_UAVs):
                    total = np.sum(actions[3][m])
                    # if total > 0:
                    if total > 1:
                        actions[3][m] = actions[3][m] / total
        elif self.continuous_associate:
            if self.fix_uav_pos:
                mask_1 = actions[1] == 0
                mask_2 = actions[2] == 0
                combined_mask = mask_1 | mask_2
                actions[0][combined_mask] = 0
                col_max = np.max(actions[0], axis=0)
                # 为了防止最大值为0的存在，导致取到多个不存在的无人机。
                col_max = np.where(col_max, col_max, 99.0)
                actions[0] = (actions[0] == col_max).astype(int)
                for n in range(self.n_GUs):
                    user_selection = actions[0][:, n].copy()
                    bandwidth_allocation = actions[1][:, n].copy()
                    computation_allocation = actions[2][:, n].copy()
                    if np.sum(user_selection) == 0:
                        actions[0][:, n] = 0
                        actions[1][:, n] = 0
                        actions[2][:, n] = 0
                    elif np.sum(user_selection) == 1:
                        best_uav = np.argmax(actions[0][:, n])  # UAV m is offloading task n
                        actions[0][:, n] = 0
                        actions[0][best_uav, n] = 1
                        actions[1][:, n] = 0
                        actions[1][best_uav, n] = bandwidth_allocation[best_uav]
                        actions[2][:, n] = 0
                        actions[2][best_uav, n] = computation_allocation[best_uav]
                    else:
                        active_mask = user_selection == 1
                        distances = np.linalg.norm(self.uav_positions - self.gu_positions[n], axis=1)
                        # 只考虑活跃无人机的距离，非活跃无人机距离设为无穷大
                        distances_masked = np.where(active_mask, distances, np.inf)
                        # 找到距离最近的无人机索引
                        nearest_uav_idx = np.argmin(distances_masked)
                        actions[0][:, n] = 0
                        actions[0][nearest_uav_idx, n] = 1
                        actions[1][:, n] = 0
                        actions[1][nearest_uav_idx, n] = bandwidth_allocation[nearest_uav_idx]
                        actions[2][:, n] = 0
                        actions[2][nearest_uav_idx, n] = computation_allocation[nearest_uav_idx]
                for m in range(self.n_UAVs):
                    total = np.sum(actions[1][m])
                    if total > 1:
                        actions[1][m] = actions[1][m] / total
                for m in range(self.n_UAVs):
                    total = np.sum(actions[2][m])
                    if total > 1:
                        actions[2][m] = actions[2][m] / total
            else:
                if self.ave_resource:
                    col_max = np.max(actions[1], axis=0)
                    # 为了防止最大值为0的存在，导致取到多个不存在的无人机。
                    col_max = np.where(col_max, col_max, 99.0)
                    actions[1] = (actions[1] == col_max).astype(int)
                    for n in range(self.n_GUs):
                        user_selection = actions[1][:, n].copy()
                        if np.sum(user_selection) == 0:
                            actions[1][:, n] = 0
                        elif np.sum(user_selection) == 1:
                            best_uav = np.argmax(actions[1][:, n])  # UAV m is offloading task n
                            actions[1][:, n] = 0
                            actions[1][best_uav, n] = 1
                        else:
                            active_mask = user_selection == 1
                            distances = np.linalg.norm(self.uav_positions - self.gu_positions[n], axis=1)
                            # 只考虑活跃无人机的距离，非活跃无人机距离设为无穷大
                            distances_masked = np.where(active_mask, distances, np.inf)
                            # 找到距离最近的无人机索引
                            nearest_uav_idx = np.argmin(distances_masked)
                            actions[1][:, n] = 0
                            actions[1][nearest_uav_idx, n] = 1
                else:
                    mask_2 = actions[2] == 0
                    mask_3 = actions[3] == 0
                    combined_mask = mask_2 | mask_3
                    actions[1][combined_mask] = 0
                    col_max = np.max(actions[1], axis=0)
                    # 为了防止最大值为0的存在，导致取到多个不存在的无人机。把0变为不可能出现的99
                    col_max = np.where(col_max, col_max, 99.0)
                    actions[1] = (actions[1] == col_max).astype(int)
                    for n in range(self.n_GUs):
                        user_selection = actions[1][:, n].copy()
                        bandwidth_allocation = actions[2][:, n].copy()
                        computation_allocation = actions[3][:, n].copy()
                        if np.sum(user_selection) == 0:
                            actions[1][:, n] = 0
                            actions[2][:, n] = 0
                            actions[3][:, n] = 0
                        elif np.sum(user_selection) == 1:
                            best_uav = np.argmax(actions[1][:, n])  # UAV m is offloading task n
                            actions[1][:, n] = 0
                            actions[1][best_uav, n] = 1
                            actions[2][:, n] = 0
                            actions[2][best_uav, n] = bandwidth_allocation[best_uav]
                            actions[3][:, n] = 0
                            actions[3][best_uav, n] = computation_allocation[best_uav]
                        else:
                            active_mask = user_selection == 1
                            distances = np.linalg.norm(self.uav_positions - self.gu_positions[n], axis=1)
                            # 只考虑活跃无人机的距离，非活跃无人机距离设为无穷大
                            distances_masked = np.where(active_mask, distances, np.inf)
                            # 找到距离最近的无人机索引
                            nearest_uav_idx = np.argmin(distances_masked)
                            actions[1][:, n] = 0
                            actions[1][nearest_uav_idx, n] = 1
                            actions[2][:, n] = 0
                            actions[2][nearest_uav_idx, n] = bandwidth_allocation[nearest_uav_idx]
                            actions[3][:, n] = 0
                            actions[3][nearest_uav_idx, n] = computation_allocation[nearest_uav_idx]
                    for m in range(self.n_UAVs):
                        total = np.sum(actions[2][m])
                        if total > 1:
                            actions[2][m] = actions[2][m] / total
                    for m in range(self.n_UAVs):
                        total = np.sum(actions[3][m])
                        if total > 1:
                            actions[3][m] = actions[3][m] / total
        else:
            if self.fix_uav_pos:
                for n in range(self.n_GUs):
                    active_mask = (actions[0][:, n] > 0) & (actions[1][:, n] > 0)
                    bandwidth_allocation = actions[0][:, n].copy()
                    computation_allocation = actions[1][:, n].copy()
                    if np.sum(active_mask) == 0:
                        actions[0][:, n] = 0
                        actions[1][:, n] = 0
                    elif np.sum(active_mask) == 1:
                        best_uav = np.argmax(active_mask)  # UAV m is offloading task n
                        actions[0][:, n] = 0
                        actions[0][best_uav, n] = bandwidth_allocation[best_uav]
                        actions[1][:, n] = 0
                        actions[1][best_uav, n] = computation_allocation[best_uav]
                    else:
                        distances = np.linalg.norm(self.uav_positions - self.gu_positions[n], axis=1)
                        # 只考虑活跃无人机的距离，非活跃无人机距离设为无穷大
                        distances_masked = np.where(active_mask, distances, np.inf)
                        # 找到距离最近的无人机索引
                        nearest_uav_idx = np.argmin(distances_masked)
                        actions[0][:, n] = 0
                        actions[0][nearest_uav_idx, n] = bandwidth_allocation[nearest_uav_idx]
                        actions[1][:, n] = 0
                        actions[1][nearest_uav_idx, n] = computation_allocation[nearest_uav_idx]
                for m in range(self.n_UAVs):
                    total = np.sum(actions[0][m])
                    # if total > 0:
                    if total > 1:
                        actions[0][m] = actions[0][m] / total
                for m in range(self.n_UAVs):
                    total = np.sum(actions[1][m])
                    # if total > 0:
                    if total > 1:
                        actions[1][m] = actions[1][m] / total
            else:
                for n in range(self.n_GUs):
                    active_mask = (actions[1][:, n] > 0) & (actions[2][:, n] > 0)
                    bandwidth_allocation = actions[1][:, n].copy()
                    computation_allocation = actions[2][:, n].copy()
                    if np.sum(active_mask) == 0:
                        actions[1][:, n] = 0
                        actions[2][:, n] = 0
                    elif np.sum(active_mask) == 1:
                        best_uav = np.argmax(active_mask)  # UAV m is offloading task n
                        actions[1][:, n] = 0
                        actions[1][best_uav, n] = bandwidth_allocation[best_uav]
                        actions[2][:, n] = 0
                        actions[2][best_uav, n] = computation_allocation[best_uav]
                    else:
                        distances = np.linalg.norm(self.uav_positions - self.gu_positions[n], axis=1)
                        # 只考虑活跃无人机的距离，非活跃无人机距离设为无穷大
                        distances_masked = np.where(active_mask, distances, np.inf)
                        # 找到距离最近的无人机索引
                        nearest_uav_idx = np.argmin(distances_masked)
                        actions[1][:, n] = 0
                        actions[1][nearest_uav_idx, n] = bandwidth_allocation[nearest_uav_idx]
                        actions[2][:, n] = 0
                        actions[2][nearest_uav_idx, n] = computation_allocation[nearest_uav_idx]
                for m in range(self.n_UAVs):
                    total = np.sum(actions[1][m])
                    # if total > 0:
                    if total > 1:
                        actions[1][m] = actions[1][m] / total
                for m in range(self.n_UAVs):
                    total = np.sum(actions[2][m])
                    # if total > 0:
                    if total > 1:
                        actions[2][m] = actions[2][m] / total
        return np.concatenate(actions, axis=-1)

    def process_local_actions(self, action):
        # 带transformed表示，处理成全局角度的动作。
        action = np.clip(action, 0., 1.)
        if self.fix_uav_pos:
            if self.nearest_associate:
                action_components = [
                    action[:, :self.max_GUs_in_range],  # Bandwidth allocation
                    action[:, self.max_GUs_in_range:]  # Computation resource allocation
                ]
                offloading_actions = None
                bandwidth_actions = action_components[0]
                computation_actions = action_components[1]
                transformed_offloading = None
            else:
                action_components = [
                    action[:, :self.max_GUs_in_range],  # Offloading decisions
                    action[:, self.max_GUs_in_range:2 * self.max_GUs_in_range],  # Bandwidth allocation
                    action[:, 2 * self.max_GUs_in_range:]  # Computation resource allocation
                ]
                offloading_actions = action_components[0]
                bandwidth_actions = action_components[1]
                computation_actions = action_components[2]
                transformed_offloading = np.zeros((self.n_UAVs, self.n_GUs), dtype=offloading_actions.dtype)
        else:
            if self.nearest_associate:
                action_components = [
                    action[:, :2],  # Movement actions
                    action[:, 2: 2 + self.max_GUs_in_range],  # Bandwidth allocation
                    action[:, 2 + self.max_GUs_in_range:]  # Computation resource allocation
                ]
                offloading_actions = None
                bandwidth_actions = action_components[1]
                computation_actions = action_components[2]
                transformed_offloading = None
            else:
                if self.ave_resource:
                    action_components = [
                        action[:, :2],  # Movement actions
                        action[:, 2:2 + self.max_GUs_in_range]  # Offloading decisions
                    ]
                    offloading_actions = action_components[1]
                    bandwidth_actions = None
                    computation_actions = None
                    transformed_offloading = np.zeros((self.n_UAVs, self.n_GUs), dtype=offloading_actions.dtype)
                else:
                    action_components = [
                        action[:, :2],  # Movement actions
                        action[:, 2:2 + self.max_GUs_in_range],  # Offloading decisions
                        action[:, 2 + self.max_GUs_in_range:2 + 2 * self.max_GUs_in_range],  # Bandwidth allocation
                        action[:, 2 + 2 * self.max_GUs_in_range:]  # Computation resource allocation
                    ]
                    offloading_actions = action_components[1]
                    bandwidth_actions = action_components[2]
                    computation_actions = action_components[3]
                    transformed_offloading = np.zeros((self.n_UAVs, self.n_GUs), dtype=offloading_actions.dtype)
        transformed_bandwidth = np.zeros((self.n_UAVs, self.n_GUs), dtype=np.float32)
        transformed_computation = np.zeros((self.n_UAVs, self.n_GUs), dtype=np.float32)
        valid_mask = self.nearby_gus_of_uavs[:, :self.max_GUs_in_range] != -1
        for uav_idx in range(self.n_UAVs):
            valid_indices = self.nearby_gus_of_uavs[uav_idx, :self.max_GUs_in_range][valid_mask[uav_idx]]
            valid_pos = np.where(valid_mask[uav_idx])[0]
            if len(valid_indices) > 0:
                if self.nearest_associate:
                    transformed_offloading = None
                else:
                    transformed_offloading[uav_idx, valid_indices] = offloading_actions[uav_idx, valid_pos]
                if self.ave_resource:
                    transformed_bandwidth = None
                    transformed_computation = None
                else:
                    transformed_bandwidth[uav_idx, valid_indices] = bandwidth_actions[uav_idx, valid_pos]
                    transformed_computation[uav_idx, valid_indices] = computation_actions[uav_idx, valid_pos]
        if self.fix_uav_pos:
            if self.nearest_associate:
                transformed_actions_array = np.concatenate([transformed_bandwidth, transformed_computation], axis=1)
            else:
                transformed_actions_array = np.concatenate([transformed_offloading, transformed_bandwidth, transformed_computation], axis=1)
        else:
            if self.nearest_associate:
                transformed_actions_array = np.concatenate([action_components[0], transformed_bandwidth, transformed_computation], axis=1)
            else:
                if self.ave_resource:
                    transformed_actions_array = np.concatenate([action_components[0], transformed_offloading], axis=1)
                else:
                    transformed_actions_array = np.concatenate([action_components[0], transformed_offloading, transformed_bandwidth, transformed_computation], axis=1)
        processed_transformed_actions = self.process_actions(transformed_actions_array)     # 全局动作，去除重复之后的。
        if self.fix_uav_pos:
            if self.nearest_associate:
                processed_transformed_offloading = None
                processed_transformed_bandwidth = processed_transformed_actions[:, :self.n_GUs]
                processed_transformed_computation = processed_transformed_actions[:, self.n_GUs:]
                processed_local_offloading = None
            else:
                processed_transformed_offloading = processed_transformed_actions[:, :self.n_GUs]
                processed_transformed_bandwidth = processed_transformed_actions[:, self.n_GUs:2 * self.n_GUs]
                processed_transformed_computation = processed_transformed_actions[:, 2 * self.n_GUs:]
                processed_local_offloading = np.zeros((self.n_UAVs, self.max_GUs_in_range), dtype=offloading_actions.dtype)
        else:
            if self.nearest_associate:
                processed_transformed_offloading = None
                processed_transformed_bandwidth = processed_transformed_actions[:, 2:2 + self.n_GUs]
                processed_transformed_computation = processed_transformed_actions[:, 2 + self.n_GUs:]
                processed_local_offloading = None
            else:
                if self.ave_resource:
                    processed_transformed_offloading = processed_transformed_actions[:, 2:2 + self.n_GUs]
                    processed_transformed_bandwidth = None
                    processed_transformed_computation = None
                    processed_local_offloading = np.zeros((self.n_UAVs, self.max_GUs_in_range), dtype=offloading_actions.dtype)
                else:
                    processed_transformed_offloading = processed_transformed_actions[:, 2:2 + self.n_GUs]
                    processed_transformed_bandwidth = processed_transformed_actions[:, 2 + self.n_GUs:2 + 2 * self.n_GUs]
                    processed_transformed_computation = processed_transformed_actions[:, 2 + 2 * self.n_GUs:]
                    processed_local_offloading = np.zeros((self.n_UAVs, self.max_GUs_in_range), dtype=offloading_actions.dtype)

        processed_local_bandwidth = np.zeros((self.n_UAVs, self.max_GUs_in_range), dtype=np.float32)
        processed_local_computation = np.zeros((self.n_UAVs, self.max_GUs_in_range), dtype=np.float32)
        for uav_idx in range(self.n_UAVs):
            valid_indices = self.nearby_gus_of_uavs[uav_idx, :self.max_GUs_in_range][valid_mask[uav_idx]]
            valid_pos = np.where(valid_mask[uav_idx])[0]
            if len(valid_indices) > 0:
                if self.nearest_associate:
                    processed_local_offloading = None
                else:
                    processed_local_offloading[uav_idx, valid_pos] = processed_transformed_offloading[uav_idx, valid_indices]
                if self.ave_resource:
                    processed_local_bandwidth = None
                    processed_local_computation = None
                else:
                    processed_local_bandwidth[uav_idx, valid_pos] = processed_transformed_bandwidth[uav_idx, valid_indices]
                    processed_local_computation[uav_idx, valid_pos] = processed_transformed_computation[uav_idx, valid_indices]
        if self.fix_uav_pos:
            if self.nearest_associate:
                processed_local_actions = np.concatenate([processed_local_bandwidth, processed_local_computation], axis=1)
            else:
                processed_local_actions = np.concatenate([processed_local_offloading, processed_local_bandwidth, processed_local_computation], axis=1)
        else:
            if self.nearest_associate:
                processed_local_actions = np.concatenate([action_components[0], processed_local_bandwidth, processed_local_computation], axis=1)
            else:
                if self.ave_resource:
                    processed_local_actions = np.concatenate([action_components[0], processed_local_offloading], axis=1)
                else:
                    processed_local_actions = np.concatenate([action_components[0], processed_local_offloading, processed_local_bandwidth, processed_local_computation], axis=1)
        return processed_local_actions

    def step(self, action):   # acts是一个(n_agents, )的数组，每个元素代表动作序号
        self.time_step += 1
        # Ensure action is a numpy array
        assert isinstance(action, (list, tuple, np.ndarray)), "Action must be a list or tuple or numpy array"

        # 计算local奖励时，先用了动作转换，用到了nearby_gus_of_uavs，但是不用先更新。因为就是用之前的信息，找到对应之前的动作。用来计算奖励。
        if self.not_process_action:
            rewards = self.calculate_local_reward_raw_action(action)
        else:
            rewards = self.calculate_local_reward(action)
        self.cumulative_reward += np.mean(rewards) * np.ones_like(rewards)

        # 更新无人机位置。Update UAV positions
        if self.fix_uav_pos:
            pass
        else:
            fly_action = action[:, :2] * np.array([2 * np.pi, self.v_max])  # 方向和速度都是0-1之间的数
            for i in range(self.n_UAVs):
                direction, velocity = fly_action[i]
                self.uav_positions[i, :2] += velocity * self.Delta_t * np.array([np.cos(direction), np.sin(direction)])
            self.uav_positions[:, :2] = np.clip(self.uav_positions[:, :2], 0, self.x_max)
        # 地面用户位置移动（发现总是会走到最左边。）
        # Update ground user velocities and directions using Gauss-Markov Model
        random_normal_vel = np.random.normal(0, 0.01*self.std_dev_gaussian, self.n_GUs)
        self.gu_velocities = self.alpha_gaussian * self.gu_velocities + (1 - self.alpha_gaussian) * self.mean_velocity + np.sqrt(1 - self.alpha_gaussian ** 2) * random_normal_vel
        random_normal_dir = np.random.normal(0, 0.01*self.std_dev_gaussian, self.n_GUs)
        self.gu_directions = self.alpha_gaussian * self.gu_directions + (1 - self.alpha_gaussian) * self.gu_directions_0 + np.sqrt(1 - self.alpha_gaussian ** 2) * random_normal_dir

        # Update ground user positions
        self.gu_positions[:, 0] += self.gu_velocities * np.cos(self.gu_directions) * self.Delta_t
        self.gu_positions[:, 1] += self.gu_velocities * np.sin(self.gu_directions) * self.Delta_t
        # With this bouncing implementation:
        for i in range(self.n_GUs):
            hit_vertical = False
            hit_horizontal = False
            # Check for x boundary collisions
            if self.gu_positions[i, 0] < 0:
                hit_vertical = True
                self.gu_positions[i, 0] = -self.gu_positions[i, 0]  # Reflect position
            elif self.gu_positions[i, 0] > self.x_max:
                hit_vertical = True
                self.gu_positions[i, 0] = 2 * self.x_max - self.gu_positions[i, 0]  # Reflect position
            # Check for y boundary collisions
            if self.gu_positions[i, 1] < 0:
                hit_horizontal = True
                self.gu_positions[i, 1] = -self.gu_positions[i, 1]  # Reflect position
            elif self.gu_positions[i, 1] > self.x_max:
                hit_horizontal = True
                self.gu_positions[i, 1] = 2 * self.x_max - self.gu_positions[i, 1]  # Reflect position

            if hit_vertical and hit_horizontal:
                # 碰到角落 - 两个方向都反射（旋转180°）
                self.gu_directions[i] = (self.gu_directions[i] + np.pi) % (2 * np.pi)
                self.gu_directions_0[i] = self.gu_directions[i].copy()
            elif hit_vertical:
                # 碰到垂直墙壁 - 水平反射
                self.gu_directions[i] = (np.pi - self.gu_directions[i]) % (2 * np.pi)
                self.gu_directions_0[i] = self.gu_directions[i].copy()
            elif hit_horizontal:
                # 碰到水平墙壁 - 垂直反射
                self.gu_directions[i] = (2 * np.pi - self.gu_directions[i]) % (2 * np.pi)
                self.gu_directions_0[i] = self.gu_directions[i].copy()

        # Generate new tasks for ground users
        gu_tasks = self.generate_tasks()
        self.gu_tasks[:, :3] = gu_tasks[:, :3]
        self.nearby_gus_of_uavs = self.get_nearby_users_sorted_all()

        self.attention_active_mask = np.zeros((self.n_UAVs, self.max_UAVs_obs_concat), dtype=np.float32)
        local_obs = self.get_local_obs()
        # # 自己的obs直接就不变了。
        self.obs = local_obs

        if self.perform_with_local_state:
            self.state = self.obs
        elif self.state_is_k_hops:
            uav_uav_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.uav_positions[:, :2].T[np.newaxis, :], axis=1)
            single_state_dim = self.state_dim // self.max_UAVs_obs_concat
            final_state = np.zeros((self.n_UAVs, self.state_dim))
            final_state[:, :single_state_dim] = local_obs
            for i in range(self.n_UAVs):
                neighbor_mask = (uav_uav_distances[i] <= self.neighbor_distance) & (np.arange(self.n_UAVs) != i)
                neighbors = np.where(neighbor_mask)[0]
                if len(neighbors) > 0:
                    sorted_neighbors = neighbors[np.argsort(uav_uav_distances[i, neighbors])]
                    closest = sorted_neighbors[:self.max_UAVs_obs_concat - 1]
                    for j, neighbor_idx in enumerate(closest):
                        start_pos = (j + 1) * single_state_dim
                        end_pos = (j + 2) * single_state_dim
                        final_state[i, start_pos:end_pos] = local_obs[neighbor_idx]
                self.attention_active_mask[i, :min(len(neighbors) + 1, self.max_UAVs_obs_concat)] = 1
            self.state = final_state
        else:
            # self.state = self.get_state()
            # self.state = np.tile(local_obs.reshape((1, -1)), (self.n_UAVs, 1))
            self.state = np.concatenate((local_obs, np.tile(local_obs.reshape((1, -1)), (self.n_UAVs, 1))), axis=-1)
            # self.state = np.concatenate((local_obs, self.get_state()), axis=-1)

        self.avail_actions = self.get_local_avail_actions()

        dones = np.asarray([False]*self.n_agents)
        info = {}
        if self.average_neighbor_advantage:
            self.Metropolis_weights = self.get_neighbor_weights()
        else:
            self.Metropolis_weights = self.get_Metropolis_weights()
        # if self.time_step % 5 == 0:
        #     transformed_action_components = self.transform_uav_actions(action)
        #     self.render(timestep=self.time_step, title='6', acts=transformed_action_components[:, 2:2+self.n_GUs])
        #     time.sleep(0.05)
        if self.time_step >= self.MAX_SIMULATION_TIME:
            dones = 1 - dones
            info = {'cumulative_reward': self.cumulative_reward, 'n_GUs_per_uav_served': self.n_GUs_per_uav_served/self.MAX_SIMULATION_TIME,
                    'uav_m_toal_energy_consumption': self.uav_energy_consumption, 'user_in_m_average_delay': self.user_average_delay/self.MAX_SIMULATION_TIME,
                    'cumulative_individual_reward': self.cumulative_individual_reward, 'system_performance': self.system_performance,
                    'system_performance_individual': self.system_performance_individual,
                    'system_performance_true_all_GUs': self.system_performance_true_all_GUs,
                    'delay_true_all_GUs':self.delay_true_all_GUs/self.MAX_SIMULATION_TIME,
                    'delay_true_coverd_GUs':self.delay_true_coverd_GUs/self.MAX_SIMULATION_TIME,
                    'energy_true_all_GUs':self.energy_true_all_GUs/self.MAX_SIMULATION_TIME,
                    'energy_all_GUs_UAVs':self.energy_all_GUs_UAVs/self.MAX_SIMULATION_TIME,
                    'system_performance_coverd_GUs': self.system_performance_coverd_GUs,
                    'n_GUs_by_coverd':self.n_GUs_by_coverd/(self.MAX_SIMULATION_TIME/2)}
        return self.obs, rewards, dones, self.state, self.avail_actions, info, self.Metropolis_weights, self.attention_active_mask

    def calculate_local_reward_raw_action(self, action):
        # 转到全局n_GUs的动作，然后处理动作。还是n_GUs的直接放到calculate_reward
        # if self.not_process_action:这个是我之前用来测试，在buffer保存raw-actions会怎么样。
        transformed_action_components = self.transform_uav_actions(action)
        processed_actions = self.process_actions(transformed_action_components)
        rewards = self.calculate_reward(processed_actions)
        return rewards

    def calculate_reward(self, action):
        if self.fix_uav_pos:
            if self.nearest_associate:
                action_components = [
                    action[:, :self.n_GUs],  # Offloading decisions
                    action[:, self.n_GUs:],  # Bandwidth allocation
                ]
                bandwidth_actions = action_components[0] * self.B
                computation_actions = action_components[1] * self.F_m
            else:
                action_components = [
                    action[:, :self.n_GUs],  # Offloading decisions
                    action[:, self.n_GUs:2 * self.n_GUs],  # Bandwidth allocation
                    action[:, 2 * self.n_GUs:]  # Computation resource allocation
                ]
                offloading_actions = action_components[0]
                bandwidth_actions = action_components[1] * self.B
                computation_actions = action_components[2] * self.F_m
        else:
            if self.nearest_associate:
                action_components = [
                    action[:, :2],  # Movement actions
                    action[:, 2:2 + self.n_GUs],  # Offloading decisions
                    action[:, 2 + self.n_GUs:],  # Bandwidth allocation
                ]
                fly_actions = action_components[0]
                bandwidth_actions = action_components[1] * self.B
                computation_actions = action_components[2] * self.F_m
            else:
                if self.ave_resource:
                    action_components = [
                        action[:, :2],  # Movement actions
                        action[:, 2:2 + self.n_GUs]  # Offloading decisions
                    ]
                    fly_actions = action_components[0]
                    offloading_actions = action_components[1]
                    ones_count = np.sum(action_components[1], axis=1, keepdims=True)# 避免除零，使用np.divide处理
                    bandwidth_actions = np.divide(action_components[1] * self.B, ones_count, where=ones_count != 0)
                    ones_count = np.sum(action_components[1], axis=1, keepdims=True)  # 避免除零，使用np.divide处理
                    computation_actions = np.divide(action_components[1] * self.F_m, ones_count, where=ones_count != 0)
                else:
                    action_components = [
                        action[:, :2],  # Movement actions
                        action[:, 2:2 + self.n_GUs],  # Offloading decisions
                        action[:, 2 + self.n_GUs:2 + 2 * self.n_GUs],  # Bandwidth allocation
                        action[:, 2 + 2 * self.n_GUs:]  # Computation resource allocation
                    ]
                    fly_actions = action_components[0]
                    offloading_actions = action_components[1]
                    bandwidth_actions = action_components[2] * self.B
                    computation_actions = action_components[3] * self.F_m
        # Initialize reward arrays
        # per_GU_delay = np.zeros(self.n_GUs)
        per_GU_delay_reward = np.zeros(self.n_GUs)
        per_GU_delay_reward_others = np.zeros(self.n_GUs)
        per_GU_energy_reward = np.zeros(self.n_GUs)
        per_GU_energy_reward_others = np.zeros(self.n_GUs)
        per_GU_task_reward = np.zeros(self.n_GUs)
        per_GU_task_reward_others = np.zeros(self.n_GUs)
        per_GU_delay_true = np.zeros(self.n_GUs)
        per_GU_delay_true_others = np.zeros(self.n_GUs)
        per_GU_energy_true = np.zeros(self.n_GUs)
        per_GU_energy_true_others = np.zeros(self.n_GUs)
        # Calculate UAV-GU distances efficiently

        uav_gu_distances_2d = np.linalg.norm(self.uav_positions[:, None, :2] - self.gu_positions[None, :, :2], axis=2)
        coverage_mask = uav_gu_distances_2d <= self.Cover_R

        R_task = np.zeros(self.n_UAVs)
        R_task_delay = np.zeros(self.n_UAVs)
        R_task_energy = np.zeros(self.n_UAVs)
        for n in range(self.n_GUs):
            uav_indices = np.where(coverage_mask[:, n])[0]
            if self.nearest_associate:
                active_mask = (bandwidth_actions[:, n] > 0) & (computation_actions[:, n] > 0)
                if np.sum(active_mask) == 0:
                    execute_local = True
                else:
                    execute_local = False
                    m = np.argmax(active_mask)
            else:
                if np.sum(offloading_actions[:, n])==0:
                    execute_local = True
                else:
                    execute_local = False
                    m = np.argmax(offloading_actions[:, n])
            if execute_local:  # If no UAV is offloading the task
                total_energy = k_local * (self.F_n ** 2) * self.gu_tasks[n, 1]
                total_delay = self.gu_tasks[n, 1] / self.F_n

                if self.gu_tasks[n, 2] > total_delay:
                    per_GU_delay_reward_others[n] = self.gamma_r * (self.gu_tasks[n, 2] - total_delay)
                else:
                    per_GU_delay_reward_others[n] = -1 * self.delta_r
                per_GU_delay_true_others[n] = total_delay
                per_GU_energy_reward_others[n] = -1 * self.lambda_r * np.clip(total_energy, 0, 10)
                per_GU_energy_true_others[n] = np.clip(total_energy, 0, 10)
                # 这里10，自己加的规定，由于大于1才重新分配动作，某些很少的资源导致计算的时延和能量巨大！ 通常情况下仅为10以内（其实看到的最大只有1.8）。
                per_GU_task_reward_others[n] = per_GU_delay_reward_others[n] + per_GU_energy_reward_others[n]
                self.gu_tasks[n, 3] += 1
            else:
                gu_n_task = self.gu_tasks[n]
                d_nm_3 = np.linalg.norm(self.uav_positions[m] - self.gu_positions[n])
                theta_nm = 180 / np.pi * np.arcsin((self.H_UAV - self.H_GU) / d_nm_3)
                P_LoS = 1 / (1 + a * np.exp(-b * (theta_nm - a)))
                P_NLoS = 1 - P_LoS
                PL_LoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_LoS
                PL_NLoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_NLoS
                PL_nm = P_LoS * PL_LoS + P_NLoS * PL_NLoS
                h_nm = 10 ** (-PL_nm / 10)
                epsilon = 0
                R_nm = bandwidth_actions[m, n] * np.log2(
                    1 + p_t * h_nm / (N_0 * (bandwidth_actions[m, n] + epsilon)))
                tau_trans = gu_n_task[0] / (R_nm + epsilon)
                E_trans = p_t * tau_trans
                tau_exe = gu_n_task[1] / (computation_actions[m, n] + epsilon)
                E_exe = k_server * (computation_actions[m, n] ** 2) * gu_n_task[1]
                total_energy = E_trans + E_exe
                total_delay = tau_trans + tau_exe
                # 指服务的无人机有奖励。
                if gu_n_task[2] > total_delay:
                    R_task_delay[m] += self.gu_tasks[n, 3] * self.gamma_r * (1 + gu_n_task[2] - total_delay)
                    self.gu_tasks[n, 3] = 1
                else:
                    R_task_delay[m] += (-1 * self.gu_tasks[n, 3] * self.delta_r)
                    self.gu_tasks[n, 3] += 1
                R_task_energy[m] += -1 * self.lambda_r * np.clip(total_energy, 0, 10)  # 这里10，自己加的规定，由于大于1才重新分配动作，某些很少的资源导致计算的时延和能量巨大！ 通常情况下仅为10以内（其实看到的最大只有1.8）。

                # 统计系统性能的。
                if self.gu_tasks[n, 2] > total_delay:
                    per_GU_delay_reward[n] = self.gamma_r * (1+self.gu_tasks[n, 2] - total_delay)
                else:
                    per_GU_delay_reward[n] = -1 * self.delta_r
                per_GU_delay_true[n] = total_delay
                per_GU_energy_reward[n] = -1 * self.lambda_r * np.clip(total_energy, 0, 10)
                per_GU_energy_true[n] = np.clip(total_energy, 0, 10)
                # 这里10，自己加的规定，由于大于1才重新分配动作，某些很少的资源导致计算的时延和能量巨大！ 通常情况下仅为10以内（其实看到的最大只有1.8）。
                per_GU_task_reward[n] = per_GU_delay_reward[n] + per_GU_energy_reward[n]
            # # 被服务的用户的奖励平分给覆盖的无人机！
            # if len(uav_indices) > 0 and not execute_local:
            #     R_task_delay[uav_indices] += self.gamma_r * (self.gu_tasks[n, 2] - total_delay)/len(uav_indices) if self.gu_tasks[n, 2] > total_delay else -1 * self.delta_r/len(uav_indices)
            #     R_task_energy[uav_indices] += -1 * self.lambda_r * np.clip(total_energy, 0, 10)/len(uav_indices)

            # if len(uav_indices) > 0:
            #     if self.gu_tasks[n, 2] > total_delay:
            #         per_GU_delay_reward[n] = self.gamma_r * (self.gu_tasks[n, 2] - total_delay)
            #     else:
            #         per_GU_delay_reward[n] = -1 * self.delta_r
            #     per_GU_energy_reward[n] = -1 * self.lambda_r * np.clip(total_energy, 0, 10)
            #     # 这里10，自己加的规定，由于大于1才重新分配动作，某些很少的资源导致计算的时延和能量巨大！ 通常情况下仅为10以内（其实看到的最大只有1.8）。
            #     # 任务奖励为，地面用户任务奖励的平分。
            #     per_GU_task_reward[n] = per_GU_delay_reward[n] + per_GU_energy_reward[n]
                # R_task[uav_indices] += (per_GU_delay_reward[n] + per_GU_energy_reward[n]) / len(uav_indices)
                # # 范围内都有奖励。
                # R_task[uav_indices] += (per_GU_delay_reward[n] + per_GU_energy_reward[n])
                # R_task_delay[uav_indices] += per_GU_delay_reward[n]
                # R_task_energy[uav_indices] += per_GU_energy_reward[n]
            # else:
            #     if self.gu_tasks[n, 2] > total_delay:
            #         per_GU_delay_reward_others[n] = self.gamma_r * (self.gu_tasks[n, 2] - total_delay)
            #     else:
            #         per_GU_delay_reward_others[n] = -1 * self.delta_r
            #     per_GU_energy_reward_others[n] = -1 * self.lambda_r * np.clip(total_energy, 0, 10)
            #     # 这里10，自己加的规定，由于大于1才重新分配动作，某些很少的资源导致计算的时延和能量巨大！ 通常情况下仅为10以内（其实看到的最大只有1.8）。
            #     per_GU_task_reward_others[n] = per_GU_delay_reward_others[n] + per_GU_energy_reward_others[n]
        if self.fix_uav_pos:
            velocity = np.zeros(self.n_UAVs)
        else:
            velocity = fly_actions[:, 1] * self.v_max
        P_fly = P1 * (1 + 3 * velocity ** 2 / U_tip ** 2) + P2 * (np.sqrt(
            1 + velocity ** 4 / (4 * v0 ** 4)) - velocity ** 2 / (2 * v0 ** 2)) ** 0.5 + 0.5 * d0 * rho * g * A * velocity ** 3
        E_fly = P_fly * self.Delta_t  # 速度0到20，能量63到90
        R_fly_energy = -1 * self.lambda_r * E_fly
        uav_uav_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.uav_positions[:, :2].T[np.newaxis, :], axis=1)
        R_collision = -1 * self.mu_r * (np.sum(uav_uav_distances < self.Dis_min, axis=1) - 1)
        uav_gu_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.gu_positions[:, :2].T[np.newaxis, :], axis=1)    #（n_UAVs, n_GUs）
        coverd_gu = np.any(uav_gu_distances<=self.Cover_R, axis=0)      # (n_GUs,)
        R_cover = -1 * self.alpha_r * np.sum(coverd_gu) / self.n_GUs
        rewards = R_cover + R_task_delay + R_task_energy + R_fly_energy + R_collision
        self.cumulative_individual_reward += rewards
        # 无人机角度出发每架无人机自己从服务用户获得的性能。 求和是system_performance。
        self.system_performance += np.sum(R_fly_energy) + np.sum(per_GU_task_reward)
        self.system_performance_individual += (R_task_delay + R_task_energy + R_fly_energy)
        # 用户角度出发计算的全局性能
        self.system_performance_coverd_GUs += np.sum(R_fly_energy) + np.sum(per_GU_task_reward)
        self.system_performance_true_all_GUs += np.sum(R_fly_energy) + np.sum(per_GU_task_reward) + np.sum(per_GU_task_reward_others)
        self.delay_true_all_GUs += (np.sum(per_GU_delay_true) + np.sum(per_GU_delay_true_others))/self.n_GUs
        self.delay_true_coverd_GUs += np.sum(per_GU_delay_true)/np.sum(per_GU_delay_true != 0)
        self.energy_true_all_GUs += np.sum(per_GU_energy_true) + np.sum(per_GU_energy_true_others)
        self.energy_all_GUs_UAVs += np.sum(per_GU_energy_true) + np.sum(per_GU_energy_true_others) + np.sum(E_fly)
        if self.time_step >= (self.MAX_SIMULATION_TIME / 2):
            uav_count_per_user = np.sum(coverage_mask, axis=0)
            self.n_GUs_by_coverd += np.sum(uav_count_per_user > 0)

        if self.local_reward:
            return rewards
        else:
            return np.mean(rewards)*np.ones_like(rewards)

    def calculate_local_reward(self, processed_actions):
        # 之前的处理动作还是回到了n_max_GUs_in_range的。 先转换动作到n_GUs，然后calculate_reward。
        # 其实transform_uav_actions就是process_local_actions的前半段。先变成n_GUs的，然后利用process_actions合法化，然后回到n_max_GUs_in_range。
        transformed_action_components = self.transform_uav_actions(processed_actions)
        rewards = self.calculate_reward(transformed_action_components)
        return rewards

    def transform_uav_actions(self, action_components):
        """
        Transform UAV action arrays based on nearby ground users mapping with optimized performance.
        Parameters:
        -----------
        action_components : list
            List containing action arrays at indices 1, 2, 3 for offloading,
            bandwidth, and computation actions respectively.
        nearby_gus_of_uavs : numpy.ndarray
            Array of shape (n_UAVs, GUs_in_action_dim) indicating the IDs of nearby ground users
            for each UAV, sorted by proximity. Value -1 indicates no user.
        n_GUs : int
            Total number of ground users

        Returns:
        --------
        list
            Updated action_components list with transformed action arrays
        """
        if self.fix_uav_pos:
            if self.nearest_associate:
                action_components = [
                    action_components[:, :self.max_GUs_in_range],
                    action_components[:, self.max_GUs_in_range:]
                ]
                offloading_actions = None
                bandwidth_actions = action_components[0]
                computation_actions = action_components[1]
                transformed_offloading = None
            else:
                action_components = [
                    action_components[:, :self.max_GUs_in_range],  # Offloading decisions
                    action_components[:, self.max_GUs_in_range:2 * self.max_GUs_in_range],
                    # Bandwidth allocation
                    action_components[:, 2 * self.max_GUs_in_range:]  # Computation resource allocation
                ]
                offloading_actions = action_components[0]
                bandwidth_actions = action_components[1]
                computation_actions = action_components[2]
                # Initialize transformed arrays with zeros - preallocate memory once
                transformed_offloading = np.zeros((self.n_UAVs, self.n_GUs), dtype=offloading_actions.dtype)
        else:
            # Extract action arrays
            if self.nearest_associate:
                action_components = [
                    action_components[:, :2],  # Movement actions
                    action_components[:, 2:2 + self.max_GUs_in_range],
                    action_components[:, 2 + self.max_GUs_in_range:]
                ]
                offloading_actions = None
                bandwidth_actions = action_components[1]
                computation_actions = action_components[2]
                transformed_offloading = None
            else:
                if self.ave_resource:
                    action_components = [
                        action_components[:, :2],  # Movement actions
                        action_components[:, 2:2 + self.max_GUs_in_range]
                    ]
                    offloading_actions = action_components[1]
                    bandwidth_actions = None
                    computation_actions = None
                    transformed_offloading = np.zeros((self.n_UAVs, self.n_GUs), dtype=offloading_actions.dtype)
                else:
                    action_components = [
                        action_components[:, :2],  # Movement actions
                        action_components[:, 2:2 + self.max_GUs_in_range],  # Offloading decisions
                        action_components[:, 2 + self.max_GUs_in_range:2 + 2 * self.max_GUs_in_range],  # Bandwidth allocation
                        action_components[:, 2 + 2 * self.max_GUs_in_range:]  # Computation resource allocation
                    ]
                    offloading_actions = action_components[1]
                    bandwidth_actions = action_components[2]
                    computation_actions = action_components[3]
                    # Initialize transformed arrays with zeros - preallocate memory once
                    transformed_offloading = np.zeros((self.n_UAVs, self.n_GUs), dtype=offloading_actions.dtype)
        transformed_bandwidth = np.zeros((self.n_UAVs, self.n_GUs), dtype=np.float32)
        transformed_computation = np.zeros((self.n_UAVs, self.n_GUs), dtype=np.float32)
        # Create a mask for valid user IDs (not -1)
        valid_mask = self.nearby_gus_of_uavs[:, :self.max_GUs_in_range] != -1
        # Use advanced indexing to vectorize the transformation
        # For each UAV
        for uav_idx in range(self.n_UAVs):
            # Get valid indices for this UAV
            valid_indices = self.nearby_gus_of_uavs[uav_idx, :self.max_GUs_in_range][valid_mask[uav_idx]]
            valid_pos = np.where(valid_mask[uav_idx])[0]
            if len(valid_indices) > 0:
                # Efficiently assign values using advanced indexing
                if self.nearest_associate:
                    transformed_offloading = None
                else:
                    transformed_offloading[uav_idx, valid_indices] = offloading_actions[uav_idx, valid_pos]
                if self.ave_resource:
                    transformed_bandwidth = None
                    transformed_computation = None
                else:
                    transformed_bandwidth[uav_idx, valid_indices] = bandwidth_actions[uav_idx, valid_pos]
                    transformed_computation[uav_idx, valid_indices] = computation_actions[uav_idx, valid_pos]
        # Create and return result without copying the entire action_components array
        result = action_components.copy()  # Convert to list if it's not already
        if self.fix_uav_pos:
            if self.nearest_associate:
                result[0] = transformed_bandwidth
                result[1] = transformed_computation
            else:
                result[0] = transformed_offloading
                result[1] = transformed_bandwidth
                result[2] = transformed_computation
        else:
            if self.nearest_associate:
                result[1] = transformed_bandwidth
                result[2] = transformed_computation
            else:
                if self.ave_resource:
                    result[1] = transformed_offloading
                else:
                    result[1] = transformed_offloading
                    result[2] = transformed_bandwidth
                    result[3] = transformed_computation
        result_array = np.concatenate(result, axis=1)
        return result_array

    # def get_obs(self):
    #     '''
    #     [总无人机数目、总用户数目、区域总长度、
    #     所有无人机的位置、
    #     地面用户个数、
    #     服务范围内用户个数、
    #     用户信息。7*_n_GUs。对应位置服务范围内的用户放信息，服务范围外的用户放0]
    #     '''
    #     # Pre-allocate result array
    #     obs = np.zeros((self.n_UAVs, self.obs_dim))
    #     uav_gu_distances_2d = np.linalg.norm(
    #         self.uav_positions[:, None, :2] - self.gu_positions[None, :, :2],
    #         axis=2
    #     )
    #     uav_gu_distances_3d = np.linalg.norm(
    #         self.uav_positions[:, None, :3] - self.gu_positions[None, :, :3],
    #         axis=2
    #     )
    #     coverage_mask = uav_gu_distances_2d <= self.Cover_R
    #     for i in range(self.n_UAVs):
    #         idx = 0
    #         # Add timestep if needed
    #         if self.ob_state_with_timestep:
    #             obs[i, idx] = self.time_step
    #             idx += 1
    #         obs[i, idx:idx + 3] = np.array([self.n_UAVs, self.n_GUs, self.x_max])
    #         idx += 3
    #         # All UAV positions
    #         obs[i, idx:idx + self.n_UAVs * 2] = self.uav_positions[:, :2].flatten()
    #         idx += self.n_UAVs * 2
    #         # Total number of ground users
    #         obs[i, idx] = self.n_GUs
    #         idx += 1
    #         # Number of ground users within coverage range
    #         gu_in_range = np.where(coverage_mask[i])[0]
    #         obs[i, idx] = len(gu_in_range)
    #         idx += 1
    #         # Process all ground users
    #         for j in range(self.n_GUs):
    #             if coverage_mask[i, j]:
    #                 # GU position
    #                 obs[i, idx:idx + 2] = self.gu_positions[j, :2]
    #                 idx += 2
    #                 obs[i, idx] = self.gu_directions[j]
    #                 idx += 1
    #                 # Calculate channel gain
    #                 d_nm_3 = uav_gu_distances_3d[i, j]
    #                 theta_nm = 180 / np.pi * np.arcsin((self.H_UAV - self.H_GU) / d_nm_3)
    #                 P_LoS = 1 / (1 + a * np.exp(-b * (theta_nm - a)))
    #                 P_NLoS = 1 - P_LoS
    #                 PL_LoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_LoS
    #                 PL_NLoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_NLoS
    #                 PL_nm = P_LoS * PL_LoS + P_NLoS * PL_NLoS
    #                 h_nm = 10 ** (-PL_nm / 10)
    #                 # Add channel gain
    #                 obs[i, idx] = h_nm
    #                 idx += 1
    #                 # Add task information
    #                 obs[i, idx:idx + 3] = self.gu_tasks[j]
    #                 idx += 3
    #             else:
    #                 # Skip 7 spots for out-of-range GUs
    #                 idx += 7
    #     return obs

    def get_neighbor_weights(self):
        # 每个智能体i对邻居的权重为1/ (d(i)+1)
        positions = self.uav_positions[:, :2]
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape: (n_UAVs, n_UAVs, 2)
        distances = np.linalg.norm(diff, axis=2)  # shape: (n_UAVs, n_UAVs)

        # Create adjacency matrix based on neighbor_distance
        # adjacency_matrix = (distances <= self.neighbor_distance) & (np.eye(self.n_UAVs) == 0)
        adjacency_matrix = (distances <= self.neighbor_distance)
        adjacency_matrix = adjacency_matrix.astype(float)

        # Calculate degree (number of neighbors) for each UAV
        degrees = np.sum(adjacency_matrix, axis=1)
        degrees_matrix = np.tile(degrees.reshape(-1, 1), (1, self.n_UAVs))
        weights_matrix = np.where(adjacency_matrix > 0,
                                  1.0 / degrees_matrix,
                                  0.0)
        return weights_matrix

    def get_Metropolis_weights(self):
        """
            Compute Metropolis weights between UAVs based on their connectivity.
            Returns:
                weights_matrix: An n_UAVs x n_UAVs matrix containing Metropolis weights
            """
        # Calculate pairwise distances between UAVs using vectorization
        positions = self.uav_positions[:, :2]
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape: (n_UAVs, n_UAVs, 2)
        distances = np.linalg.norm(diff, axis=2)  # shape: (n_UAVs, n_UAVs)

        # Create adjacency matrix based on neighbor_distance
        adjacency_matrix = (distances <= self.neighbor_distance) & (np.eye(self.n_UAVs) == 0)
        adjacency_matrix = adjacency_matrix.astype(float)

        # Calculate degree (number of neighbors) for each UAV
        degrees = np.sum(adjacency_matrix, axis=1)

        # Compute Metropolis weights
        max_degrees_matrix = np.maximum.outer(degrees, degrees)
        weights_matrix = np.where(adjacency_matrix > 0,
                                  1.0 / (1.0 + max_degrees_matrix),
                                  0.0)

        # Calculate self-weights to ensure row stochasticity
        diagonal_values = 1.0 - np.sum(weights_matrix, axis=1)
        # Ensure diagonal values are non-negative (for numerical stability)
        diagonal_values = np.maximum(diagonal_values, 0.0)
        np.fill_diagonal(weights_matrix, diagonal_values)

        return weights_matrix

    def get_local_obs(self):
        """
        添加get_local_obs()方法，每个无人机的局部obs为
        [总无人机数目、总用户数目、区域总长度、
        自身无人机位置；
        与自身距离小于neighbor_distance的邻居无人机的数目；
        距离由近到远的前max_UAVs_in_neighbor架邻居无人机的位置；
        与自身距离小于Cover_R的地面用户的数目；
        距离由近到远的前self.max_GUs_in_range个地面用户的位置、信道增益和计算任务的信息]。
        """
        # Pre-calculate all UAV-to-UAV distances at once (n_UAVs × n_UAVs matrix)
        uav_uav_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.uav_positions[:, :2].T[np.newaxis, :], axis=1)
        # Pre-calculate all UAV-to-GU distances at once (n_UAVs × n_GUs matrix)
        uav_gu_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.gu_positions[:, :2].T[np.newaxis, :], axis=1)
        # Pre-calculate 3D distances for channel gain calculations
        uav_gu_distances_3d = np.linalg.norm(self.uav_positions[:, :3, np.newaxis] - self.gu_positions[:, :3].T[np.newaxis, :], axis=1)
        # Pre-allocate the result array

        # # obsdim应该不变了。直接就是s_{i,t}已经不动了。
        # if self.concat_neighbor_obs:
        #     obs_dim = self.obs_dim // self.max_UAVs_obs_concat
        # else:
        #     obs_dim = self.obs_dim
        obs_dim = self.obs_dim

        local_obs = np.zeros((self.n_UAVs, obs_dim))

        idx = 0
        for i in range(self.n_UAVs):
            if self.ob_state_with_timestep:
                local_obs[i, idx] = self.time_step
                idx += 1

            local_obs[i, idx:idx + 3] = np.array([self.n_UAVs, self.n_GUs, self.x_max])
            idx += 3

            # 2. UAV's own position
            local_obs[i, idx:idx + 2] = self.uav_positions[i, :2]
            idx += 2

            # 3. Find neighboring UAVs (excluding self)
            neighbor_mask = (uav_uav_distances[i] <= self.Cover_R) & (np.arange(self.n_UAVs) != i)
            neighbor_indices = np.where(neighbor_mask)[0]
            neighbor_count = len(neighbor_indices)
            # 4. Number of neighboring UAVs
            local_obs[i, idx] = neighbor_count
            idx += 1
            # 5. Get closest neighbors
            if neighbor_count > 0:
                # Sort neighbors by distance
                neighbor_distances = uav_uav_distances[i, neighbor_indices]
                sorted_idx = np.argsort(neighbor_distances)
                closest_neighbors = neighbor_indices[sorted_idx[:self.max_UAVs_in_neighbor]]
                # Add positions of closest neighbors
                for j, neighbor_idx in enumerate(closest_neighbors):
                    if j < self.max_UAVs_in_neighbor:
                        local_obs[i, idx:idx + 2] = self.uav_positions[neighbor_idx, :2]
                        idx += 2
            # Pad with zeros if there are fewer than max_UAVs_in_neighbor
            padding_neighbors = self.max_UAVs_in_neighbor - min(neighbor_count, self.max_UAVs_in_neighbor)
            idx += padding_neighbors * 2

            # 6. Find ground users within coverage range
            in_range_mask = uav_gu_distances[i] <= self.Cover_R
            in_range_indices = np.where(in_range_mask)[0]
            in_range_count = len(in_range_indices)
            # 7. Number of GUs within coverage
            local_obs[i, idx] = in_range_count
            idx += 1

            # 8. Sort GUs by distance and take closest max_GUs_in_range
            if in_range_count > 0:
                # Sort by distance
                gu_distances = uav_gu_distances[i, in_range_indices]
                sorted_idx = np.argsort(gu_distances)
                closest_gus = in_range_indices[sorted_idx[:self.max_GUs_in_range]]
                for j, gu_idx in enumerate(closest_gus):
                    if j < self.max_GUs_in_range:
                        # GU position (3 values)
                        local_obs[i, idx:idx + 2] = self.gu_positions[gu_idx, :2]
                        idx += 2
                        local_obs[i, idx] = self.gu_directions[gu_idx]
                        idx += 1
                        # Calculate channel gain h_nm
                        d_nm_3 = uav_gu_distances_3d[i, gu_idx]
                        theta_nm = 180 / np.pi * np.arcsin((self.H_UAV - self.H_GU) / d_nm_3)
                        P_LoS = 1 / (1 + a * np.exp(-b * (theta_nm - a)))
                        P_NLoS = 1 - P_LoS
                        PL_LoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_LoS
                        PL_NLoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_NLoS
                        PL_nm = P_LoS * PL_LoS + P_NLoS * PL_NLoS
                        h_nm = 10 ** (-PL_nm / 10)
                        # Add channel gain (1 value)
                        local_obs[i, idx] = h_nm
                        idx += 1
                        # Add task information (4 values)
                        local_obs[i, idx:idx + 4] = self.gu_tasks[gu_idx]
                        idx += 4
                        local_obs[i, idx:idx + 1] = gu_idx
                        idx += 1
            # Reset idx for next UAV's observations
            idx = 0
        return local_obs

    def get_state(self):
        # 目前使用的是利用local_obs拼接的state。这个已经没用了。
        uav_state = []
        if self.ob_state_with_timestep:
            uav_state.append(self.time_step)
        uav_state.extend(np.array([self.n_UAVs, self.n_GUs, self.x_max]))
        uav_state.extend(self.uav_positions[:, :2].flatten())  # 无人机位置2*self.n_UAVs
        uav_state.extend(np.hstack((self.gu_positions[:, :2], self.gu_directions.reshape(-1, 1))).flatten())   # 地面用户位置3*self.n_GUs
        uav_state.extend(self.gu_tasks.flatten())  # 地面用户任务3*self.n_GUs
        state = np.tile(uav_state, (self.n_UAVs, 1))
        state = np.concatenate((np.arange(self.n_UAVs).reshape(-1, 1), state), axis=1)
        return state

    # def get_avail_actions(self):
    #     uav_pos = self.uav_positions[:, :2][:, np.newaxis, :]  # 形状: [n_UAVs, 1, 2]
    #     gu_pos = self.gu_positions[:, :2][np.newaxis, :, :]  # 形状: [1, n_GUs, 2]
    #     # 计算平方距离，避免开方操作
    #     squared_distances = np.sum((uav_pos - gu_pos) ** 2, axis=2)  # 形状: [n_UAVs, n_GUs]
    #     # 生成可用动作矩阵
    #     avail_actions = (squared_distances <= self.Cover_R ** 2).astype(int)
    #     return avail_actions

    def get_local_avail_actions(self):
        """
        根据get_local_obs()方法添加get_local_avail_actions()方法，每个无人机avail_actions的针对
        前 “与自身距离小于Cover_R的地面用户的数目” 个地面用户的可用动作设为1，
        针对后续直到max_GUs_in_range的地面用户的avail_actions占位补位0。
        """
        local_avail_actions = np.zeros((self.n_UAVs, self.max_GUs_in_range), dtype=int)
        # 计算所有UAV和地面用户之间的距离平方（避免计算平方根）
        uav_positions_expanded = self.uav_positions[:, np.newaxis, :2]
        gu_positions_expanded = self.gu_positions[np.newaxis, :, :2]
        distances_squared = np.sum((uav_positions_expanded - gu_positions_expanded) ** 2, axis=2)

        in_range_mask = distances_squared <= self.Cover_R ** 2
        num_in_range = np.sum(in_range_mask, axis=1)
        for i in range(self.n_UAVs):
            if num_in_range[i] >= self.max_GUs_in_range:
                local_avail_actions[i] = 1
            else:
                local_avail_actions[i, :num_in_range[i]] = 1
        return local_avail_actions

    # def calculate_user_reward(self, uav_idx, gu_idx, offloading_action, bandwidth_action, computation_action, with_penalty):
    #     """
    #     Calculate the reward for a specific user given UAV actions
    #     Args:
    #         uav_idx: Index of the UAV
    #         gu_idx: Index of the ground user
    #         offloading_action: Offloading decision。在process_local_actions（）的调用中。offloading_action=1，才调用这个函数
    #         bandwidth_action: Bandwidth allocation
    #         computation_action: Computation resource allocation
    #         with_penalty: 是否考虑分配的资源导致超时，奖励直接设置为-40。 在process_local_actions()中调用时with_penalty=False。在calculate_local_reward()中调用时with_penalty=True
    #     Returns:
    #         reward: The negative of delay and energy consumption
    #     """
    #     uav_m_position = self.uav_positions[uav_idx]
    #     gu_n_position = self.gu_positions[gu_idx]
    #     gu_n_task = self.gu_tasks[gu_idx]
    #
    #     # Calculate local computation energy and delay
    #     if offloading_action==1:
    #         bandwidth = self.B * bandwidth_action
    #         f_uav = self.F_m * computation_action
    #
    #         # Calculate offloading energy and delay
    #         epsilon = 1e-10  # 防止分母为0，其实分母为0计算的值没加和到奖励里边。
    #         d_nm_3 = np.linalg.norm(uav_m_position - gu_n_position)
    #         theta_nm = 180 / np.pi * np.arcsin((self.H_UAV - self.H_GU) / d_nm_3)
    #         P_LoS = 1 / (1 + a * np.exp(-b * (theta_nm - a)))
    #         P_NLoS = 1 - P_LoS
    #         PL_LoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_LoS
    #         PL_NLoS = 20 * np.log10(4 * np.pi * f_c * d_nm_3 / 3e8) + eta_NLoS
    #         PL_nm = P_LoS * PL_LoS + P_NLoS * PL_NLoS
    #         h_nm = 10 ** (-PL_nm / 10)
    #         R_nm = bandwidth * np.log2(1 + p_t * h_nm / (N_0 * (bandwidth + epsilon)))
    #         tau_trans = gu_n_task[0] / (R_nm + epsilon)
    #         E_trans = p_t * tau_trans
    #         tau_exe = gu_n_task[1] / (f_uav + epsilon)
    #         E_exe = k_server * (f_uav ** 2) * gu_n_task[1]
    #         total_energy = E_trans + E_exe
    #         total_delay = tau_trans + tau_exe
    #     else:
    #         total_energy = k_local * (self.F_n ** 2) * gu_n_task[1]
    #         total_delay = gu_n_task[1] / self.F_n
    #
    #     if with_penalty:
    #         # Calculate reward for user n
    #         if gu_n_task[2] < total_delay:
    #             reward = -40
    #         else:
    #             reward = self.w1 * (gu_n_task[2] - total_delay) - self.w2 * total_energy
    #     else:
    #         reward = self.w1 * (gu_n_task[2] - total_delay) - self.w2 * total_energy
    #
    #     return reward

    def get_nearby_users_sorted_all(self):
        """
        For each UAV, find all users within its coverage radius (Cover_R)
        and sort them by distance from closest to farthest.
        Returns:
            nearby_users_sorted: numpy array of shape (n_UAVs, n_GUs) containing
                                 the sorted user IDs for each UAV. Users outside
                                 the coverage radius are marked as -1.
        """
        # Pre-allocate the output array with -1 (indicating no user in range)
        nearby_users_sorted = np.full((self.n_UAVs, self.n_GUs), -1, dtype=np.int32)
        # Compute all distances at once using broadcasting
        # UAVs_pos shape: (n_UAVs, 2 or 3)
        # GUs_pos shape: (n_GUs, 2 or 3)
        # Reshape to enable broadcasting
        uav_positions = self.uav_positions[:, :2].reshape(self.n_UAVs, 1, -1)  # (n_UAVs, 1, dim)
        user_positions = self.gu_positions[:, :2].reshape(1, self.n_GUs, -1)  # (1, n_GUs, dim)
        # Calculate squared distances using broadcasting (avoids sqrt until needed)
        # Result shape: (n_UAVs, n_GUs)
        squared_distances = np.sum((uav_positions - user_positions) ** 2, axis=2)
        # Process each UAV's distances
        for i in range(self.n_UAVs):
            # Get distances for this UAV and identify users within range
            distances = np.sqrt(squared_distances[i])
            in_range_mask = distances <= self.Cover_R
            # Get users that are within range
            in_range_indices = np.where(in_range_mask)[0]
            # If no users in range, continue to next UAV
            if len(in_range_indices) == 0:
                continue
            # Sort the indices by their distances
            sorted_indices = in_range_indices[np.argsort(distances[in_range_indices])]
            # Fill in the sorted user IDs for this UAV (up to the number of users in range)
            nearby_users_sorted[i, :len(sorted_indices)] = sorted_indices
        return nearby_users_sorted

    def get_info(self):
        return {'n_agents': self.n_agents,
                'obs_space': self.observation_space.shape[0],
                'state_space': self.state_space.shape[0],
                'action_space': self.action_space}

    def render(self, mode="human", acts=None, timestep=None, title=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, self.x_max)
        ax.set_ylim(0, self.x_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        # Plot UAVs and annotate their IDs
        for i in range(self.n_UAVs):
            ax.scatter(self.uav_positions[i, 0], self.uav_positions[i, 1], c='r', label='UAV' if i == 0 else "")
            ax.annotate(f'UAV {i}', (self.uav_positions[i, 0], self.uav_positions[i, 1]))
            # Draw service range
            circle = plt.Circle((self.uav_positions[i, 0], self.uav_positions[i, 1]), self.Cover_R, color='r',
                                fill=False, linestyle='--')
            ax.add_patch(circle)

        # Plot GUs and annotate their IDs
        for j in range(self.n_GUs):
            ax.scatter(self.gu_positions[j, 0], self.gu_positions[j, 1], c='b', label='GU' if j == 0 else "")
            # Basic annotation
            # annotation_text = f'GU {j}'
            annotation_text = f'{j}'
            if acts is not None:
                column = acts[:, j]
                if np.any(column == 1):  # 检查是否有1
                    serving_uavs = np.argmax(column == 1)  # 找到第一个1的位置
                else:
                    serving_uavs = -1
                # annotation_text += f'/ {serving_uavs}'
                annotation_text = f'{serving_uavs}'
            ax.annotate(annotation_text, (self.gu_positions[j, 0], self.gu_positions[j, 1]))
        if title is not None:
            plt.title(title)

        ax.legend()
        if timestep is not None:
            if title is not None:
                save_path = title + '-timestep' + str(timestep) + ".png"
            else:
                save_path = str(timestep) + ".png"
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
            plt.close()
        else:
            plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    class Args:
        def __init__(self):
            self.n_UAVs = 5
            self.n_GUs = 50
            self.x_max = 1000

            self.B = 20 * 10 ** 6  # 信道带宽，Hz，20MHz
            self.H_UAV = 120  # m，无人机服务器固定飞行高度
            self.H_GU = 1  # m，地面用户高度
            # 边缘计算参数
            self.F_m = 15 * 10 ** 9  # 15 Gigacycles，无人机服务器的最大可用计算资源
            self.F_n = 1.5 * 10 ** 9  # 1.5 Gigacycles，地面用户的最大可用计算资源
            # 任务参数
            self.D_min = 1 * 10 ** 6  # 1 MB，生成任务的最小数据量
            self.D_max = 3 * 10 ** 6  # 3 MB，生成任务的最大数据量
            self.C_min = 300 * 10 ** 6  # 300 Megacycles，生成任务的最小资源需求
            self.C_max = 500 * 10 ** 6  # 500 Megacycles，生成任务的最大资源需求
            self.delay_min = 0.25  # 250 ms，生成任务的最小延迟要求
            self.delay_max = 0.3  # 300 ms，生成任务的最大延迟要求
            # 这几个参数自己设置的
            self.Dis_min = 20  # 20m, 无人机服务器之间的最小防碰撞距离
            self.Cover_R = self.H_UAV  # 120m, 无人机服务器二维的通信覆盖半径，在Evolutionary Multi-Objective中有最大仰角45度的限制，覆盖范围等于无人机高度。
            self.Delta_t = 0.5  # 0.5s, 无人机服务器的最小时间间隔
            self.episode_length = 50  # 50步数，仿真的最大步数
            # self.MAX_SIMULATION_TIME = args.episode_length  # 50步数，仿真的最大步数，在config.py里边有episode_length
            self.w1 = 20  # 计算奖励时时延部分的权重
            self.w2 = 1  # 计算奖励时能耗部分的权重
            self.p3 = 500  # 计算奖励时防碰撞部分的惩罚值
            self.v_max = 20      # 20m/s，无人机飞行的最大速度
            self.mean_velocity = 10  # 10m/s，地面用户的平均移动速度

    args = Args()
    env = MEC(args)

    # Reset the environment
    obs, state, avail_actions = env.reset()
    print("Initial Observation:", obs)
    print("Initial State:", state)
    print("Initial Available Actions:", avail_actions)
    # Take a random action within the action space bounds
    # flight_action = np.random.uniform(low=env.flight_action_space.low, high=env.flight_action_space.high,
    #                                   size=(env.n_UAVs, 2))
    # task_offloading_action = np.random.randint(low=0, high=2, size=(env.n_UAVs, env.n_GUs))
    # bandwidth_allocation_action = np.random.uniform(low=env.bandwidth_allocation_space.low,
    #                                                 high=env.bandwidth_allocation_space.high,
    #                                                 size=(env.n_UAVs, env.n_GUs))
    # computation_allocation_action = np.random.uniform(low=env.computation_allocation_space.low,
    #                                                   high=env.computation_allocation_space.high,
    #                                                   size=(env.n_UAVs, env.n_GUs))
    #
    # action = (flight_action, task_offloading_action, bandwidth_allocation_action, computation_allocation_action)
    # Take a random action
    action = [np.random.uniform(low=0, high=1, size=(env.n_UAVs, 2)),
              np.random.uniform(low=0, high=1, size=(env.n_UAVs, env.n_GUs)),
              np.random.uniform(low=0, high=1, size=(env.n_UAVs, env.n_GUs)),
              np.random.uniform(low=0, high=1, size=(env.n_UAVs, env.n_GUs))]

    obs, rewards, dones, state, avail_actions, info = env.step(action)
    print("Observation after step:", obs)
    print("Rewards after step:", rewards)
    print("Dones after step:", dones)
    print("State after step:", state)
    print("Available Actions after step:", avail_actions)
    print("Info after step:", info)
