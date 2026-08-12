import copy
import numpy as np
try:
    import gym
    import gym.spaces as spaces
except ImportError:
    import gymnasium as gym
    import gymnasium.spaces as spaces
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
import time
from typing import Dict, Tuple, List, Optional
try:
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
except ImportError:
    Point = Polygon = unary_union = None
import warnings
from matplotlib.patches import Circle
from onpolicy.utils.unreliable_communication import (
    sample_configured_timely_receptions,
)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from torch.cuda.random import seed_all

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
# 常量预计算
LOG_TERM = 20 * np.log10(4 * np.pi * f_c / 3e8)
CONST_TERM_LoS = LOG_TERM + eta_LoS
CONST_TERM_NLoS = LOG_TERM + eta_NLoS


def _offload_delay(tasks, uav_ids, gu_ids, bandwidth, computation, channel_gains):
    """Calculate offloading delays for selected UAV-GU pairs."""
    selected_tasks = tasks[gu_ids]
    selected_bandwidth = bandwidth[uav_ids, gu_ids]
    selected_computation = computation[uav_ids, gu_ids]
    selected_gains = channel_gains[uav_ids, gu_ids]
    rates = selected_bandwidth * np.log2(
        1 + p_t * selected_gains / (N_0 * (selected_bandwidth + 1e-10))
    )
    transmission_delay = selected_tasks[:, 0] / np.maximum(rates, 1e-10)
    execution_delay = selected_tasks[:, 1] / np.maximum(selected_computation, 1e-10)
    return transmission_delay, execution_delay


def _expected_local_task_reward(c_min, c_max, delay_min, delay_max,
                                f_local, gamma_r, delta_r, lambda_r):
    """Expected reward of an i.i.d. task executed locally."""
    cycles = c_min + (np.arange(4096) + 0.5) * (c_max - c_min) / 4096
    local_delay = cycles / f_local
    if delay_max == delay_min:
        delay_reward = np.where(
            delay_min > local_delay,
            gamma_r * (1 + delay_min - local_delay),
            -delta_r,
        )
    else:
        lower = np.clip(local_delay, delay_min, delay_max)
        successful_width = delay_max - lower
        success_probability = successful_width / (delay_max - delay_min)
        successful_reward = gamma_r * (
            (1 - local_delay) * successful_width
            + 0.5 * (delay_max ** 2 - lower ** 2)
        ) / (delay_max - delay_min)
        delay_reward = successful_reward - delta_r * (1 - success_probability)
    local_energy = np.clip(k_local * f_local ** 2 * cycles, 0, 10)
    return np.mean(delay_reward - lambda_r * local_energy)


def _cartesian_flight_velocity(proposals, v_max):
    """Project unconstrained 2-D velocity proposals onto the unit speed disk."""
    proposals = np.asarray(proposals, dtype=np.float64)
    norms = np.linalg.norm(proposals, axis=-1, keepdims=True)
    return v_max * proposals / np.maximum(norms, 1.0)


class DroneInfo:
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        # {other_drone_id: (position, timestamp, hops)}
        self.known_positions: Dict[int, Tuple[np.ndarray, int, int]] = {}
        # 每架无人机当前已知的无人机位置信息
        self.drone_knowledge = set()

    def update_info(self, other_id: int, position: np.ndarray, timestamp: int, hops: int):
        """更新其他无人机的位置信息"""
        if other_id not in self.known_positions:
            self.known_positions[other_id] = (position.copy(), timestamp, hops)
        else:
            _, old_timestamp, old_hops = self.known_positions[other_id]
            # 优先选择更新的信息，如果时间相同则选择跳数更少的
            if (timestamp > old_timestamp or
                    (timestamp == old_timestamp and hops < old_hops)):
                self.known_positions[other_id] = (position.copy(), timestamp, hops)

def calculate_coverage_area(R_cover: float, area_size: float,
                            known_positions: Optional[Dict[int, Tuple[np.ndarray, int, int]]]=None,
                            true_positions: Optional[np.ndarray]=None) -> float:
    """
    计算基于已知位置或者真实位置信息的总覆盖面积
    Args:
        R_cover: 覆盖半径
        area_size: 区域大小 (正方形边长)
        known_positions: 已知的无人机位置信息
        true_positions: 真实位置信息
    Returns:
        覆盖面积占总面积的比例
    """
    if Polygon is None:
        raise ImportError("calculate_coverage_area requires shapely")
    # 创建区域边界
    area_boundary = Polygon([(0, 0), (area_size, 0), (area_size, area_size), (0, area_size)])
    # 创建所有无人机的覆盖圆
    circles = []
    if known_positions is not None:
        for drone_id, (position, _, _) in known_positions.items():
            x, y = position[0], position[1]
            circle = Point(x, y).buffer(R_cover)
            # 只考虑在区域内的部分
            circle_in_area = circle.intersection(area_boundary)
            if not circle_in_area.is_empty:
                circles.append(circle_in_area)
    elif true_positions is not None:
        for position in true_positions:
            x, y = position[0], position[1]
            circle = Point(x, y).buffer(R_cover)
            # 只考虑在区域内的部分
            circle_in_area = circle.intersection(area_boundary)
            if not circle_in_area.is_empty:
                circles.append(circle_in_area)
    else:
        return 0.0
    if not circles:
        return 0.0
    # 计算总覆盖面积（去除重叠）
    try:
        total_coverage = unary_union(circles)
        coverage_area = total_coverage.area
        return coverage_area
    except:
        warnings.warn(f"注意：覆盖面积计算失败")
        # 如果计算失败，返回0
        return 0.0

def update_drone_knowledge_id(drones: List[DroneInfo], positions: np.ndarray, R_cover: float, R_comm: float):
    """更新所有无人机的位置知识"""
    m = len(drones)
    # 计算距离矩阵
    distance_matrix = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
    # 重置所有无人机的知识
    # # 每架无人机都知道自己的位置
    # for i, drone in enumerate(drones):
    #     drone.drone_knowledge = {i}
    # # 第一步：每架无人机通过感知获得直接信息
    # for i, drone in enumerate(drones):
    #     sensing_neighbors_mask = (distance_matrix[i] <= R_cover) & (np.arange(m) != i)
    #     sensing_neighbors = np.where(sensing_neighbors_mask)[0]
    #     drone.drone_knowledge.update(sensing_neighbors)
    #
    # # 第二步：每架无人机通过与一跳邻居通信获得间接信息
    # for i, drone in enumerate(drones):
    #     communication_neighbors_mask = (distance_matrix[i] <= R_comm) & (np.arange(m) != i)
    #     communication_neighbors = np.where(communication_neighbors_mask)[0]
    #     # 从每个通信邻居那里获得其感知范围内的信息
    #     for comm_neighbor in communication_neighbors:
    #         neighbor_sensing_mask = (distance_matrix[comm_neighbor] <= R_cover) & (np.arange(m) != comm_neighbor)
    #         neighbor_sensing = np.where(neighbor_sensing_mask)[0]
    #         # 邻居感知到的无人机，当前无人机也可以知道
    #         drone.drone_knowledge.update(neighbor_sensing)
    #         # 同时也知道这个通信邻居本身的位置
    #         drone.drone_knowledge.add(comm_neighbor)

    # 不要上边了。直接更新通信范围内id
    for i, drone in enumerate(drones):
        communication_neighbors_mask = distance_matrix[i] <= R_comm
        communication_neighbors = np.where(communication_neighbors_mask)[0]
        drone.drone_knowledge.update(communication_neighbors)

def update_drone_knowledge(drones: List[DroneInfo], positions: np.ndarray,
                           current_time: int, k: float):
    """
    更新所有无人机的位置知识：无人机类集合；所有无人机位置；当前时间步；定义交换信息的通信的距离。
    """
    m = len(drones)

    # 计算距离矩阵
    distance_matrix = np.linalg.norm(positions[:, :2, np.newaxis] - positions[:, :2].T[np.newaxis, :], axis=1)
    # 确定一跳邻居
    one_hop_neighbors = {}
    for i in range(m):
        neighbor_mask = (distance_matrix[i] <= k) & (np.arange(m) != i)
        neighbors = np.where(neighbor_mask)[0]
        one_hop_neighbors[i] = neighbors
    # 每个无人机先更新自己的位置信息
    for i, drone in enumerate(drones):
        drone.known_positions[i] = (positions[i].copy(), current_time, 0)
    # 创建临时存储，避免同时读写问题
    temp_updates = {i: {} for i in range(m)}
    # 收集要发送的信息
    for i, drone in enumerate(drones):
        # 向一跳邻居发送自己已知的所有信息
        for neighbor_id in one_hop_neighbors[i]:
            for other_id, (pos, timestamp, hops) in drone.known_positions.items():
                if other_id == neighbor_id:
                    # 邻居的当前位置，跳数为1
                    new_entry = (positions[neighbor_id].copy(), current_time, 1)
                else:
                    # 其他信息，跳数+1
                    new_entry = (pos.copy(), timestamp, hops + 1)

                if neighbor_id not in temp_updates:
                    temp_updates[neighbor_id] = {}
                if other_id not in temp_updates[neighbor_id]:
                    temp_updates[neighbor_id][other_id] = []
                temp_updates[neighbor_id][other_id].append(new_entry)
    # 应用更新
    for i, drone in enumerate(drones):
        if i in temp_updates:
            for other_id, updates in temp_updates[i].items():
                if other_id != i:  # 不更新自己
                    # 选择最好的更新
                    best_update = min(updates, key=lambda x: (current_time - x[1], x[2]))
                    drone.update_info(other_id, best_update[0], best_update[1], best_update[2])

class MEC(gym.Env):
    def __init__(self, args=None):
        assert args is not None
        self.args = args
        self.ob_state_with_timestep = args.ob_state_with_timestep
        self.ob_state_with_id = args.ob_state_with_id
        self.n_UAVs = args.n_UAVs
        self.n_GUs = args.n_GUs
        self.max_GUs_in_range = args.max_GUs_in_range   # 无人机的服务范围内距离由近到远，保留信息的最大用户数目。
        self.dynamic_md = args.dynamic_md
        self.md_arrivals_min = args.md_arrivals_min
        self.md_arrivals_max = args.md_arrivals_max
        self.md_arrivals_per_region = getattr(args, "md_arrivals_per_region", None)
        self.regional_dynamic_md = self.dynamic_md and self.md_arrivals_per_region is not None
        self.hotspot_layout_mode = getattr(args, "hotspot_layout_mode", "fixed_legacy")
        if self.hotspot_layout_mode not in {
            "fixed_legacy", "episode_template4", "episode_template4_600_200",
            "episode_template12", "episode_template12_600_200",
            "episode_moving_template4"
        }:
            raise ValueError(f"unknown hotspot_layout_mode: {self.hotspot_layout_mode}")
        raw_layout_indices = getattr(args, "hotspot_layout_indices", None)
        if raw_layout_indices is None:
            self.hotspot_layout_indices = None
        else:
            self.hotspot_layout_indices = tuple(int(index) for index in raw_layout_indices)
            if not self.hotspot_layout_indices:
                raise ValueError("hotspot_layout_indices must contain at least one index")
            if len(set(self.hotspot_layout_indices)) != len(self.hotspot_layout_indices):
                raise ValueError("hotspot_layout_indices must not contain duplicates")
        self.five_uav_start_layout = getattr(args, "five_uav_start_layout", "line")
        if self.five_uav_start_layout not in {"line", "staggered"}:
            raise ValueError(
                f"unknown five_uav_start_layout: {self.five_uav_start_layout}"
            )
        raw_uav_start_positions = getattr(args, "uav_start_positions", None)
        if raw_uav_start_positions is None:
            self.uav_start_positions = None
        else:
            values = np.asarray(raw_uav_start_positions, dtype=np.float32).reshape(-1)
            expected = 2 * self.n_UAVs
            if values.size != expected:
                raise ValueError(
                    f"uav_start_positions must contain {expected} values "
                    f"for {self.n_UAVs} UAVs, got {values.size}"
                )
            self.uav_start_positions = values.reshape(self.n_UAVs, 2).copy()
        self.hotspot_layout_rng = np.random.default_rng()
        self.candidate_birth_rng = np.random.default_rng()
        self.hotspot_layout_index = 0
        self.hotspot_route_direction = 1
        self.episode_layout_context_enabled = bool(
            getattr(args, "episode_layout_context", False)
        )
        self.episode_layout_context_units = getattr(
            args, "episode_layout_context_units", "normalized_v1"
        )
        if self.episode_layout_context_units not in {
            "normalized_v1", "meters_v2"
        }:
            raise ValueError(
                "unknown episode_layout_context_units: "
                f"{self.episode_layout_context_units}"
            )
        if self.episode_layout_context_enabled and not self.regional_dynamic_md:
            raise ValueError(
                "episode_layout_context requires dynamic_md with regional arrivals"
            )
        if (
            self.episode_layout_context_enabled
            and self.hotspot_layout_mode == "episode_moving_template4"
        ):
            raise ValueError(
                "episode_layout_context is only defined for static hotspot layouts"
            )
        self.layout_context_dim = 8 if self.episode_layout_context_enabled else 0
        self.episode_layout_context = np.zeros(
            self.layout_context_dim, dtype=np.float32
        )
        regional_map_size = float(args.x_max_gu - args.x_min_gu)
        regional_small_size = (
            200.0
            if self.hotspot_layout_mode in {
                "episode_template4_600_200", "episode_template12_600_200"
            }
            else 175.0
        )
        regional_large_near = regional_map_size - 400.0
        self.episode_hotspot_bounds = np.array(
            [
                [0.0, regional_small_size, 0.0, regional_small_size],
                [regional_large_near, regional_map_size,
                 regional_large_near, regional_map_size],
            ],
            dtype=np.float64,
        )
        self.md_lifetime_min = args.md_lifetime_min
        self.md_lifetime_max = args.md_lifetime_max
        if self.dynamic_md:
            assert (
                self.n_UAVs == 5
                and args.x_min_gu == args.y_min_gu == 0
                and args.x_max_gu == args.y_max_gu
                and args.x_min_uav == args.y_min_uav == 0
                and args.x_max_uav == args.y_max_uav == args.x_max_gu
            ), "dynamic_md requires matching square UAV/GU maps for the 5-UAV scenario."
            assert 0 <= self.md_arrivals_min <= self.md_arrivals_max
            assert 1 <= self.md_lifetime_min <= self.md_lifetime_max
            max_arrivals = self.md_arrivals_max
            if self.regional_dynamic_md:
                assert args.fix_hotspot and len(self.md_arrivals_per_region) == 2
                assert all(count >= 0 for count in self.md_arrivals_per_region)
                assert self.hotspot_layout_mode in {
                    "fixed_legacy", "episode_template4",
                    "episode_template4_600_200", "episode_template12",
                    "episode_template12_600_200", "episode_moving_template4"
                }
                if self.hotspot_layout_mode == "fixed_legacy":
                    assert args.x_max_gu == 600, \
                        "fixed_legacy preserves the original 600m hotspot geometry."
                else:
                    assert args.x_max_gu in {600, 700}, \
                        "template hotspot layouts are validated only for 600m and 700m maps."
                if self.hotspot_layout_mode in {
                    "episode_template12", "episode_moving_template4"
                }:
                    assert args.x_max_gu == 700, \
                        f"{self.hotspot_layout_mode} is defined for the 700m map."
                if self.hotspot_layout_mode == "episode_template12_600_200":
                    assert args.x_max_gu == 600, \
                        "episode_template12_600_200 is defined for the 600m map."
                if self.hotspot_layout_mode == "episode_moving_template4":
                    assert args.x_max_gu == 700, \
                        "episode_moving_template4 is defined for the 700m map."
                max_arrivals = sum(self.md_arrivals_per_region)
            assert self.n_GUs >= max_arrivals * self.md_lifetime_max, \
                "dynamic_md requires n_GUs >= maximum arrivals * md_lifetime_max"
            assert 1 <= self.max_GUs_in_range <= self.n_GUs, \
                "dynamic_md requires 1 <= max_GUs_in_range <= n_GUs."
        self.gu_obs_features = 10 if self.dynamic_md else 9
        self.type_s_schema_bits = (
            int(np.ceil(np.log2(self.n_UAVs)))
            + 32 * (2 + int(self.ob_state_with_timestep))
            + int(np.ceil(np.log2(self.n_GUs + 1)))
            + self.max_GUs_in_range * (64 + 32 * self.gu_obs_features + 1)
        )
        self.max_UAVs_in_neighbor = args.max_UAVs_in_neighbor  # 只用到自己的观测s_{i,t}中的其他无人机数目。无人机的邻居范围内距离由近到远，保留信息的最大无人机数目。
        self.neighbor_distance = args.neighbor_distance  # 无人机之间定义为通信的k跳的距离。 之前为d_cov*2=240。我的last-obs设为覆盖范围内的无人机数目。因此设置为120
        self.neighbor_R = args.neighbor_R  # 无人机之间定义为1跳的距离。240+20米。用来告诉无人机其一跳范围内的无人机，感知到的无人机位置共享
        self.d_optimal = args.d_optimal
        self.perform_with_local_state = args.perform_with_local_state
        self.state_is_k_hops = args.state_is_k_hops
        self.all_uav_k_hops = args.all_uav_k_hops   # 如果这个为True，就是k跳邻居的状态不再是由近到远排列。直接按所有id排列，邻居的信息补进去。
        self.use_atten_actor = args.use_atten_actor
        self.communication_mode = getattr(args, "communication_mode", "reliable")
        self.state_reconstruction = getattr(args, "state_reconstruction", "zero")
        if self.state_reconstruction not in {"zero", "last_obs", "md_gru"}:
            raise ValueError("state_reconstruction must be zero, last_obs, or md_gru")
        if self.state_reconstruction != "zero" and (
            self.communication_mode != "unreliable" or not self.dynamic_md
        ):
            raise ValueError(
                "state reconstruction requires unreliable communication and dynamic MDs"
            )
        self.critic_md_metadata = bool(
            getattr(args, "critic_md_metadata", False)
            or self.state_reconstruction != "zero"
        )
        self.critic_md_metadata_features = 3 if self.critic_md_metadata else 0
        if self.communication_mode == "unreliable":
            if not self.state_is_k_hops or not self.all_uav_k_hops:
                raise ValueError(
                    "unreliable Type-S communication requires ID-aligned k-hop critic states"
                )
            if self.use_atten_actor:
                raise ValueError(
                    "unreliable Type-S communication changes only the training critic state"
                )
            if args.state_payload_bits < self.type_s_schema_bits:
                raise ValueError(
                    f"state_payload_bits={args.state_payload_bits} is smaller than "
                    f"the Type-S schema budget {self.type_s_schema_bits} bits"
                )
        self.communication_rng = np.random.default_rng()
        self.last_state_reception_mask = np.ones(
            (self.n_UAVs, self.n_UAVs), dtype=bool
        )
        self.state_packets_received = 0
        self.state_packets_attempted = 0
        self.cartesian_flight = getattr(args, "cartesian_flight", False)
        self.actor_neighbor_obs = getattr(args, "actor_neighbor_obs", False)
        self.actor_neighbor_obs_dim = 3 * (self.n_UAVs - 1) if self.actor_neighbor_obs else 0
        self.actor_message_mode = getattr(args, "actor_message_mode", "disabled")
        if self.actor_message_mode not in {
            "disabled", "zero", "geometry", "task_summary"
        }:
            raise ValueError(f"unknown actor_message_mode: {self.actor_message_mode}")
        self.actor_message_contract = getattr(
            args, "actor_message_contract", "relative_scaled_v1"
        )
        if self.actor_message_contract not in {
            "relative_scaled_v1", "absolute_raw_v2"
        }:
            raise ValueError(
                f"unknown actor_message_contract: {self.actor_message_contract}"
            )
        self.actor_message_features = 10
        self.actor_message_dim = (
            self.actor_message_features * (self.n_UAVs - 1)
            if self.actor_message_mode != "disabled"
            else 0
        )
        self.actor_only_obs_dim = (
            self.actor_neighbor_obs_dim + self.actor_message_dim
        )
        self.spatial_flight_actor = getattr(args, "spatial_flight_actor", False)
        explicit_distance_sort = getattr(args, "distance_only_user_sort", False)
        self.completion_priority_user_sort = getattr(
            args, "completion_priority_user_sort", False
        )
        assert not (
            explicit_distance_sort and self.completion_priority_user_sort
        ), "distance-only and completion-priority user sorting are mutually exclusive"
        # Preserve old Spatial-Actor checkpoints/configs, which implicitly used
        # distance-only sorting, while allowing new runs to request the legacy
        # completion-priority ordering explicitly.
        self.distance_only_user_sort = explicit_distance_sort or (
            self.spatial_flight_actor and not self.completion_priority_user_sort
        )
        self.uav_reset_curriculum = (
            getattr(args, "uav_reset_curriculum", False)
            and getattr(args, "uav_reset_curriculum_training", False)
        )
        self.uav_reset_curriculum_schedule = getattr(
            args, "uav_reset_curriculum_schedule", "legacy"
        )
        if self.uav_reset_curriculum_schedule not in {
            "legacy", "p0p7_10m_25m"
        }:
            raise ValueError(
                f"unknown uav_reset_curriculum_schedule: "
                f"{self.uav_reset_curriculum_schedule}"
            )
        self.curriculum_reset_count = 0
        self.curriculum_random_reset = False
        self.curriculum_random_probability = 0.0
        self.curriculum_reset_rng = np.random.default_rng()
        if self.spatial_flight_actor:
            assert self.dynamic_md
            assert not self.actor_neighbor_obs, \
                "spatial_flight_actor must not receive neighbor UAV observations"
        if self.actor_message_mode != "disabled":
            assert self.spatial_flight_actor, \
                "actor messages currently target only the spatial flight head"
            assert not self.actor_neighbor_obs, \
                "legacy actor_neighbor_obs and actor_message_mode are mutually exclusive"
            assert not self.use_atten_actor, \
                "actor messages require local actor observations"
        assert not (self.actor_neighbor_obs and self.use_atten_actor), \
            "actor_neighbor_obs requires local actor observations"
        if self.uav_reset_curriculum:
            assert self.dynamic_md and self.n_UAVs == 5

        assert not (self.perform_with_local_state and self.state_is_k_hops), "不能同时使用和obs一样的local_state，和k_hops state"
        # self.concat_neighbor_obs = args.concat_neighbor_obs
        self.max_UAVs_obs_concat = args.max_UAVs_obs_concat # 这个是s_{M_i^k}跳要拼接的无人机s_{i,t}的数目。 放在之前就是构造一跳观测的s_{M_i^1}拼接数目。
        if self.all_uav_k_hops:
            assert self.max_UAVs_obs_concat == self.n_UAVs, "固定形式的state，那么state中拼接的obs的个数max_UAVs，得直接等于所有无人机个数。"
        # self.max_UAVs_obs_concat = args.n_UAVs # 这个是s_{M_i^k}跳要拼接的无人机s_{i,t}的数目。 放在之前就是构造一跳观测的s_{M_i^1}拼接数目。
        self.local_reward = args.local_reward
        self.n_agents = self.n_UAVs                 # 100架飞机。
        self.x_min_uav = args.x_min_uav
        self.x_max_uav = args.x_max_uav
        self.y_min_uav = args.y_min_uav
        self.y_max_uav = args.y_max_uav
        self.x_min_gu = args.x_min_gu  # 15km*15km的范围
        self.x_max_gu = args.x_max_gu  # 15km*15km的范围
        self.y_min_gu = args.y_min_gu  # 15km*15km的范围
        self.y_max_gu = args.y_max_gu  # 15km*15km的范围

        self.fix_hotspot = args.fix_hotspot
        self.random_hotspot = args.random_hotspot
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
        self.expected_inactive_md_local_reward = _expected_local_task_reward(
            self.C_min, self.C_max, self.delay_min, self.delay_max,
            self.F_n, self.gamma_r, self.delta_r, self.lambda_r,
        )
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
        self.nearest_avail_actions = args.nearest_avail_actions
        assert self.discrete_associate + self.continuous_associate + self.nearest_associate == 1
        self.not_process_action = args.not_process_action
        self.fix_uav_pos = args.fix_uav_pos
        self.ave_resource = args.ave_resource
        self.ave_bandwidth = args.ave_bandwidth
        if self.ave_resource or self.ave_bandwidth:
            assert (self.continuous_associate and (not self.fix_uav_pos)) or ((not self.fix_uav_pos) and self.nearest_associate), "只写了在无人机飞行且连续associate条件下的平均分配资源"
        if self.nearest_associate and (self.ave_resource or self.ave_bandwidth):
            assert self.not_process_action, "只写了怎么计算奖励。只有飞行动作，不需要处理动作。"
        self.mean_velocity = args.mean_velocity
        self.md_velocity_init_std = float(
            getattr(args, "md_velocity_init_std", 0.3)
        )
        self.md_velocity_init_min_factor = float(
            getattr(args, "md_velocity_init_min_factor", 0.7)
        )
        self.md_velocity_init_max_factor = float(
            getattr(args, "md_velocity_init_max_factor", 1.3)
        )
        self.md_velocity_update_clip_min = getattr(
            args, "md_velocity_update_clip_min", None
        )
        self.md_velocity_update_clip_max = getattr(
            args, "md_velocity_update_clip_max", None
        )
        if self.md_velocity_init_std < 0:
            raise ValueError("md_velocity_init_std must be non-negative")
        if self.md_velocity_init_min_factor > self.md_velocity_init_max_factor:
            raise ValueError(
                "md_velocity_init_min_factor must not exceed "
                "md_velocity_init_max_factor"
            )
        if (
            self.md_velocity_update_clip_min is not None
            and self.md_velocity_update_clip_max is not None
            and self.md_velocity_update_clip_min > self.md_velocity_update_clip_max
        ):
            raise ValueError(
                "md_velocity_update_clip_min must not exceed "
                "md_velocity_update_clip_max"
            )
        # Define the UAV flight direction and distance action space (continuous)
        self.not_served_rew_to_ave = args.not_served_rew_to_ave
        self.not_served_rew_to_nearest = args.not_served_rew_to_nearest
        assert not (self.not_served_rew_to_ave and self.not_served_rew_to_nearest), "未被服务用户，只能要么被平均分，要么给最近的无人机。二选一"
        if self.perform_with_local_state:
            # [总无人机数目、总用户数目、区域总长度、
            # 自身无人机位置；
            # 与自身距离小于Cover_R的邻居无人机的数目；
            # 距离由近到远的前max_UAVs_in_neighbor架邻居无人机的位置；
            # 与自身距离小于Cover_R的地面用户的数目；
            # 距离由近到远的前self.max_GUs_in_range个地面用户的位置、信道增益和计算任务的信息]
            # 再加一位呢，用户自己计算不完，添加标志1.。否则为0.。
            self.GUs_in_action_dim = self.max_GUs_in_range
            # self.obs_dim = 3 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range
            # self.state_dim = 3 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range
            # self.obs_dim = 1 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range
            # self.state_dim = 1 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range

            # 不要邻居无人机的位置。
            # self.obs_dim = 1 + 2 + 1 + 9 * self.max_GUs_in_range
            # self.state_dim = 1 + 2 + 1 + 9 * self.max_GUs_in_range
            self.obs_dim = (
                3
                + self.actor_only_obs_dim
                + self.layout_context_dim
                + self.gu_obs_features * self.max_GUs_in_range
            )
            self.state_dim = self.obs_dim
        elif self.state_is_k_hops:  # last-obs的k跳。自己的s_{i,t}是包括覆盖范围内的无人机的。
            self.GUs_in_action_dim = self.max_GUs_in_range
            # # 包括覆盖范围内d_cov的无人机信息。
            # self.obs_dim = 3 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range
            # self.state_dim = 3 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range
            # self.obs_dim = 1 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range
            # self.state_dim = 1 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range
            # 不要邻居无人机的位置。
            # self.obs_dim = 1 + 2 + 1 + 9 * self.max_GUs_in_range
            # self.state_dim = 1 + 2 + 1 + 9 * self.max_GUs_in_range
            self.obs_dim = (
                3
                + self.actor_only_obs_dim
                + self.layout_context_dim
                + self.gu_obs_features * self.max_GUs_in_range
            )
            self.state_dim = self.obs_dim
        else:
            # self.GUs_in_action_dim = self.n_GUs
            self.GUs_in_action_dim = self.max_GUs_in_range
            # self.obs_dim = 3 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range   # 局部obs的dim
            # self.obs_dim = 1 + 2 + 1 + 2 * self.max_UAVs_in_neighbor + 1 + 9 * self.max_GUs_in_range
            # 不要邻居无人机的位置。
            # self.obs_dim = 1 + 2 + 1 + 9 * self.max_GUs_in_range
            self.obs_dim = (
                3
                + self.actor_only_obs_dim
                + self.layout_context_dim
                + self.gu_obs_features * self.max_GUs_in_range
            )

            # self.state_dim = 1 + 3 +  2*self.n_UAVs+6*self.n_GUs +1
            # self.state_dim = self.n_UAVs * (self.obs_dim + int(self.ob_state_with_timestep))
            self.state_dim = (self.n_UAVs+1) * (self.obs_dim + int(self.ob_state_with_timestep))
            # self.state_dim = (self.obs_dim + int(self.ob_state_with_timestep)) + 1 + 3 +  2*self.n_UAVs+6*self.n_GUs +1

        if self.ob_state_with_timestep:
            self.obs_dim += 1
        if self.ob_state_with_timestep and (self.perform_with_local_state or self.state_is_k_hops):
            self.state_dim += 1

        if self.ob_state_with_id:
            self.obs_dim += self.n_UAVs
        if self.ob_state_with_id and (self.perform_with_local_state or self.state_is_k_hops):
            self.state_dim += self.n_UAVs

        # One-hop UAV positions are actor-only. Keep the critic state identical
        # to the no-neighbor baseline so B0/B1 differ in one factor only.
        self.critic_local_state_dim = (
            self.obs_dim
            - self.actor_only_obs_dim
            + self.critic_md_metadata_features * self.max_GUs_in_range
        )
        if self.perform_with_local_state or self.state_is_k_hops:
            self.state_dim = self.critic_local_state_dim
        else:
            self.state_dim = (self.n_UAVs + 1) * self.critic_local_state_dim

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
            if self.use_atten_actor:
                # 0803，obs和state一样。CTCE。
                self.obs_dim *= self.max_UAVs_obs_concat

        if self.fix_uav_pos:
            pass
        else:
            if self.cartesian_flight:
                self.flight_action_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                )
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
        elif self.ave_bandwidth:
            self.computation_allocation_space = spaces.Box(low=0.0, high=1.0, shape=(self.GUs_in_action_dim,), dtype=np.float32)
        else:
            self.bandwidth_allocation_space = spaces.Box(low=0.0, high=1.0, shape=(self.GUs_in_action_dim,), dtype=np.float32)
            self.computation_allocation_space = spaces.Box(low=0.0, high=1.0, shape=(self.GUs_in_action_dim,), dtype=np.float32)
        # 这里的动作空间设置的n_GUs，也影响这process_actions里边的actions的处理，所以要注意。
        # 组成完整动作空间
        if self.fix_uav_pos and self.nearest_associate:
            self.action_space = spaces.Tuple((self.bandwidth_allocation_space, self.computation_allocation_space))
        elif self.fix_uav_pos and (not self.nearest_associate):
            self.action_space = spaces.Tuple((self.task_offloading_space, self.bandwidth_allocation_space,self.computation_allocation_space))
        elif not self.fix_uav_pos and self.nearest_associate and (not self.ave_resource):
            self.action_space = spaces.Tuple((self.flight_action_space, self.bandwidth_allocation_space, self.computation_allocation_space))
        elif not self.fix_uav_pos and self.ave_resource and (not self.nearest_associate):
            assert self.continuous_associate
            self.action_space = spaces.Tuple((self.flight_action_space, self.task_offloading_space))
        elif not self.fix_uav_pos and self.ave_bandwidth and (not self.nearest_associate):
            assert self.continuous_associate
            self.action_space = spaces.Tuple((self.flight_action_space, self.task_offloading_space, self.computation_allocation_space))
        elif not self.fix_uav_pos and self.nearest_associate and self.ave_resource:
            self.action_space = self.flight_action_space
        else:
            self.action_space = spaces.Tuple((self.flight_action_space, self.task_offloading_space, self.bandwidth_allocation_space,self.computation_allocation_space))
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(self.obs_dim,), dtype=np.float32)
        self.state_space = spaces.Box(low=0.0, high=0.0, shape=(self.state_dim,), dtype=np.float32)

        self.state = None
        self.obs = None
        self.avail_actions = None
        self.time_step = None
        self.attention_active_mask = np.zeros((self.n_UAVs, self.max_UAVs_obs_concat), dtype=np.float32)

        # 缓存计算结果的属性
        self.uav_uav_distances_2d = None
        self.uav_gu_distances_2d = None
        self.uav_gu_distances_3d = None
        self.channel_gains = None
        self.coverage_mask = None

        # 初始化UAV位置
        uav_x = np.random.uniform(self.x_min_uav, self.x_max_uav, self.n_UAVs)
        uav_y = np.random.uniform(self.y_min_uav, self.y_max_uav, self.n_UAVs)
        uav_z = np.full(self.n_UAVs, self.H_UAV)
        self.uav_positions = np.column_stack((uav_x, uav_y, uav_z))
        # self.uav_positions = np.random.uniform(0, self.x_max, (self.n_UAVs, 2))
        # self.uav_positions = np.hstack((self.uav_positions, self.H_UAV * np.ones((self.n_UAVs, 1))))
        # self.drones = [DroneInfo(i) for i in range(self.n_UAVs)]
        # # 修改1: 初始化时每个无人机知道所有其他无人机的初始位置
        # for i, drone in enumerate(self.drones):
        #     for j in range(self.n_UAVs):
        #         drone.known_positions[j] = (self.uav_positions[j][:2].copy(), 0, 0 if i == j else 1)

        # Initialize ground user positions with fixed height of 1m
        # 用的是reset的办法。这里没啥用。
        gu_x = np.random.uniform(self.x_min_gu, self.x_max_gu, self.n_GUs)
        gu_y = np.random.uniform(self.y_min_gu, self.y_max_gu, self.n_GUs)
        gu_z = np.full(self.n_GUs, self.H_GU)
        self.gu_positions = np.column_stack((gu_x, gu_y, gu_z))
        # self.gu_positions = np.random.uniform(0, self.x_max, (self.n_GUs, 2))
        # self.gu_positions = np.hstack((self.gu_positions, self.H_GU * np.ones((self.n_GUs, 1))))
        # Gauss-Markov Model parameters
        # self.alpha_gaussian = 0.95
        self.alpha_gaussian = 0.7
        self.std_dev_gaussian = 1
        self.gu_velocities = self._sample_md_velocities(self.n_GUs)
        # np.random.seed(0)
        self.gu_directions = np.random.uniform(0, 2 * np.pi, self.n_GUs)
        self.gu_directions_0 = self.gu_directions.copy()
        self.active_md_mask = np.ones(self.n_GUs, dtype=bool)
        self.md_remaining_lifetime = np.zeros(self.n_GUs, dtype=np.int32)
        self.md_session_ids = np.full(self.n_GUs, -1, dtype=np.int64)
        self.md_region_ids = np.full(self.n_GUs, -1, dtype=np.int8)
        self.next_md_session_id = 0
        self.dynamic_md_candidates = 0
        self.dynamic_md_admitted = 0
        self.dynamic_md_candidates_by_region = np.zeros(2, dtype=np.int64)
        self.dynamic_md_admitted_by_region = np.zeros(2, dtype=np.int64)
        self.dynamic_md_expired = 0
        self.dynamic_md_departed = 0
        self.dynamic_md_active_sum = 0
        self.dynamic_md_active_sum_second_half = 0
        self.hotspot_small_nearest_uav_distance_sum = 0.0
        self.hotspot_large_four_uav_distance_sum = 0.0
        self.cartesian_proposal_norm_sum = 0.0
        self.cartesian_proposal_count = 0
        self.cartesian_projection_count = 0
        self.actor_message_neighbor_fraction_sum = 0.0
        self.actor_message_payload_norm_sum = 0.0
        self.actor_message_observation_count = 0

        # Initialize ground user tasks
        self.gu_tasks = self.generate_tasks()
        self.cumulative_reward = np.zeros((self.n_agents,))
        self.system_performance = np.zeros((self.n_agents,))
        self.system_performance_individual = np.zeros((self.n_agents,))
        self.system_performance_true_all_GUs = np.zeros((self.n_agents,))
        self.system_performance_equivalent_full_GUs = np.zeros((self.n_agents,))
        self.system_performance_coverd_GUs = np.zeros((self.n_agents,))
        self.delay_true_all_GUs = np.zeros((self.n_agents,))
        self.delay_true_coverd_GUs = np.zeros((self.n_agents, 2))
        self.energy_true_all_GUs = np.zeros((self.n_agents,))
        self.energy_all_GUs_UAVs = np.zeros((self.n_agents,))
        self.n_GUs_by_coverd = 0
        self.cumulative_individual_reward = np.zeros((self.n_agents,))
        self.uav_energy_consumption = np.zeros((self.n_UAVs,))  # 无人机及其范围内用户每时刻能耗和，再对step求和
        self.user_average_delay = np.zeros((self.n_UAVs,))  # 每架无人机范围内用户平均时延，再对step求平均
        self.n_GUs_per_uav_served = np.zeros((self.n_UAVs,))  # 每架无人机每个时刻服务的用户数目，再对step求平均
        self.nearby_gus_of_uavs = -np.ones((self.n_UAVs, self.n_GUs))   # process_local_actions里用到了（类似transform_uav_actions的代码。需要处理和反处理）。
        self.complete_task = np.zeros((self.n_GUs, ))   # 记录当前时刻的动作下，用户是否完成任务。完成为1
        self.self_complete_task = np.zeros((self.n_GUs, ))   # 当前时刻，如果自己计算能不能完成任务？能的话为1
        self.complete_task_ratio = 0  # 统计整个episode完成任务的比率
        self.cumulative_individual_reward_wo_cover = np.zeros((self.n_UAVs,))
        self.cumulative_reward_wo_cover = np.zeros((self.n_UAVs,))
        # calculate_local_reward里先调用到了transform_uav_actions()。
        # get_local_obs()里边，是重新挨个计算的距离。
        self.average_neighbor_advantage = args.average_neighbor_advantage
        self.Metropolis_weights = None      # Metropolis_weights，用来对邻居的Adv进行加权求和。
        self.proposed_offload_actions = np.zeros((self.n_UAVs, self.n_GUs))     # 用来画render。
        self.proposed_offload_actions_all_time = np.zeros((self.MAX_SIMULATION_TIME+1, self.n_UAVs, self.n_GUs))     # 用来画render。
        self.uav_positions_all_time = np.zeros((self.MAX_SIMULATION_TIME+1, self.n_UAVs, 2))
        self.gu_positions_all_time = np.zeros((self.MAX_SIMULATION_TIME+1, self.n_GUs, 4))    # 位置(x, y)和速度，方向
        # self.system_performance_all_uavs = np.zeros((self.MAX_SIMULATION_TIME+1, self.n_UAVs))

    def generate_tasks(self):
        tasks = np.zeros((self.n_GUs, 3))
        gu_ids = np.flatnonzero(self.active_md_mask) if self.dynamic_md else np.arange(self.n_GUs)
        count = len(gu_ids)
        tasks[gu_ids, 0] = np.random.uniform(self.D_min, self.D_max, count)  # Data size
        tasks[gu_ids, 1] = np.random.uniform(self.C_min, self.C_max, count)  # compute Resource demand
        tasks[gu_ids, 2] = np.random.uniform(self.delay_min, self.delay_max, count)  # Delay requirement
        return tasks

    def _sample_dynamic_md_positions(self, count):
        positions = np.zeros((count, 3), dtype=np.float64)
        bounds = np.tile(
            [self.x_min_gu, self.x_max_gu, self.y_min_gu, self.y_max_gu],
            (count, 1),
        )
        positions[:, 2] = self.H_GU
        if self.fix_hotspot and self.n_UAVs == 5 and self.x_min_gu == 0 and self.x_max_gu == 600:
            lower_hotspot = np.random.random(count) < (1.0 / 6.0)
            positions[lower_hotspot, 0] = np.random.uniform(0, 175, np.sum(lower_hotspot))
            positions[lower_hotspot, 1] = np.random.uniform(0, 175, np.sum(lower_hotspot))
            positions[~lower_hotspot, 0] = np.random.uniform(200, 600, np.sum(~lower_hotspot))
            positions[~lower_hotspot, 1] = np.random.uniform(200, 600, np.sum(~lower_hotspot))
            bounds[lower_hotspot] = [0, 175, 0, 175]
            bounds[~lower_hotspot] = [200, 600, 200, 600]
        else:
            positions[:, 0] = np.random.uniform(self.x_min_gu, self.x_max_gu, count)
            positions[:, 1] = np.random.uniform(self.y_min_gu, self.y_max_gu, count)
        return positions, bounds

    def _admit_dynamic_mds(self):
        if self.regional_dynamic_md:
            candidate_rng = (
                self.candidate_birth_rng
                if self.hotspot_layout_mode in {
                    "episode_template4", "episode_template12",
                    "episode_template12_600_200",
                    "episode_moving_template4"
                }
                else np.random
            )
            small_count, large_count = self.md_arrivals_per_region
            candidate_count = small_count + large_count
            candidate_positions = np.zeros((candidate_count, 3), dtype=np.float64)
            candidate_bounds = np.empty((candidate_count, 4), dtype=np.float64)
            candidate_region_ids = np.concatenate((
                np.zeros(small_count, dtype=np.int8),
                np.ones(large_count, dtype=np.int8),
            ))
            candidate_positions[:, 2] = self.H_GU
            for region_id, count in enumerate((small_count, large_count)):
                start = 0 if region_id == 0 else small_count
                stop = start + count
                x_min, x_max, y_min, y_max = self.episode_hotspot_bounds[region_id]
                candidate_positions[start:stop, 0] = candidate_rng.uniform(x_min, x_max, count)
                candidate_positions[start:stop, 1] = candidate_rng.uniform(y_min, y_max, count)
                candidate_bounds[start:stop] = self.episode_hotspot_bounds[region_id]
        else:
            candidate_count = np.random.randint(self.md_arrivals_min, self.md_arrivals_max + 1)
            candidate_positions, candidate_bounds = self._sample_dynamic_md_positions(candidate_count)
            # Preserve the legacy regional accounting for non-regional dynamic
            # arrivals (e.g. the older 3-arrivals/20-slot environment).
            candidate_region_ids = (candidate_bounds[:, 0] != 0).astype(np.int8)
        # Diagnostic-only snapshots. Keeping the candidate-birth stream separate
        # from admission-dependent randomness permits paired environment checks.
        self.last_candidate_positions = candidate_positions.copy()
        self.last_candidate_region_ids = candidate_region_ids.copy()
        covered = np.any(
            np.linalg.norm(
                self.uav_positions[:, None, :2] - candidate_positions[None, :, :2],
                axis=2,
            ) <= self.Cover_R,
            axis=0,
        )
        admitted_positions = candidate_positions[covered]
        admitted_bounds = candidate_bounds[covered]
        admitted_region_ids = candidate_region_ids[covered]
        free_slots = np.flatnonzero(~self.active_md_mask)
        assert len(admitted_positions) <= len(free_slots), "Dynamic MD capacity bound was violated."
        slots = free_slots[:len(admitted_positions)]
        if len(slots):
            self.active_md_mask[slots] = True
            self.gu_positions[slots] = admitted_positions
            self.x_min_all_gus[slots] = admitted_bounds[:, 0]
            self.x_max_all_gus[slots] = admitted_bounds[:, 1]
            self.y_min_all_gus[slots] = admitted_bounds[:, 2]
            self.y_max_all_gus[slots] = admitted_bounds[:, 3]
            self.gu_velocities[slots] = self._sample_md_velocities(len(slots))
            self.gu_directions[slots] = np.random.uniform(0, 2 * np.pi, len(slots))
            self.gu_directions_0[slots] = self.gu_directions[slots]
            self.md_remaining_lifetime[slots] = np.random.randint(
                self.md_lifetime_min, self.md_lifetime_max + 1, len(slots)
            )
            self.md_session_ids[slots] = np.arange(
                self.next_md_session_id, self.next_md_session_id + len(slots), dtype=np.int64
            )
            self.md_region_ids[slots] = admitted_region_ids
            self.next_md_session_id += len(slots)
        self.dynamic_md_candidates += candidate_count
        self.dynamic_md_admitted += len(slots)
        self.dynamic_md_candidates_by_region += np.bincount(
            candidate_region_ids, minlength=2
        )[:2]
        self.dynamic_md_admitted_by_region += np.bincount(
            admitted_region_ids, minlength=2
        )[:2]

    def _advance_dynamic_md_population(self):
        active_slots = np.flatnonzero(self.active_md_mask)
        self.md_remaining_lifetime[active_slots] -= 1
        still_covered = np.any(
            np.linalg.norm(
                self.uav_positions[:, None, :2] - self.gu_positions[None, :, :2],
                axis=2,
            ) <= self.Cover_R,
            axis=0,
        )
        expired = self.active_md_mask & (self.md_remaining_lifetime <= 0)
        departed = self.active_md_mask & ~expired & ~still_covered
        removed = expired | departed
        self.dynamic_md_expired += int(np.sum(expired))
        self.dynamic_md_departed += int(np.sum(departed))
        self.active_md_mask[removed] = False
        self.md_remaining_lifetime[removed] = 0
        self.md_session_ids[removed] = -1
        self.md_region_ids[removed] = -1
        self.gu_positions[removed] = np.array([0.0, 0.0, self.H_GU])
        self.gu_velocities[removed] = 0
        self.gu_directions[removed] = 0
        self.gu_directions_0[removed] = 0
        self.gu_tasks[removed] = 0
        if self.hotspot_layout_mode == "episode_moving_template4":
            centers = np.column_stack((
                self.episode_hotspot_bounds[:, :2].mean(axis=1),
                self.episode_hotspot_bounds[:, 2:].mean(axis=1),
            ))
            distances = np.linalg.norm(
                self.uav_positions[:, None, :2] - centers[None, :, :], axis=2
            )
            self.hotspot_small_nearest_uav_distance_sum += float(distances[:, 0].min())
            self.hotspot_large_four_uav_distance_sum += float(
                np.sort(distances[:, 1])[:4].mean()
            )
        if self.time_step < self.MAX_SIMULATION_TIME:
            self._update_moving_hotspot_layout(self.time_step)
            self._admit_dynamic_mds()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        communication_seed = None if seed is None else seed + 209759
        self.communication_rng = np.random.default_rng(communication_seed)
        self.curriculum_reset_rng = np.random.default_rng(seed)
        layout_seed, candidate_seed = np.random.SeedSequence(seed).spawn(2)
        self.hotspot_layout_rng = np.random.default_rng(layout_seed)
        self.candidate_birth_rng = np.random.default_rng(candidate_seed)

    def _sample_state_reception_mask(self):
        """Return the per-slot Type-S availability with axes (receiver, sender)."""
        if self.communication_mode == "reliable":
            return np.ones((self.n_UAVs, self.n_UAVs), dtype=bool)
        receptions = sample_configured_timely_receptions(
            self.uav_positions[None, ...],
            1,
            self.communication_rng,
            self.args,
            payload_bits=self.args.state_payload_bits,
            deadline_ms=self.args.state_deadline_ms,
        )[0, 0].T
        off_diagonal = ~np.eye(self.n_UAVs, dtype=bool)
        self.state_packets_received += int(np.sum(receptions[off_diagonal]))
        self.state_packets_attempted += int(np.sum(off_diagonal))
        np.fill_diagonal(receptions, True)
        return receptions

    def _build_critic_state(self, critic_local_obs):
        """Build the speed-branch critic layout, optionally masking lost Type-S packets."""
        self.attention_active_mask = np.zeros(
            (self.n_UAVs, self.max_UAVs_obs_concat), dtype=np.float32
        )
        self.last_state_reception_mask = self._sample_state_reception_mask()
        if self.perform_with_local_state:
            self.attention_active_mask[:, 0] = 1.0
            return critic_local_obs
        if not self.state_is_k_hops:
            return np.concatenate((
                critic_local_obs,
                np.tile(critic_local_obs.reshape((1, -1)), (self.n_UAVs, 1)),
            ), axis=-1)

        single_state_dim = self.state_dim // self.max_UAVs_obs_concat
        final_state = np.zeros((self.n_UAVs, self.state_dim))
        if self.all_uav_k_hops:
            available = (
                self.uav_uav_distances_2d <= self.neighbor_distance
                if self.neighbor_distance > 0
                else np.eye(self.n_UAVs, dtype=bool)
            )
            available &= self.last_state_reception_mask
            np.fill_diagonal(available, True)
            all_uavs = np.arange(self.n_UAVs)
            sender_order = np.asarray([
                np.concatenate(([receiver], all_uavs[all_uavs != receiver]))
                for receiver in range(self.n_UAVs)
            ])
            state_by_sender = np.broadcast_to(
                critic_local_obs[None],
                (self.n_UAVs, self.n_UAVs, single_state_dim),
            ).copy()
            state_by_sender[~available] = 0
            ordered = np.take_along_axis(
                state_by_sender, sender_order[..., None], axis=1
            )
            self.attention_active_mask = np.take_along_axis(
                available, sender_order, axis=1
            ).astype(np.float32)
            return ordered.reshape(self.n_UAVs, self.state_dim)

        final_state[:, :single_state_dim] = critic_local_obs
        self.attention_active_mask[:, 0] = 1.0
        for receiver in range(self.n_UAVs):
            neighbor_mask = (
                (self.neighbor_distance > 0)
                & (self.uav_uav_distances_2d[receiver] <= self.neighbor_distance)
                & self.last_state_reception_mask[receiver]
                & (np.arange(self.n_UAVs) != receiver)
            )
            neighbors = np.where(neighbor_mask)[0]
            closest = neighbors[
                np.argsort(self.uav_uav_distances_2d[receiver, neighbors])
            ][:self.max_UAVs_obs_concat - 1]
            for slot, sender in enumerate(closest, start=1):
                start = slot * single_state_dim
                final_state[receiver, start:start + single_state_dim] = (
                    critic_local_obs[sender]
                )
            self.attention_active_mask[receiver, 1:len(closest) + 1] = 1.0
        return final_state

    @staticmethod
    def _regional_hotspot_layouts(map_size):
        """Four area-preserving diagonal layouts on a square map."""
        map_size = float(map_size)
        small_far = map_size - 175.0
        large_near = map_size - 400.0
        return np.array([
            [[0, 175, 0, 175], [large_near, map_size, large_near, map_size]],
            [[small_far, map_size, 0, 175], [0, 400, large_near, map_size]],
            [[0, 175, small_far, map_size], [large_near, map_size, 0, 400]],
            [[small_far, map_size, small_far, map_size], [0, 400, 0, 400]],
        ], dtype=np.float64)

    @staticmethod
    def _regional_hotspot_layouts600_200(map_size):
        """Four diagonal layouts for the 600m 200m/400m protocol."""
        map_size = float(map_size)
        small_far = map_size - 200.0
        large_near = map_size - 400.0
        return np.array([
            [[0, 200, 0, 200], [large_near, map_size, large_near, map_size]],
            [[small_far, map_size, 0, 200], [0, 400, large_near, map_size]],
            [[0, 200, small_far, map_size], [large_near, map_size, 0, 400]],
            [[small_far, map_size, small_far, map_size], [0, 400, 0, 400]],
        ], dtype=np.float64)

    @staticmethod
    def _regional_hotspot_layouts12(map_size, small_size=175.0):
        """All ordered non-overlapping corner pairs on a square map.

        Region 0 is always the small birth region and region 1 is
        always the large (400m) birth region.  The large corner is sampled
        first, then the small corner is sampled from the other three corners.
        ``small_size`` is 175m for the legacy 700m protocol and 200m for the
        600m protocol.
        """
        map_size = float(map_size)
        small_size = float(small_size)
        corners = ((0.0, 0.0), (map_size - small_size, 0.0),
                   (0.0, map_size - small_size),
                   (map_size - small_size, map_size - small_size))
        large_near = map_size - 400.0
        large_corners = ((0.0, 0.0), (large_near, 0.0),
                         (0.0, large_near), (large_near, large_near))
        layouts = []
        for large_id, (large_x, large_y) in enumerate(large_corners):
            for small_id, (small_x, small_y) in enumerate(corners):
                if small_id == large_id:
                    continue
                layouts.append([
                    [small_x, small_x + small_size,
                     small_y, small_y + small_size],
                    [large_x, large_x + 400.0, large_y, large_y + 400.0],
                ])
        return np.asarray(layouts, dtype=np.float64)

    def _set_episode_hotspot_layout(self, layout_index=None):
        """Select the episode layout and refresh the optional task context."""
        if not self.regional_dynamic_md:
            return
        map_size = self.x_max_gu - self.x_min_gu
        layouts = (
            self._regional_hotspot_layouts600_200(map_size)
            if self.hotspot_layout_mode == "episode_template4_600_200"
            else self._regional_hotspot_layouts12(map_size, small_size=200.0)
            if self.hotspot_layout_mode == "episode_template12_600_200"
            else self._regional_hotspot_layouts12(map_size)
            if self.hotspot_layout_mode == "episode_template12"
            else self._regional_hotspot_layouts(map_size)
        )
        if self.hotspot_layout_mode in {"fixed_legacy", "episode_template4_600_200"}:
            layout_index = 0
        elif layout_index is None:
            candidate_indices = (
                self.hotspot_layout_indices
                if self.hotspot_layout_indices is not None
                else tuple(range(len(layouts)))
            )
            if any(index < 0 or index >= len(layouts) for index in candidate_indices):
                raise ValueError(
                    f"hotspot_layout_indices must be within [0, {len(layouts) - 1}]"
                )
            layout_index = int(self.hotspot_layout_rng.choice(candidate_indices))
        if not 0 <= int(layout_index) < len(layouts):
            raise ValueError(f"invalid regional hotspot layout index: {layout_index}")
        self.hotspot_layout_index = int(layout_index)
        if self.hotspot_layout_mode == "episode_moving_template4":
            self.hotspot_route_direction = int(
                self.hotspot_layout_rng.choice(np.array([-1, 1], dtype=np.int8))
            )
            self._update_moving_hotspot_layout(0)
        else:
            self.episode_hotspot_bounds = layouts[self.hotspot_layout_index].copy()
        self._refresh_episode_layout_context()

    def _refresh_episode_layout_context(self):
        """Encode static birth-region bounds for the optional task prior.

        The context contains only the two reset-time birth rectangles, in the
        fixed order [small, large]. It contains no active-user state, current
        counts, or future random arrivals.
        """
        if not self.episode_layout_context_enabled:
            return
        bounds = np.asarray(self.episode_hotspot_bounds, dtype=np.float64).copy()
        if self.episode_layout_context_units == "normalized_v1":
            x_scale = max(float(self.x_max_gu - self.x_min_gu), 1.0)
            y_scale = max(float(self.y_max_gu - self.y_min_gu), 1.0)
            bounds[:, [0, 1]] = (
                bounds[:, [0, 1]] - float(self.x_min_gu)
            ) / x_scale
            bounds[:, [2, 3]] = (
                bounds[:, [2, 3]] - float(self.y_min_gu)
            ) / y_scale
        self.episode_layout_context = bounds.astype(np.float32).reshape(-1)

    @staticmethod
    def _moving_hotspot_waypoints_700():
        """Symmetric region waypoints with equal 300 m center translations."""
        small_half = 87.5
        large_half = 200.0
        centers = np.array([
            [200.0, 200.0], [500.0, 200.0],
            [500.0, 500.0], [200.0, 500.0],
        ])
        opposite = np.roll(centers, 2, axis=0)
        layouts = np.empty((4, 2, 4), dtype=np.float64)
        for index, (small_center, large_center) in enumerate(zip(centers, opposite)):
            layouts[index, 0] = [
                small_center[0] - small_half, small_center[0] + small_half,
                small_center[1] - small_half, small_center[1] + small_half,
            ]
            layouts[index, 1] = [
                large_center[0] - large_half, large_center[0] + large_half,
                large_center[1] - large_half, large_center[1] + large_half,
            ]
        return layouts

    def _update_moving_hotspot_layout(self, time_step):
        """Move two hidden birth regions continuously; active MDs are untouched."""
        if self.hotspot_layout_mode != "episode_moving_template4":
            return
        waypoints = self._moving_hotspot_waypoints_700()
        # Two 300 m route edges per 400-slot episode: 3 m/s at Delta_t=0.5 s.
        route_progress = 2.0 * min(max(float(time_step), 0.0), self.MAX_SIMULATION_TIME) \
            / float(self.MAX_SIMULATION_TIME)
        completed_edges = int(min(np.floor(route_progress), 1.0))
        fraction = route_progress - completed_edges
        start = (
            self.hotspot_layout_index
            + self.hotspot_route_direction * completed_edges
        ) % len(waypoints)
        stop = (start + self.hotspot_route_direction) % len(waypoints)
        self.episode_hotspot_bounds = (
            (1.0 - fraction) * waypoints[start] + fraction * waypoints[stop]
        )

    def _sample_md_velocities(self, count):
        """Sample newly born MD speeds using the configured Gaussian profile."""
        if count <= 0:
            return np.empty(0, dtype=np.float64)
        speeds = np.random.normal(
            self.mean_velocity, self.md_velocity_init_std, int(count)
        )
        return np.clip(
            speeds,
            self.md_velocity_init_min_factor * self.mean_velocity,
            self.md_velocity_init_max_factor * self.mean_velocity,
        )

    def _curriculum_random_reset_probability(self):
        """Return the random-UAV-reset probability for the current budget."""
        completed_steps = (
            self.curriculum_reset_count
            * self.MAX_SIMULATION_TIME
            * max(int(getattr(self.args, "n_rollout_threads", 1)), 1)
        )
        progress = completed_steps / max(float(getattr(self.args, "num_env_steps", 1)), 1.0)
        if self.uav_reset_curriculum_schedule == "p0p7_10m_25m":
            completed_budget_steps = completed_steps
            if completed_budget_steps < 10_000_000:
                return 0.70
            if completed_budget_steps < 25_000_000:
                return 0.70 * (
                    25_000_000 - completed_budget_steps
                ) / 15_000_000
            return 0.0
        if progress < 0.20:
            return 0.50
        if progress < 0.50:
            return 0.50 * (0.50 - progress) / 0.30
        return 0.0

    def _sample_curriculum_uav_positions(self):
        """Uniform map resets with no hotspot or target-coordinate information."""
        positions = np.zeros((self.n_UAVs, 3), dtype=np.float32)
        positions[:, 2] = self.H_UAV
        minimum_distance = float(self.Cover_R)
        for i in range(self.n_UAVs):
            accepted = False
            for _ in range(1000):
                candidate = np.array([
                    self.curriculum_reset_rng.uniform(self.x_min_uav, self.x_max_uav),
                    self.curriculum_reset_rng.uniform(self.y_min_uav, self.y_max_uav),
                ])
                if i == 0 or np.all(
                    np.linalg.norm(positions[:i, :2] - candidate, axis=1)
                    >= minimum_distance
                ):
                    positions[i, :2] = candidate
                    accepted = True
                    break
            if not accepted:
                positions[i, :2] = candidate
        return positions[self.curriculum_reset_rng.permutation(self.n_UAVs)]

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
        if self.n_UAVs == 9 and self.x_min_gu == 0 and self.x_max_gu == 600 and self.n_GUs == 80:
            # self.uav_positions = np.array([[120, 120, self.H_UAV], [300,120, self.H_UAV], [480, 120, self.H_UAV],
            #                                [120, 300, self.H_UAV], [300, 300, self.H_UAV], [480, 300, self.H_UAV],
            #                                [120, 480, self.H_UAV],[300, 480, self.H_UAV], [480, 480, self.H_UAV]], dtype=np.float32)
            self.uav_positions = np.array([[100, 100, self.H_UAV], [300, 100, self.H_UAV], [500, 100, self.H_UAV],
                                           [100, 300, self.H_UAV], [300, 300, self.H_UAV], [500, 300, self.H_UAV],
                                           [100, 500, self.H_UAV], [300, 500, self.H_UAV], [500, 500, self.H_UAV]],
                                          dtype=np.float32)
            if self.fix_hotspot:
                self.x_min_all_gus = np.array([self.x_min_gu] * 50 + [350] * 30)
                self.x_max_all_gus = np.array([self.x_max_gu] * 50 + [600] * 30)
                self.y_min_all_gus = np.array([self.y_min_gu] * 50 + [350] * 30)
                self.y_max_all_gus = np.array([self.y_max_gu] * 50 + [600] * 30)
            elif self.random_hotspot:
                x_min_gu_hotspot = random.choice([0, 350])
                y_min_gu_hotspot = random.choice([0, 350])
                self.x_min_all_gus = np.array([self.x_min_gu] * 50 + [x_min_gu_hotspot] * 30)
                self.x_max_all_gus = np.array([self.x_max_gu] * 50 + [x_min_gu_hotspot+250] * 30)
                self.y_min_all_gus = np.array([self.y_min_gu] * 50 + [y_min_gu_hotspot] * 30)
                self.y_max_all_gus = np.array([self.y_max_gu] * 50 + [y_min_gu_hotspot+250] * 30)
            else:
                self.x_min_all_gus = np.array([self.x_min_gu] * 80)
                self.x_max_all_gus = np.array([self.x_max_gu] * 80)
                self.y_min_all_gus = np.array([self.y_min_gu] * 80)
                self.y_max_all_gus = np.array([self.y_max_gu] * 80)
        elif self.n_UAVs == 4 and self.x_min_gu == 0 and self.x_max_gu == 400 and self.n_GUs == 40:
            self.uav_positions = np.array([[50, 25, self.H_UAV], [100, 25, self.H_UAV], [150, 25, self.H_UAV],
                                           [200, 25, self.H_UAV]], dtype=np.float32)
            # self.uav_positions = np.array([[100, 100, self.H_UAV], [300, 100, self.H_UAV], [100, 300, self.H_UAV],
            #                                    [300, 300, self.H_UAV]], dtype=np.float32)
            if self.fix_hotspot:
                self.x_min_all_gus = np.array([self.x_min_gu] * 20 + [250] * 20)
                self.x_max_all_gus = np.array([self.x_max_gu] * 20 + [400] * 20)
                self.y_min_all_gus = np.array([self.y_min_gu] * 20 + [250] * 20)
                self.y_max_all_gus = np.array([self.y_max_gu] * 20 + [400] * 20)
            elif self.random_hotspot:
                x_min_gu_hotspot = random.choice([0, 250])
                y_min_gu_hotspot = random.choice([0, 250])
                self.x_min_all_gus = np.array([self.x_min_gu] * 20 + [x_min_gu_hotspot] * 20)
                self.x_max_all_gus = np.array([self.x_max_gu] * 20 + [x_min_gu_hotspot+150] * 20)
                self.y_min_all_gus = np.array([self.y_min_gu] * 20 + [y_min_gu_hotspot] * 20)
                self.y_max_all_gus = np.array([self.y_max_gu] * 20 + [y_min_gu_hotspot+150] * 20)
            else:
                self.x_min_all_gus = np.array([self.x_min_gu] * 40)
                self.x_max_all_gus = np.array([self.x_max_gu] * 40)
                self.y_min_all_gus = np.array([self.y_min_gu] * 40)
                self.y_max_all_gus = np.array([self.y_max_gu] * 40)
        elif self.n_UAVs == 4 and self.x_min_gu == 0 and self.x_max_gu == 600 and self.n_GUs == 40:
            self.uav_positions = np.array([[400, 400, self.H_UAV], [50, 50, self.H_UAV], [100, 50, self.H_UAV],
                                               [150, 50, self.H_UAV]], dtype=np.float32)
            if self.fix_hotspot:
                self.x_min_all_gus = np.array([200] * 40)
                self.x_max_all_gus = np.array([600] * 40)
                self.y_min_all_gus = np.array([200] * 40)
                self.y_max_all_gus = np.array([600] * 40)
        elif self.n_UAVs == 5 and self.x_min_gu == 0 and self.x_max_gu in {600, 700}:
            if self.uav_start_positions is not None:
                self.uav_positions = np.column_stack((
                    self.uav_start_positions,
                    np.full((self.n_UAVs, 1), self.H_UAV, dtype=np.float32),
                ))
            elif (
                self.regional_dynamic_md
                and self.hotspot_layout_mode in {
                    "episode_template4_600_200", "episode_template12_600_200"
                }
                and self.x_max_gu == 600
            ):
                self.uav_positions = np.array([
                    [110, 180, self.H_UAV],
                    [220, 180, self.H_UAV],
                    [330, 180, self.H_UAV],
                    [440, 180, self.H_UAV],
                    [400, 400, self.H_UAV],
                ], dtype=np.float32)
            elif (
                self.regional_dynamic_md
                and self.hotspot_layout_mode in {
                    "episode_template4", "episode_template12"
                }
                and self.x_max_gu == 700
            ):
                if self.hotspot_layout_mode == "episode_template4":
                    # Rotation-neutral fixed reset for the four hidden layouts.
                    self.uav_positions = np.array([
                        [350, 350, self.H_UAV],
                        [150, 350, self.H_UAV],
                        [550, 350, self.H_UAV],
                        [350, 150, self.H_UAV],
                        [350, 550, self.H_UAV],
                    ], dtype=np.float32)
                else:
                    # Fixed asymmetric scout/service starts for the 12-layout
                    # coordination diagnostic.  ``staggered`` is an optional
                    # geometry ablation; the default keeps the lower four on
                    # one line so the R520 topology is unambiguous.
                    lower = (
                        [[100, 70], [200, 70], [300, 140], [400, 140]]
                        if self.five_uav_start_layout == "staggered"
                        else [[100, 70], [200, 70], [300, 70], [400, 70]]
                    )
                    base_xy = np.asarray(lower + [[450, 550]], dtype=np.float32)
                    self.uav_positions = np.column_stack((
                        base_xy,
                        np.full((self.n_UAVs, 1), self.H_UAV, dtype=np.float32),
                    ))
            else:
                upper_uav = (
                    [200, 525, self.H_UAV]
                    if self.regional_dynamic_md
                    else [400, 400, self.H_UAV]
                )
                self.uav_positions = np.array([
                    upper_uav,
                    [110, 70, self.H_UAV],
                    [220, 70, self.H_UAV],
                    [330, 70, self.H_UAV],
                    [440, 70, self.H_UAV],
                ], dtype=np.float32)
            if self.dynamic_md:
                self.x_min_all_gus = np.full(self.n_GUs, self.x_min_gu)
                self.x_max_all_gus = np.full(self.n_GUs, self.x_max_gu)
                self.y_min_all_gus = np.full(self.n_GUs, self.y_min_gu)
                self.y_max_all_gus = np.full(self.n_GUs, self.y_max_gu)
            elif self.n_GUs == 40:
                if self.fix_hotspot:
                    self.x_min_all_gus = np.array([0] * 7 + [200] * 33)
                    self.x_max_all_gus = np.array([175] * 7 + [600] * 33)
                    self.y_min_all_gus = np.array([0] * 7 + [200] * 33)
                    self.y_max_all_gus = np.array([175] * 7 + [600] * 33)
            elif self.n_GUs == 50:
                if self.fix_hotspot:
                    self.x_min_all_gus = np.array([0] * 9 + [200] * 41)
                    self.x_max_all_gus = np.array([175] * 9 + [600] * 41)
                    self.y_min_all_gus = np.array([0] * 9 + [200] * 41)
                    self.y_max_all_gus = np.array([175] * 9 + [600] * 41)
            elif self.n_GUs == 60:
                if self.fix_hotspot:
                    self.x_min_all_gus = np.array([0] * 10 + [200] * 50)
                    self.x_max_all_gus = np.array([175] * 10 + [600] * 50)
                    self.y_min_all_gus = np.array([0] * 10 + [200] * 50)
                    self.y_max_all_gus = np.array([175] * 10 + [600] * 50)
            elif self.n_GUs == 70:
                if self.fix_hotspot:
                    self.x_min_all_gus = np.array([0] * 12 + [200] * 58)
                    self.x_max_all_gus = np.array([175] * 12 + [600] * 58)
                    self.y_min_all_gus = np.array([0] * 12 + [200] * 58)
                    self.y_max_all_gus = np.array([175] * 12 + [600] * 58)
            elif self.n_GUs == 80:
                if self.fix_hotspot:
                    self.x_min_all_gus = np.array([0] * 14 + [200] * 66)
                    self.x_max_all_gus = np.array([175] * 14 + [600] * 66)
                    self.y_min_all_gus = np.array([0] * 14 + [200] * 66)
                    self.y_max_all_gus = np.array([175] * 14 + [600] * 66)
        # elif self.n_UAVs == 4 and self.x_min_gu == 0 and self.x_max_gu == 400 and self.n_GUs == 40:
        #     self.uav_positions = np.array([[100, 100, self.H_UAV], [300, 100, self.H_UAV], [100, 300, self.H_UAV],
        #                                    [300, 300, self.H_UAV]], dtype=np.float32)
            # # 添加上随机性。
            # np.random.shuffle(self.uav_positions)
        elif self.n_UAVs in [3, 4, 6, 7] and self.x_min_gu == 0 and self.x_max_gu == 600 and self.n_GUs == 60:
            if self.fix_hotspot:
                self.x_min_all_gus = np.array([0] * 10 + [200] * 50)
                self.x_max_all_gus = np.array([175] * 10 + [600] * 50)
                self.y_min_all_gus = np.array([0] * 10 + [200] * 50)
                self.y_max_all_gus = np.array([175] * 10 + [600] * 50)
            if self.n_UAVs == 3:
                self.uav_positions = np.array([[400, 400, self.H_UAV], [70, 70, self.H_UAV], [140, 70, self.H_UAV]], dtype=np.float32)
            elif self.n_UAVs == 4:
                self.uav_positions = np.array([[400, 400, self.H_UAV], [70, 70, self.H_UAV], [140, 70, self.H_UAV],
                                           [210, 70, self.H_UAV]], dtype=np.float32)
            elif self.n_UAVs == 6:
                self.uav_positions = np.array([[400, 400, self.H_UAV], [70, 70, self.H_UAV], [140, 70, self.H_UAV],
                                           [210, 70, self.H_UAV], [280, 70, self.H_UAV], [70, 140, self.H_UAV]], dtype=np.float32)
            elif self.n_UAVs == 7:
                self.uav_positions = np.array([[400, 400, self.H_UAV], [70, 70, self.H_UAV], [140, 70, self.H_UAV],
                                           [210, 70, self.H_UAV], [280, 70, self.H_UAV], [70, 140, self.H_UAV], [140, 140, self.H_UAV]], dtype=np.float32)
        elif self.n_UAVs == 4 and self.x_min_gu == 0 and self.x_max_gu == 300 and self.n_GUs == 40 and self.x_max_uav==600:
            self.uav_positions = np.array([[75, 150, self.H_UAV], [225, 150, self.H_UAV], [600, 500, self.H_UAV],
                                           [500, 600, self.H_UAV]], dtype=np.float32)
        elif self.n_UAVs == 9 and self.x_min_gu == 0 and self.x_max_gu == 650 and self.n_GUs == 100 and self.x_max_uav==650:
            # self.uav_positions = np.array([[110, 110, self.H_UAV], [330, 110, self.H_UAV], [550, 110, self.H_UAV],
            #          [110, 330, self.H_UAV], [330, 330, self.H_UAV], [550, 330, self.H_UAV],
            #          [110, 550, self.H_UAV], [330, 550, self.H_UAV], [550, 550, self.H_UAV]], dtype=np.float32)
            self.uav_positions = np.array([[50, 100, self.H_UAV], [100, 100, self.H_UAV], [150, 100, self.H_UAV],
                                           [200, 100, self.H_UAV], [250, 100, self.H_UAV], [300, 100, self.H_UAV],
                                           [350, 100, self.H_UAV], [400, 100, self.H_UAV], [450, 100, self.H_UAV]],
                                          dtype=np.float32)
            np.random.shuffle(self.uav_positions)
        elif self.n_UAVs == 16 and self.x_min_gu == 0 and self.x_max_gu == 900 and self.n_GUs == 200:
            self.uav_positions = np.array(
                    [[112.5, 112.5, self.H_UAV], [337.5, 112.5, self.H_UAV], [562.5, 112.5, self.H_UAV], [787.5, 112.5, self.H_UAV],
                     [112.5, 337.5, self.H_UAV], [337.5, 337.5, self.H_UAV], [562.5, 337.5, self.H_UAV], [787.5, 337.5, self.H_UAV],
                     [112.5, 562.5, self.H_UAV], [337.5, 562.5, self.H_UAV], [562.5, 562.5, self.H_UAV], [787.5, 562.5, self.H_UAV],
                     [112.5, 787.5, self.H_UAV],[337.5, 787.5, self.H_UAV], [562.5, 787.5, self.H_UAV], [787.5, 787.5, self.H_UAV]],
                    dtype=np.float32)
            # # 居中
            # self.uav_positions[:,:2] = self.uav_positions[:,:2]/2 + 225.5
            # # 四角
            # self.uav_positions = np.array([[ 56.25,  56.25, self.H_UAV], [168.75,  56.25, self.H_UAV], [ 56.25, 168.75, self.H_UAV], [168.75, 168.75, self.H_UAV],
            #                                [731.25,  56.25, self.H_UAV], [843.75,  56.25, self.H_UAV], [731.25, 168.75, self.H_UAV], [843.75, 168.75, self.H_UAV],
            #                                [ 56.25, 731.25, self.H_UAV], [168.75, 731.25, self.H_UAV], [ 56.25, 843.75, self.H_UAV], [168.75, 843.75, self.H_UAV],
            #                                [731.25, 731.25, self.H_UAV], [843.75, 731.25, self.H_UAV], [731.25, 843.75, self.H_UAV], [843.75, 843.75, self.H_UAV]], dtype=np.float32)
            # 添加上随机性。
            np.random.shuffle(self.uav_positions)
        elif self.n_UAVs == 16 and self.x_min_gu == 0 and self.x_max_gu == 850 and self.n_GUs == 200:
            self.uav_positions = np.array(
                [[110, 110, self.H_UAV], [330, 110, self.H_UAV], [550, 110, self.H_UAV], [770, 110, self.H_UAV],
                 [110, 330, self.H_UAV], [330, 330, self.H_UAV], [550, 330, self.H_UAV], [770, 330, self.H_UAV],
                 [110, 550, self.H_UAV], [330, 550, self.H_UAV], [550, 550, self.H_UAV], [770, 550, self.H_UAV],
                 [110, 770, self.H_UAV], [330, 770, self.H_UAV], [550, 770, self.H_UAV], [770, 770, self.H_UAV]],
                    dtype=np.float32)
            np.random.shuffle(self.uav_positions)
        elif self.n_UAVs == 25 and self.x_min_gu == 0 and self.x_max_gu == 900 and self.n_GUs == 200:
            self.uav_positions = np.array(
                    [[90, 90, self.H_UAV], [270, 90, self.H_UAV], [450, 90, self.H_UAV], [630, 90, self.H_UAV], [810, 90, self.H_UAV],
                     [90, 270, self.H_UAV], [270, 270, self.H_UAV], [450, 270, self.H_UAV], [630, 270, self.H_UAV], [810, 270, self.H_UAV],
                     [90, 450, self.H_UAV], [270, 450, self.H_UAV], [450, 450, self.H_UAV], [630, 450, self.H_UAV], [810, 450, self.H_UAV],
                     [90, 630, self.H_UAV], [270, 630, self.H_UAV], [450, 630, self.H_UAV], [630, 630, self.H_UAV], [810, 630, self.H_UAV],
                     [90, 810, self.H_UAV], [270, 810, self.H_UAV], [450, 810, self.H_UAV], [630, 810, self.H_UAV], [810, 810, self.H_UAV]], dtype=np.float32)
            # 添加上随机性。
            np.random.shuffle(self.uav_positions)

        elif self.n_UAVs in [5,10,15,20,25] and self.x_min_gu == 0 and self.x_max_gu==1000 and self.n_GUs==220:
            if self.n_UAVs == 5:
                self.uav_positions = np.array([[250, 250, self.H_UAV], [750, 250, self.H_UAV], [500, 500, self.H_UAV],
                                               [250, 750, self.H_UAV], [750, 750, self.H_UAV]], dtype=np.float32)
            elif self.n_UAVs == 10:
                self.uav_positions = np.array([[190, 190, self.H_UAV], [500, 190, self.H_UAV], [810, 190, self.H_UAV],
                                               [128, 500, self.H_UAV], [376, 500, self.H_UAV], [628, 500, self.H_UAV], [872, 500, self.H_UAV],
                                               [190, 810, self.H_UAV], [500, 810, self.H_UAV], [810, 810, self.H_UAV]], dtype=np.float32)
            elif self.n_UAVs == 15:
                self.uav_positions = np.array(
                    [[128, 128, self.H_UAV], [376, 128, self.H_UAV], [628, 128, self.H_UAV], [872, 128, self.H_UAV],
                     [128, 376, self.H_UAV], [376, 376, self.H_UAV], [628, 376, self.H_UAV], [872, 376, self.H_UAV],
                     [128, 628, self.H_UAV], [376, 628, self.H_UAV], [628, 628, self.H_UAV], [872, 628, self.H_UAV],
                     [190, 872, self.H_UAV],[500, 872, self.H_UAV], [810, 872, self.H_UAV]],
                    dtype=np.float32)
            elif self.n_UAVs == 20:
                self.uav_positions = np.array([[100, 128, self.H_UAV], [300, 128, self.H_UAV], [500, 128, self.H_UAV], [700, 128, self.H_UAV], [900, 128, self.H_UAV],
                                               [100, 376, self.H_UAV], [300, 376, self.H_UAV],[500, 376, self.H_UAV],[700, 376, self.H_UAV],[900, 376, self.H_UAV],
                                               [100, 628, self.H_UAV], [300, 628, self.H_UAV], [500, 628, self.H_UAV], [700, 628, self.H_UAV], [900, 628, self.H_UAV],
                                               [100, 872, self.H_UAV], [300, 872, self.H_UAV], [500, 872, self.H_UAV], [700, 872, self.H_UAV], [900, 872, self.H_UAV]],
                                              dtype=np.float32)
            elif self.n_UAVs == 25:
                self.uav_positions = np.array(
                    [[100, 100, self.H_UAV], [300, 100, self.H_UAV],[500, 100, self.H_UAV],[700, 100, self.H_UAV],[900, 100, self.H_UAV],
                     [100, 300, self.H_UAV], [300, 300, self.H_UAV], [500, 300, self.H_UAV], [700, 300, self.H_UAV], [900, 300, self.H_UAV],
                     [100, 500, self.H_UAV], [300, 500, self.H_UAV], [500, 500, self.H_UAV], [700, 500, self.H_UAV], [900, 500, self.H_UAV],
                     [100, 700, self.H_UAV], [300, 700, self.H_UAV], [500, 700, self.H_UAV], [700, 700, self.H_UAV], [900, 700, self.H_UAV],
                     [100, 900, self.H_UAV], [300, 900, self.H_UAV], [500, 900, self.H_UAV], [700, 900, self.H_UAV], [900, 900, self.H_UAV]],
                    dtype=np.float32)


        if self.uav_reset_curriculum:
            self.curriculum_random_probability = self._curriculum_random_reset_probability()
            self.curriculum_random_reset = (
                self.curriculum_reset_rng.random() < self.curriculum_random_probability
            )
            if self.curriculum_random_reset:
                self.uav_positions = self._sample_curriculum_uav_positions()
            self.curriculum_reset_count += 1

        self._set_episode_hotspot_layout()

        # self.uav_positions = np.array([[120, 120,self.H_UAV], [480, 120, self.H_UAV], [120, 480, self.H_UAV],[480, 480, self.H_UAV]], dtype=np.float32)
        # assert self.uav_positions.shape[0] == self.n_UAVs

        self.drones = [DroneInfo(i) for i in range(self.n_UAVs)]
        # 初始化时每个无人机知道所有其他无人机的初始位置
        for i, drone in enumerate(self.drones):
            for j in range(self.n_UAVs):
                drone.known_positions[j] = (self.uav_positions[j][:2].copy(), 0, 0 if i == j else 1)

        # np.random.seed(0)
        # # Initialize ground user positions with fixed height of 1m
        # self.gu_positions = np.random.uniform(0, self.x_max, (self.n_GUs, 2))
        # distances = np.sqrt(np.sum(self.gu_positions ** 2, axis=1))
        # sorted_indices = np.argsort(distances)
        # self.gu_positions = self.gu_positions[sorted_indices]
        # self.gu_positions = np.hstack((self.gu_positions, self.H_GU * np.ones((self.n_GUs, 1))))

        # gu_x = np.random.uniform(self.x_min_gu, self.x_max_gu, self.n_GUs)
        # gu_y = np.random.uniform(self.y_min_gu, self.y_max_gu, self.n_GUs)
        # if self.x_max_gu == 300 and self.n_GUs == 40:
        #     gu_x[-5:] =  np.random.uniform(100, 200, 5)
        #     gu_y[-5:] =  np.random.uniform(100, 200, 5)

        if self.dynamic_md:
            self.gu_positions = np.zeros((self.n_GUs, 3), dtype=np.float64)
            self.gu_positions[:, 2] = self.H_GU
            self.gu_velocities = np.zeros(self.n_GUs, dtype=np.float64)
            self.gu_directions = np.zeros(self.n_GUs, dtype=np.float64)
            self.gu_directions_0 = np.zeros(self.n_GUs, dtype=np.float64)
            self.active_md_mask = np.zeros(self.n_GUs, dtype=bool)
            self.md_remaining_lifetime = np.zeros(self.n_GUs, dtype=np.int32)
            self.md_session_ids = np.full(self.n_GUs, -1, dtype=np.int64)
            self.md_region_ids = np.full(self.n_GUs, -1, dtype=np.int8)
            self.next_md_session_id = 0
            self.dynamic_md_candidates = 0
            self.dynamic_md_admitted = 0
            self.dynamic_md_candidates_by_region = np.zeros(2, dtype=np.int64)
            self.dynamic_md_admitted_by_region = np.zeros(2, dtype=np.int64)
            self.dynamic_md_expired = 0
            self.dynamic_md_departed = 0
            self.dynamic_md_active_sum = 0
            self.dynamic_md_active_sum_second_half = 0
            self.hotspot_small_nearest_uav_distance_sum = 0.0
            self.hotspot_large_four_uav_distance_sum = 0.0
            self._admit_dynamic_mds()
        else:
            gu_x = self.x_min_all_gus + np.random.uniform(0, 1, self.n_GUs)*(self.x_max_all_gus - self.x_min_all_gus)
            gu_y = self.y_min_all_gus + np.random.uniform(0, 1, self.n_GUs)*(self.y_max_all_gus - self.y_min_all_gus)
            gu_z = np.full(self.n_GUs, self.H_GU)
            self.gu_positions = np.column_stack((gu_x, gu_y, gu_z))
            distances = np.linalg.norm(self.gu_positions, axis=1)
            sorted_indices = np.argsort(distances)
            self.gu_positions = self.gu_positions[sorted_indices]
            self.x_min_all_gus = self.x_min_all_gus[sorted_indices]
            self.x_max_all_gus = self.x_max_all_gus[sorted_indices]
            self.y_min_all_gus = self.y_min_all_gus[sorted_indices]
            self.y_max_all_gus = self.y_max_all_gus[sorted_indices]
            self.gu_velocities = self._sample_md_velocities(self.n_GUs)
            self.gu_directions = np.random.uniform(0, 2 * np.pi, self.n_GUs)
            self.active_md_mask = np.ones(self.n_GUs, dtype=bool)

        self.cartesian_proposal_norm_sum = 0.0
        self.cartesian_proposal_count = 0
        self.cartesian_projection_count = 0
        self.actor_message_neighbor_fraction_sum = 0.0
        self.actor_message_payload_norm_sum = 0.0
        self.actor_message_observation_count = 0

        # np.random.seed(None)
        # Initialize tasks for ground users
        self.gu_tasks = self.generate_tasks()

        # 预计算距离矩阵和其他辅助数据
        self._update_distance_matrices()
        self.nearby_gus_of_uavs = self.get_nearby_users_sorted_all()

        # 预计算信道增益
        self._calculate_channel_gains()

        # 在state_k_hops中，带自己的s_{i,t}，总共有max_UAVs_obs_concat个信息。 如果是全局拼接state的话，也不需要mask了。正好。
        self.attention_active_mask = np.zeros((self.n_UAVs, self.max_UAVs_obs_concat), dtype=np.float32)
        local_obs = self.get_local_obs()
        critic_local_obs = self.get_critic_local_obs(local_obs)
        # # 自己的obs直接就不变了。
        self.obs = local_obs

        self.state = self._build_critic_state(critic_local_obs)
        if self.use_atten_actor and self.state_is_k_hops:
            self.obs = self.state

        self.avail_actions = self.get_local_avail_actions()

        if self.average_neighbor_advantage:
            self.Metropolis_weights = self.get_neighbor_weights()
        else:
            self.Metropolis_weights = self.get_Metropolis_weights()
        self.cumulative_reward = np.zeros((self.n_agents,))
        self.system_performance = np.zeros((self.n_agents,))
        self.system_performance_individual = np.zeros((self.n_agents,))
        self.system_performance_true_all_GUs = np.zeros((self.n_agents,))
        self.system_performance_equivalent_full_GUs = np.zeros((self.n_agents,))
        self.system_performance_coverd_GUs = np.zeros((self.n_agents,))
        self.delay_true_all_GUs = np.zeros((self.n_agents,))
        self.delay_true_coverd_GUs = np.zeros((self.n_agents, 2))
        self.energy_true_all_GUs = np.zeros((self.n_agents,))
        self.energy_all_GUs_UAVs = np.zeros((self.n_agents,))
        self.n_GUs_by_coverd = 0
        self.complete_task_ratio = 0
        self.cumulative_individual_reward = np.zeros((self.n_agents,))
        self.uav_energy_consumption = np.zeros((self.n_UAVs,))  # 无人机及其范围内用户每时刻能耗和，再对step求和
        self.user_average_delay = np.zeros((self.n_UAVs,))  # 每架无人机范围内用户平均时延，再对step求平均
        self.n_GUs_per_uav_served = np.zeros((self.n_UAVs,))  # 每架无人机每个时刻服务的用户数目，再对step求平均
        self.cumulative_individual_reward_wo_cover = np.zeros((self.n_UAVs,))
        self.cumulative_reward_wo_cover = np.zeros((self.n_UAVs,))
        self.env_id = np.random.randn()
        if self.env_id:
            self.uav_positions_all_time[0] = self.uav_positions[:, :2]
            self.gu_positions_all_time[0, :, :2] = self.gu_positions[:, :2]
            self.gu_positions_all_time[0, :, 2] = self.gu_velocities
            self.gu_positions_all_time[0, :, 3] = self.gu_directions
        # 测试
        # if self.env_id in [0.2645712998311468, -0.39160536737987467, -0.05837604037964826, 0.2307092654537442, 1.428101796849161, -0.7050493998306726]:
        # if self.env_id == -0.7050493998306726:
        # if self.env_id == -0.020948109923857535:
        # if self.env_id == 0.8320153446702446:
        # # # # if self.env_id:
        #     if self.time_step % 4 == 0:
        #         self.render(timestep=self.time_step, title=str(np.round(self.env_id, 4))+'-'+'30901')
        #         # # # # # # self.render(timestep=self.time_step, title='28')  # 28是只有飞行动作，不能用上边的带process_actions的画图。后边也没用了，只跑了这一个，而且似乎有问题。
        #         time.sleep(0.05)
        return self.obs, self.state, self.avail_actions, self.Metropolis_weights, self.attention_active_mask

    def _update_distance_matrices(self):
        """更新距离矩阵"""
        # 计算UAV-UAV 2D距离
        self.uav_uav_distances_2d = np.linalg.norm(
            self.uav_positions[:, :2, np.newaxis] - self.uav_positions[:, :2].T[np.newaxis, :],
            axis=1
        )

        # 计算UAV-GU 2D和3D距离
        self.uav_gu_distances_2d = np.linalg.norm(
            self.uav_positions[:, :2, np.newaxis] - self.gu_positions[:, :2].T[np.newaxis, :],
            axis=1
        )

        self.uav_gu_distances_3d = np.linalg.norm(
            self.uav_positions[:, :3, np.newaxis] - self.gu_positions[:, :3].T[np.newaxis, :],
            axis=1
        )

        # 计算覆盖掩码
        self.coverage_mask = self.uav_gu_distances_2d <= self.Cover_R
        if self.dynamic_md:
            self.coverage_mask &= self.active_md_mask[np.newaxis, :]

    def _calculate_channel_gains(self):
        """计算信道增益"""
        # 计算角度
        with np.errstate(invalid='ignore'):  # 忽略可能的除零警告
            theta_nm = 180 / np.pi * np.arcsin(
                np.clip((self.H_UAV - self.H_GU) / self.uav_gu_distances_3d, -1, 1)
            )

        # 计算概率
        P_LoS = 1 / (1 + a * np.exp(-b * (theta_nm - a)))
        P_NLoS = 1 - P_LoS

        # 计算路径损耗
        PL_LoS = CONST_TERM_LoS + 20 * np.log10(np.maximum(self.uav_gu_distances_3d, 1e-10))
        PL_NLoS = CONST_TERM_NLoS + 20 * np.log10(np.maximum(self.uav_gu_distances_3d, 1e-10))

        PL_nm = P_LoS * PL_LoS + P_NLoS * PL_NLoS
        self.channel_gains = 10 ** (-PL_nm / 10)
        if self.dynamic_md:
            self.channel_gains[:, ~self.active_md_mask] = 0

    def process_actions(self, action):
        action = np.array(action, copy=True)
        if self.cartesian_flight and not self.fix_uav_pos:
            action[:, 2:] = np.clip(action[:, 2:], 0., 1.)
        else:
            action = np.clip(action, 0., 1.)
        # Convert action to a list with the specified slices
        if self.fix_uav_pos:
            assert self.continuous_associate
            actions = [
                action[:, :self.n_GUs],
                action[:, self.n_GUs:2 * self.n_GUs],
                action[:, 2 * self.n_GUs:]
            ]
        else:
            if self.nearest_associate:
                if self.ave_resource:
                    actions = [
                        action[:, :2]
                    ]
                elif self.ave_bandwidth:
                    actions = [
                        action[:, :2],
                        action[:, 2:2 + self.n_GUs]
                    ]
                else:
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
                elif self.ave_bandwidth:
                    actions = [
                        action[:, :2],
                        action[:, 2:2 + self.n_GUs],
                        action[:, 2 + self.n_GUs:]
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
                actions[0] = np.where(actions[0] >= 0.5, actions[0], 0)
                col_max = np.max(actions[0], axis=0)
                # 为了防止最大值为0的存在，导致取到多个不存在的无人机。
                col_max = np.where(col_max, col_max, 99.0)
                actions[0] = (actions[0] == col_max).astype(int)
                for n in range(self.n_GUs):
                    user_selection = actions[0][:, n].copy()
                    bandwidth_allocation = actions[1][:, n].copy()
                    computation_allocation = actions[2][:, n].copy()
                    sum_selection = np.sum(user_selection)
                    if sum_selection == 0:
                        actions[0][:, n] = 0
                        actions[1][:, n] = 0
                        actions[2][:, n] = 0
                    elif sum_selection == 1:
                        best_uav = np.argmax(actions[0][:, n])  # UAV m is offloading task n
                        actions[0][:, n] = 0
                        actions[0][best_uav, n] = 1
                        actions[1][:, n] = 0
                        actions[1][best_uav, n] = bandwidth_allocation[best_uav]
                        actions[2][:, n] = 0
                        actions[2][best_uav, n] = computation_allocation[best_uav]
                    else:
                        active_mask = user_selection == 1
                        distances = self.uav_gu_distances_3d[:, n]
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

                # 标准化资源
                for m in range(self.n_UAVs):
                    total_bw = np.sum(actions[1][m])
                    total_comp = np.sum(actions[2][m])

                    if total_bw > 0:
                        actions[1][m] = actions[1][m] / total_bw

                    if total_comp > 0:
                        actions[2][m] = actions[2][m] / total_comp
                # 计算实际带宽和计算资源
                bandwidth_actions = actions[1] * self.B
                computation_actions = actions[2] * self.F_m

                # 验证每个用户任务
                for n in range(self.n_GUs):
                    if np.any(actions[0][:, n]):
                        m = np.argmax(actions[0][:, n])
                        task = self.gu_tasks[n]

                        # 计算传输速率
                        h_nm = self.channel_gains[m, n]
                        bw = bandwidth_actions[m, n]
                        R_nm = bw * np.log2(1 + p_t * h_nm / (N_0 * (bw + 1e-10)))

                        # 计算时延
                        tau_trans = task[0] / (R_nm + 1e-10)
                        tau_exe = task[1] / (computation_actions[m, n] + 1e-10)
                        total_delay = tau_trans + tau_exe

                        # 如果无法满足延迟要求，取消分配
                        if total_delay > task[2]:
                            actions[0][m, n] = 0
                            actions[1][m, n] = 0
                            actions[2][m, n] = 0

                # 重新标准化资源
                for m in range(self.n_UAVs):
                    total_bw = np.sum(actions[1][m])
                    total_comp = np.sum(actions[2][m])

                    if total_bw > 0:
                        actions[1][m] = actions[1][m] / total_bw

                    if total_comp > 0:
                        actions[2][m] = actions[2][m] / total_comp

            else:
                if self.ave_resource:
                    actions[1] = np.where(actions[1]>=0.5, actions[1], 0)   # 添加了卸载的阈值为0.5。
                    col_max = np.max(actions[1], axis=0)
                    # 为了防止最大值为0的存在，导致取到多个不存在的无人机。
                    col_max = np.where(col_max, col_max, 99.0)
                    actions[1] = (actions[1] == col_max).astype(int)
                    for n in range(self.n_GUs):
                        user_selection = actions[1][:, n].copy()
                        sum_selection = np.sum(user_selection)
                        if sum_selection == 0:
                            actions[1][:, n] = 0
                        elif sum_selection == 1:
                            best_uav = np.argmax(actions[1][:, n])  # UAV m is offloading task n
                            actions[1][:, n] = 0
                            actions[1][best_uav, n] = 1
                        else:
                            # 如果多个UAV选择同一用户，只保留最近的
                            active_mask = user_selection == 1
                            distances = self.uav_gu_distances_3d[:, n]
                            distances_masked = np.where(active_mask, distances, np.inf)
                            nearest_uav_idx = np.argmin(distances_masked)

                            actions[1][:, n] = 0
                            actions[1][nearest_uav_idx, n] = 1
                elif self.ave_bandwidth:
                    mask_2 = actions[2] == 0
                    actions[1][mask_2] = 0
                    actions[1] = np.where(actions[1] >= 0.5, actions[1], 0)  # 添加了卸载的阈值为0.5。
                    col_max = np.max(actions[1], axis=0)
                    # 为了防止最大值为0的存在，导致取到多个不存在的无人机。把0变为不可能出现的99
                    col_max = np.where(col_max, col_max, 99.0)
                    actions[1] = (actions[1] == col_max).astype(int)
                    for n in range(self.n_GUs):
                        user_selection = actions[1][:, n].copy()
                        computation_allocation = actions[2][:, n].copy()
                        sum_selection = np.sum(user_selection)
                        if sum_selection == 0:
                            actions[1][:, n] = 0
                            actions[2][:, n] = 0
                        elif sum_selection == 1:
                            best_uav = np.argmax(actions[1][:, n])  # UAV m is offloading task n
                            actions[1][:, n] = 0
                            actions[1][best_uav, n] = 1
                            actions[2][:, n] = 0
                            actions[2][best_uav, n] = computation_allocation[best_uav]
                        else:
                            active_mask = user_selection == 1
                            distances = self.uav_gu_distances_3d[:, n]
                            # 只考虑活跃无人机的距离，非活跃无人机距离设为无穷大
                            distances_masked = np.where(active_mask, distances, np.inf)
                            # 找到距离最近的无人机索引
                            nearest_uav_idx = np.argmin(distances_masked)
                            actions[1][:, n] = 0
                            actions[1][nearest_uav_idx, n] = 1
                            actions[2][:, n] = 0
                            actions[2][nearest_uav_idx, n] = computation_allocation[nearest_uav_idx]
                    for m in range(self.n_UAVs):
                        total = np.sum(actions[2][m])
                        if total > 0:
                            actions[2][m] = actions[2][m] / total
                    # 判断卸载分配的资源能不能完成任务..
                    ones_count = np.sum(actions[1], axis=1, keepdims=True)  # 避免除零，使用np.divide处理
                    bandwidth_actions = np.divide(actions[1] * self.B, ones_count, where=ones_count != 0)
                    computation_actions = actions[2] * self.F_m
                    for n in range(self.n_GUs):
                        if np.sum(actions[1][:, n]) != 0:
                            m = np.argmax(actions[1][:, n])
                            gu_n_task = self.gu_tasks[n]

                            # 计算传输速率
                            h_nm = self.channel_gains[m, n]
                            bw = bandwidth_actions[m, n]
                            R_nm = bw * np.log2(1 + p_t * h_nm / (N_0 * (bw + 1e-10)))

                            epsilon = 0
                            tau_trans = gu_n_task[0] / (R_nm + epsilon)
                            tau_exe = gu_n_task[1] / (computation_actions[m, n] + epsilon)
                            total_delay = tau_trans + tau_exe
                            if total_delay > self.gu_tasks[n, 2]:

                            # if gu_n_task[1] / computation_actions[m, n] > self.gu_tasks[n, 2]:

                                actions[1][m, n] = 0
                                actions[2][m, n] = 0
                    for m in range(self.n_UAVs):
                        total = np.sum(actions[2][m])
                        if total > 0:
                            actions[2][m] = actions[2][m] / total
                else:
                    mask_2 = actions[2] == 0
                    mask_3 = actions[3] == 0
                    combined_mask = mask_2 | mask_3
                    actions[1][combined_mask] = 0
                    actions[1] = np.where(actions[1] >= 0.5, actions[1], 0)  # 添加了卸载的阈值为0.5。
                    col_max = np.max(actions[1], axis=0)
                    # 为了防止最大值为0的存在，导致取到多个不存在的无人机。把0变为不可能出现的99
                    col_max = np.where(col_max, col_max, 99.0)
                    actions[1] = (actions[1] == col_max).astype(int)
                    selection_counts = np.sum(actions[1], axis=0)
                    duplicate_gu_ids = np.flatnonzero(selection_counts > 1)
                    if duplicate_gu_ids.size:
                        active_distances = np.where(
                            actions[1][:, duplicate_gu_ids] == 1,
                            self.uav_gu_distances_3d[:, duplicate_gu_ids],
                            np.inf,
                        )
                        nearest_uav_ids = np.argmin(active_distances, axis=0)
                        actions[1][:, duplicate_gu_ids] = 0
                        actions[1][nearest_uav_ids, duplicate_gu_ids] = 1

                    selected_mask = actions[1].astype(bool)
                    actions[2] *= selected_mask
                    actions[3] *= selected_mask
                    # 标准化资源
                    for m in range(self.n_UAVs):
                        total_bw = np.sum(actions[2][m])
                        total_comp = np.sum(actions[3][m])

                        if total_bw > 0:
                            actions[2][m] = actions[2][m] / total_bw

                        if total_comp > 0:
                            actions[3][m] = actions[3][m] / total_comp
                    # 判断卸载分配的资源能不能完成任务..
                    bandwidth_actions = actions[2] * self.B
                    computation_actions = actions[3] * self.F_m
                    selected_gu_ids = np.flatnonzero(np.any(actions[1], axis=0))
                    if selected_gu_ids.size:
                        selected_uav_ids = np.argmax(actions[1][:, selected_gu_ids], axis=0)
                        tau_trans, tau_exe = _offload_delay(
                            self.gu_tasks,
                            selected_uav_ids,
                            selected_gu_ids,
                            bandwidth_actions,
                            computation_actions,
                            self.channel_gains,
                        )
                        overdue = tau_trans + tau_exe > self.gu_tasks[selected_gu_ids, 2]
                        overdue_gu_ids = selected_gu_ids[overdue]
                        overdue_uav_ids = selected_uav_ids[overdue]
                        actions[1][overdue_uav_ids, overdue_gu_ids] = 0
                        actions[2][overdue_uav_ids, overdue_gu_ids] = 0
                        actions[3][overdue_uav_ids, overdue_gu_ids] = 0
                    # 重新标准化资源
                    for m in range(self.n_UAVs):
                        total_bw = np.sum(actions[2][m])
                        total_comp = np.sum(actions[3][m])

                        if total_bw > 0:
                            actions[2][m] = actions[2][m] / total_bw

                        if total_comp > 0:
                            actions[3][m] = actions[3][m] / total_comp
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
                if self.ave_resource:
                    pass
                elif self.ave_bandwidth:
                    for n in range(self.n_GUs):
                        active_mask = actions[1][:, n] > 0
                        computation_allocation = actions[1][:, n].copy()
                        if np.sum(active_mask) == 0:
                            actions[1][:, n] = 0
                        elif np.sum(active_mask) == 1:
                            best_uav = np.argmax(active_mask)  # UAV m is offloading task n
                            actions[1][:, n] = 0
                            actions[1][best_uav, n] = computation_allocation[best_uav]
                        else:
                            distances = np.linalg.norm(self.uav_positions - self.gu_positions[n], axis=1)
                            # 只考虑活跃无人机的距离，非活跃无人机距离设为无穷大
                            distances_masked = np.where(active_mask, distances, np.inf)
                            # 找到距离最近的无人机索引
                            nearest_uav_idx = np.argmin(distances_masked)
                            actions[1][:, n] = 0
                            actions[1][nearest_uav_idx, n] = computation_allocation[nearest_uav_idx]
                    for m in range(self.n_UAVs):
                        total = np.sum(actions[1][m])
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
                        if total > 1:
                            actions[1][m] = actions[1][m] / total
                    for m in range(self.n_UAVs):
                        total = np.sum(actions[2][m])
                        if total > 1:
                            actions[2][m] = actions[2][m] / total
        if actions[0].shape[1] == 2:
            self.proposed_offload_actions = actions[1]
        else:
            self.proposed_offload_actions = actions[0]
        self.proposed_offload_actions_all_time[self.time_step] = self.proposed_offload_actions
        return np.concatenate(actions, axis=-1)

    def step(self, action):   # acts是一个(n_agents, )的数组，每个元素代表动作序号
        self.time_step += 1
        # Ensure action is a numpy array
        assert isinstance(action, (list, tuple, np.ndarray)), "Action must be a list or tuple or numpy array"
        self.complete_task = np.zeros((self.n_GUs,))
        self.self_complete_task = np.zeros((self.n_GUs,))

        # 计算local奖励时，先用了动作转换，用到了nearby_gus_of_uavs，但是不用先更新。因为就是用之前的信息，找到对应之前的动作。用来计算奖励。
        # 在计算奖励时也修改了self.complete_task标记。供可视化画图。
        if self.not_process_action:
            rewards = self.calculate_local_reward_raw_action(action)
        else:
            rewards = self.calculate_local_reward(action)
        self.cumulative_reward += np.mean(rewards) * np.ones_like(rewards)

        # 更新无人机位置放到了calculate_reward()里边

        # 更新地面用户位置，放到了calculate_reward()里边

        # Generate new tasks for ground users
        self.gu_tasks = self.generate_tasks()

        # 更新距离矩阵和覆盖信息
        # self._update_distance_matrices()  # 计算reward时，更新完位置后直接更新了。
        self._calculate_channel_gains()
        self.nearby_gus_of_uavs = self.get_nearby_users_sorted_all()

        self.attention_active_mask = np.zeros((self.n_UAVs, self.max_UAVs_obs_concat), dtype=np.float32)
        local_obs = self.get_local_obs()
        critic_local_obs = self.get_critic_local_obs(local_obs)
        # # 自己的obs直接就不变了。
        self.obs = local_obs

        self.state = self._build_critic_state(critic_local_obs)
        if self.use_atten_actor and self.state_is_k_hops:
            self.obs = self.state

        self.avail_actions = self.get_local_avail_actions()

        dones = np.asarray([False]*self.n_agents)
        info = {}
        if self.communication_mode == "unreliable":
            info["state_timely_reception_rate"] = (
                self.state_packets_received / max(self.state_packets_attempted, 1)
            )
        if self.average_neighbor_advantage:
            self.Metropolis_weights = self.get_neighbor_weights()
        else:
            self.Metropolis_weights = self.get_Metropolis_weights()
        # # if self.env_id in [0.2645712998311468, -0.39160536737987467, -0.05837604037964826, 0.2307092654537442, 1.428101796849161, -0.7050493998306726]:
        # #     if self.time_step % 5 == 0:
        # #         self.render(timestep=self.time_step, title=str(np.round(self.env_id, 4)) + '-' + '185', acts=self.proposed_offload_actions)
        # #         # self.render(timestep=self.time_step, title='28')  # 28是只有飞行动作，不能用上边的带process_actions的画图。后边也没用了，只跑了这一个，而且似乎有问题。
        # #         time.sleep(0.05)
        if self.env_id:
            self.uav_positions_all_time[self.time_step] = self.uav_positions[:, :2]
            self.gu_positions_all_time[self.time_step, :, :2] = self.gu_positions[:, :2]
            self.gu_positions_all_time[self.time_step, :, 2] = self.gu_velocities
            self.gu_positions_all_time[self.time_step, :, 3] = self.gu_directions
        # 测试
        # if self.env_id == -0.7050493998306726:
        # if self.env_id == -0.020948109923857535:
        # if self.env_id == 0.6345626069494656:
        # if self.env_id == 0.8320153446702446:
        # # # # if self.env_id:
        #     if (self.time_step % 4 == 0) or (self.time_step in [185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215]):
        #         self.render(timestep=self.time_step, title=str(np.round(self.env_id, 4)) + '-' + '30901',
        #                     acts=self.proposed_offload_actions)
        #         time.sleep(0.05)
        # if self.time_step % 5 == 0:
        #     self.render(timestep=self.time_step, title='301', acts=self.proposed_offload_actions)
        #     # self.render(timestep=self.time_step, title='28')  # 28是只有飞行动作，不能用上边的带process_actions的画图。后边也没用了，只跑了这一个，而且似乎有问题。
        #     time.sleep(0.05)
        if self.time_step >= self.MAX_SIMULATION_TIME:
            # 测试
            # if self.env_id == -0.7050493998306726:
            #     np.save('-0.705uav_positions_all_time' + '.npy', self.uav_positions_all_time)
            #     np.save('-0.705gu_positions_all_time' + '.npy', self.gu_positions_all_time)
            #     np.save('-0.705system_performance_all_uavs' + '.npy', self.system_performance_all_uavs)
            #     np.save('-0.705proposed_offload_actions_all_time' + '.npy', self.proposed_offload_actions_all_time)
            #     print('save successfully.')
            # if self.env_id == -0.020948109923857535:
            #     np.save('-0.0209uav_positions_all_time' + '.npy', self.uav_positions_all_time)
            #     np.save('-0.0209gu_positions_all_time' + '.npy', self.gu_positions_all_time)
            #     np.save('-0.0209proposed_offload_actions_all_time' + '.npy', self.proposed_offload_actions_all_time)
            #     print('save successfully.')
            # if self.env_id == 0.8320153446702446:
            #     np.save('-0.8320uav_positions_all_time' + '.npy', self.uav_positions_all_time)
            #     np.save('-0.8320gu_positions_all_time' + '.npy', self.gu_positions_all_time)
            #     np.save('-0.8320proposed_offload_actions_all_time' + '.npy', self.proposed_offload_actions_all_time)
            #     print('save successfully.')
            dones = 1 - dones
            info = {'cumulative_reward': self.cumulative_reward, 'n_GUs_per_uav_served': self.n_GUs_per_uav_served/self.MAX_SIMULATION_TIME,
                    'uav_m_toal_energy_consumption': self.uav_energy_consumption, 'user_in_m_average_delay': self.user_average_delay/self.MAX_SIMULATION_TIME,
                    'cumulative_individual_reward': self.cumulative_individual_reward, 'system_performance': self.system_performance,
                    'cumulative_individual_reward_wo_cover': self.cumulative_individual_reward_wo_cover,
                    'cumulative_reward_wo_cover': self.cumulative_reward_wo_cover,
                    'system_performance_individual': self.system_performance_individual,
                    'system_performance_true_all_GUs': self.system_performance_true_all_GUs,
                    'system_performance_equivalent_full_GUs': self.system_performance_equivalent_full_GUs,
                    'delay_true_all_GUs':self.delay_true_all_GUs/self.MAX_SIMULATION_TIME,
                    'delay_true_coverd_GUs':self.delay_true_coverd_GUs[:, 0] / self.delay_true_coverd_GUs[:, 1] /self.MAX_SIMULATION_TIME,
                    'energy_true_all_GUs':self.energy_true_all_GUs/self.MAX_SIMULATION_TIME,
                    'energy_all_GUs_UAVs':self.energy_all_GUs_UAVs/self.MAX_SIMULATION_TIME,
                    'system_performance_coverd_GUs': self.system_performance_coverd_GUs,
                    'n_GUs_by_coverd':self.n_GUs_by_coverd/(self.MAX_SIMULATION_TIME/2),
                     'complete_task_ratio':self.complete_task_ratio/self.MAX_SIMULATION_TIME,
                     'uav_positions': self.uav_positions[:, :2],
                     }
            if self.communication_mode == "unreliable":
                info["state_timely_reception_rate"] = (
                    self.state_packets_received / max(self.state_packets_attempted, 1)
                )
            if self.dynamic_md:
                info.update({
                    'candidate_md_arrivals': self.dynamic_md_candidates,
                    'admitted_md_arrivals': self.dynamic_md_admitted,
                    'candidate_md_arrivals_small_region': self.dynamic_md_candidates_by_region[0],
                    'candidate_md_arrivals_large_region': self.dynamic_md_candidates_by_region[1],
                    'admitted_md_arrivals_small_region': self.dynamic_md_admitted_by_region[0],
                    'admitted_md_arrivals_large_region': self.dynamic_md_admitted_by_region[1],
                    'md_admission_ratio_small_region': (
                        self.dynamic_md_admitted_by_region[0]
                        / max(self.dynamic_md_candidates_by_region[0], 1)
                    ),
                    'md_admission_ratio_large_region': (
                        self.dynamic_md_admitted_by_region[1]
                        / max(self.dynamic_md_candidates_by_region[1], 1)
                    ),
                    **{
                        f'hotspot_layout_{layout_id}_fraction': float(
                            self.hotspot_layout_index == layout_id
                        )
                        for layout_id in range(
                            12 if self.hotspot_layout_mode in {
                                "episode_template12", "episode_template12_600_200"
                            } else 4
                        )
                    },
                    'md_admission_ratio': self.dynamic_md_admitted / max(self.dynamic_md_candidates, 1),
                    'expired_md_accesses': self.dynamic_md_expired,
                    'coverage_departed_md_accesses': self.dynamic_md_departed,
                    'average_active_mds': self.dynamic_md_active_sum / self.MAX_SIMULATION_TIME,
                    'average_active_mds_second_half': (
                        self.dynamic_md_active_sum_second_half
                        / max(self.MAX_SIMULATION_TIME / 2, 1)
                    ),
                })
                if self.hotspot_layout_mode == "episode_moving_template4":
                    info.update({
                        'moving_hotspot_small_nearest_uav_distance': (
                            self.hotspot_small_nearest_uav_distance_sum
                            / self.MAX_SIMULATION_TIME
                        ),
                        'moving_hotspot_large_mean_four_uav_distance': (
                            self.hotspot_large_four_uav_distance_sum
                            / self.MAX_SIMULATION_TIME
                        ),
                    })
                if self.hotspot_layout_mode == "fixed_legacy":
                    info.update({
                        'candidate_md_arrivals_lower_left': self.dynamic_md_candidates_by_region[0],
                        'candidate_md_arrivals_upper_right': self.dynamic_md_candidates_by_region[1],
                        'admitted_md_arrivals_lower_left': self.dynamic_md_admitted_by_region[0],
                        'admitted_md_arrivals_upper_right': self.dynamic_md_admitted_by_region[1],
                        'md_admission_ratio_lower_left': (
                            self.dynamic_md_admitted_by_region[0]
                            / max(self.dynamic_md_candidates_by_region[0], 1)
                        ),
                        'md_admission_ratio_upper_right': (
                            self.dynamic_md_admitted_by_region[1]
                            / max(self.dynamic_md_candidates_by_region[1], 1)
                        ),
                    })
                if getattr(self.args, "uav_reset_curriculum", False):
                    info.update({
                        'curriculum_random_reset': float(self.curriculum_random_reset),
                        'curriculum_random_probability': self.curriculum_random_probability,
                    })
            if self.cartesian_flight:
                info.update({
                    'flight_proposal_norm_mean': (
                        self.cartesian_proposal_norm_sum
                        / max(self.cartesian_proposal_count, 1)
                    ),
                    'flight_disk_projection_ratio': (
                        self.cartesian_projection_count
                        / max(self.cartesian_proposal_count, 1)
                    ),
                })
            if self.actor_message_mode != "disabled":
                info.update({
                    'actor_message_neighbor_fraction': (
                        self.actor_message_neighbor_fraction_sum
                        / max(self.actor_message_observation_count, 1)
                    ),
                    'actor_message_payload_norm': (
                        self.actor_message_payload_norm_sum
                        / max(self.actor_message_observation_count, 1)
                    ),
                })
        return self.obs, rewards, dones, self.state, self.avail_actions, info, self.Metropolis_weights, self.attention_active_mask

    def calculate_local_reward_raw_action(self, action):
        # 转到全局n_GUs的动作，然后处理动作。还是n_GUs的直接放到calculate_reward
        # if self.not_process_action:这个是我之前用来测试，在buffer保存raw-actions会怎么样。
        # 不处理动作，需要先对动作合法化，然后去计算奖励。if self.nearest_associate: and if self.ave_resource:了。即只有飞行动作。就不需要再合法化动作了。
        if self.nearest_associate and self.ave_resource:
            rewards = self.calculate_reward(action)
        else:
            transformed_action_components = self.transform_uav_actions(action)
            processed_actions = self.process_actions(transformed_action_components)
            rewards = self.calculate_reward(processed_actions)
        return rewards

    def calculate_reward(self, action):
        reward_active_mask = self.active_md_mask.copy() if self.dynamic_md else np.ones(self.n_GUs, dtype=bool)
        reward_active_ids = np.flatnonzero(reward_active_mask)
        reward_active_count = len(reward_active_ids)
        if self.fix_uav_pos:
            assert self.continuous_associate
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
                if self.ave_resource:
                    fly_actions = action[:, :2]
                    uav_gu_distances_2d = np.linalg.norm(self.uav_positions[:, None, :2] - self.gu_positions[None, :, :2], axis=2)
                    coverage_mask = uav_gu_distances_2d <= self.Cover_R
                    if self.dynamic_md:
                        coverage_mask &= reward_active_mask[np.newaxis, :]
                    uav_gu_distances_2d_min = np.min(uav_gu_distances_2d, axis=0)
                    offloading_actions = ((uav_gu_distances_2d==uav_gu_distances_2d_min) & (coverage_mask)).astype(int)
                    ones_count = np.sum(offloading_actions, axis=1, keepdims=True)  # 避免除零，使用np.divide处理
                    bandwidth_actions = np.divide(offloading_actions * self.B, ones_count, where=ones_count != 0)
                    ones_count = np.sum(offloading_actions, axis=1, keepdims=True)  # 避免除零，使用np.divide处理
                    computation_actions = np.divide(offloading_actions * self.F_m, ones_count, where=ones_count != 0)
                elif self.ave_bandwidth:
                    raise ValueError("最近关联的情况下，没有写仅平均带宽。")  # 主动抛出异常
                else:
                    action_components = [
                        action[:, :2],  # Movement actions
                        action[:, 2:2 + self.n_GUs],  # Bandwidth decisions
                        action[:, 2 + self.n_GUs:],  # computation allocation
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
                    computation_actions = np.divide(action_components[1] * self.F_m, ones_count, where=ones_count != 0)
                elif self.ave_bandwidth:
                    action_components = [
                        action[:, :2],  # Movement actions
                        action[:, 2:2 + self.n_GUs],  # Offloading decisions
                        action[:, 2 + self.n_GUs:]  # Computation resource allocation
                    ]
                    fly_actions = action_components[0]
                    offloading_actions = action_components[1]
                    ones_count = np.sum(action_components[1], axis=1, keepdims=True)  # 避免除零，使用np.divide处理
                    bandwidth_actions = np.divide(action_components[1] * self.B, ones_count, where=ones_count != 0)
                    computation_actions = action_components[2] * self.F_m
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
        if not self.nearest_associate:
            offloading_actions = np.where(
                (bandwidth_actions > 0) & (computation_actions > 0),
                offloading_actions,
                0,
            )
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

        # 使用缓存的距离和掩码
        uav_gu_distances_2d = self.uav_gu_distances_2d
        coverage_mask = self.coverage_mask

        R_task = np.zeros(self.n_UAVs)
        R_task_delay = np.zeros(self.n_UAVs)
        R_task_energy = np.zeros(self.n_UAVs)
        use_vectorized_task_rewards = (
            self.continuous_associate
            and not self.fix_uav_pos
            and not self.nearest_associate
            and not self.ave_resource
            and not self.ave_bandwidth
            and self.not_served_rew_to_nearest
        )
        if use_vectorized_task_rewards:
            offloaded_mask = np.any(offloading_actions, axis=0)
            local_gu_ids = reward_active_ids[~offloaded_mask[reward_active_ids]]
            offloaded_gu_ids = reward_active_ids[offloaded_mask[reward_active_ids]]
            credited_uav_ids = np.full(self.n_GUs, -1, dtype=int)
            credited_delay_rewards = np.zeros(self.n_GUs)
            credited_energy_rewards = np.zeros(self.n_GUs)

            if local_gu_ids.size:
                local_tasks = self.gu_tasks[local_gu_ids]
                local_delay = local_tasks[:, 1] / self.F_n
                local_energy = k_local * (self.F_n ** 2) * local_tasks[:, 1]
                local_energy_clipped = np.clip(local_energy, 0, 10)
                local_complete = local_tasks[:, 2] > local_delay
                local_delay_reward = np.where(
                    local_complete,
                    self.gamma_r * (1 + local_tasks[:, 2] - local_delay),
                    -self.delta_r,
                )
                local_energy_reward = -self.lambda_r * local_energy_clipped

                self.complete_task[local_gu_ids] = local_complete
                self.self_complete_task[local_gu_ids] = local_complete
                per_GU_delay_reward_others[local_gu_ids] = local_delay_reward
                per_GU_energy_reward_others[local_gu_ids] = local_energy_reward
                per_GU_task_reward_others[local_gu_ids] = local_delay_reward + local_energy_reward
                per_GU_delay_true_others[local_gu_ids] = local_delay
                per_GU_energy_true_others[local_gu_ids] = local_energy_clipped

                covered_local = np.any(coverage_mask[:, local_gu_ids], axis=0)
                covered_gu_ids = local_gu_ids[covered_local]
                if covered_gu_ids.size:
                    nearest_uav_ids = np.argmin(uav_gu_distances_2d[:, covered_gu_ids], axis=0)
                    credited_uav_ids[covered_gu_ids] = nearest_uav_ids
                    credited_delay_rewards[covered_gu_ids] = local_delay_reward[covered_local]
                    credited_energy_rewards[covered_gu_ids] = local_energy_reward[covered_local]

            if offloaded_gu_ids.size:
                serving_uav_ids = np.argmax(offloading_actions[:, offloaded_gu_ids], axis=0)
                transmission_delay, execution_delay = _offload_delay(
                    self.gu_tasks,
                    serving_uav_ids,
                    offloaded_gu_ids,
                    bandwidth_actions,
                    computation_actions,
                    self.channel_gains,
                )
                offloaded_tasks = self.gu_tasks[offloaded_gu_ids]
                total_delay = transmission_delay + execution_delay
                allocated_computation = computation_actions[serving_uav_ids, offloaded_gu_ids]
                total_energy = (
                    p_t * transmission_delay
                    + k_server * allocated_computation ** 2 * offloaded_tasks[:, 1]
                )
                energy_clipped = np.clip(total_energy, 0, 10)
                complete = offloaded_tasks[:, 2] > total_delay
                delay_reward = np.where(
                    complete,
                    self.gamma_r * (1 + offloaded_tasks[:, 2] - total_delay),
                    -self.delta_r,
                )
                energy_reward = -self.lambda_r * energy_clipped

                self.complete_task[offloaded_gu_ids] = complete
                self.self_complete_task[offloaded_gu_ids] = (
                    offloaded_tasks[:, 2] > offloaded_tasks[:, 1] / self.F_n
                )
                per_GU_delay_reward[offloaded_gu_ids] = delay_reward
                per_GU_energy_reward[offloaded_gu_ids] = energy_reward
                per_GU_task_reward[offloaded_gu_ids] = delay_reward + energy_reward
                per_GU_delay_true[offloaded_gu_ids] = total_delay
                per_GU_energy_true[offloaded_gu_ids] = energy_clipped
                credited_uav_ids[offloaded_gu_ids] = serving_uav_ids
                credited_delay_rewards[offloaded_gu_ids] = delay_reward
                credited_energy_rewards[offloaded_gu_ids] = energy_reward

            for gu_id in np.flatnonzero(credited_uav_ids >= 0):
                uav_id = credited_uav_ids[gu_id]
                R_task_delay[uav_id] += credited_delay_rewards[gu_id]
                R_task_energy[uav_id] += credited_energy_rewards[gu_id]

            task_gu_ids = ()
        else:
            task_gu_ids = reward_active_ids

        for n in task_gu_ids:
            uav_indices = np.where(coverage_mask[:, n])[0]
            if self.nearest_associate and (self.ave_resource is False): # nearest_associate的还有问题呢。
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
                    per_GU_delay_reward_others[n] = self.gamma_r * (1+self.gu_tasks[n, 2] - total_delay)
                    self.complete_task[n] = 1
                    self.self_complete_task[n] = 1
                else:
                    per_GU_delay_reward_others[n] = -1 * self.delta_r
                    self.complete_task[n] = 0
                    self.self_complete_task[n] = 0
                per_GU_delay_true_others[n] = total_delay
                per_GU_energy_reward_others[n] = -1 * self.lambda_r * np.clip(total_energy, 0, 10)
                per_GU_energy_true_others[n] = np.clip(total_energy, 0, 10)
                # 这里10，自己加的规定，由于大于1才重新分配动作，某些很少的资源导致计算的时延和能量巨大！ 通常情况下仅为10以内（其实看到的最大只有1.8）。
                per_GU_task_reward_others[n] = per_GU_delay_reward_others[n] + per_GU_energy_reward_others[n]

                if self.not_served_rew_to_ave:
                    # 未被服务的用户奖励平分给覆盖的无人机。
                    if len(uav_indices) > 0:
                        R_task_delay[uav_indices] += self.gamma_r * (1 + self.gu_tasks[n, 2] - total_delay) / len(uav_indices) if self.gu_tasks[n, 2] > total_delay else -1 * self.delta_r / len(uav_indices)
                        R_task_energy[uav_indices] += -1 * self.lambda_r * np.clip(total_energy, 0, 10) / len(uav_indices)
                if self.not_served_rew_to_nearest:
                    # 未被服务的用户奖励只给距离最近的那个无人机。
                    if len(uav_indices) > 0:
                        nearest_uav = np.argmin(uav_gu_distances_2d[:, n])
                        R_task_delay[nearest_uav] += self.gamma_r * (1 + self.gu_tasks[n, 2] - total_delay) if self.gu_tasks[n, 2] > total_delay else -1 * self.delta_r
                        R_task_energy[nearest_uav] += -1 * self.lambda_r * np.clip(total_energy, 0, 10)
            else:
                # 记录自己能不能完成任务。
                self_total_delay = self.gu_tasks[n, 1] / self.F_n
                if self.gu_tasks[n, 2] > self_total_delay:
                    self.self_complete_task[n] = 1
                else:
                    self.self_complete_task[n] = 0

                gu_n_task = self.gu_tasks[n]
                # 使用缓存的信道增益
                h_nm = self.channel_gains[m, n]

                # 计算传输速率
                bw = bandwidth_actions[m, n]
                R_nm = bw * np.log2(1 + p_t * h_nm / (N_0 * (bw + 1e-10)))

                epsilon = 0
                tau_trans = gu_n_task[0] / (R_nm + epsilon)
                E_trans = p_t * tau_trans
                tau_exe = gu_n_task[1] / (computation_actions[m, n] + epsilon)
                E_exe = k_server * (computation_actions[m, n] ** 2) * gu_n_task[1]
                total_energy = E_trans + E_exe
                total_delay = tau_trans + tau_exe
                # 指服务的无人机有奖励。
                R_task_delay[m] += self.gamma_r * (1+gu_n_task[2] - total_delay) if gu_n_task[2] > total_delay else -1 * self.delta_r
                R_task_energy[m] += -1 * self.lambda_r * np.clip(total_energy, 0, 10)  # 这里10，自己加的规定，由于大于1才重新分配动作，某些很少的资源导致计算的时延和能量巨大！ 通常情况下仅为10以内（其实看到的最大只有1.8）。

                if self.gu_tasks[n, 2] > total_delay:
                    per_GU_delay_reward[n] = self.gamma_r * (1+self.gu_tasks[n, 2] - total_delay)
                    self.complete_task[n] = 1
                else:
                    per_GU_delay_reward[n] = -1 * self.delta_r
                    self.complete_task[n] = 0
                per_GU_delay_true[n] = total_delay
                per_GU_energy_reward[n] = -1 * self.lambda_r * np.clip(total_energy, 0, 10)
                per_GU_energy_true[n] = np.clip(total_energy, 0, 10)
                # 这里10，自己加的规定，由于大于1才重新分配动作，某些很少的资源导致计算的时延和能量巨大！ 通常情况下仅为10以内（其实看到的最大只有1.8）。
                per_GU_task_reward[n] = per_GU_delay_reward[n] + per_GU_energy_reward[n]
        if self.fix_uav_pos:
            velocity = np.zeros(self.n_UAVs)
        elif self.cartesian_flight:
            proposal_norms = np.linalg.norm(fly_actions, axis=1)
            self.cartesian_proposal_norm_sum += np.sum(proposal_norms)
            self.cartesian_proposal_count += proposal_norms.size
            self.cartesian_projection_count += np.count_nonzero(proposal_norms > 1.0)
            velocity_vectors = _cartesian_flight_velocity(fly_actions, self.v_max)
            velocity = np.linalg.norm(velocity_vectors, axis=1)
        else:
            velocity = fly_actions[:, 1] * self.v_max
        P_fly = P1 * (1 + 3 * velocity ** 2 / U_tip ** 2) + P2 * (np.sqrt(
            1 + velocity ** 4 / (4 * v0 ** 4)) - velocity ** 2 / (2 * v0 ** 2)) ** 0.5 + 0.5 * d0 * rho * g * A * velocity ** 3
        E_fly = P_fly * self.Delta_t  # 速度0到20，能量63到90
        R_fly_energy = -1 * self.lambda_r * E_fly

        # 更新无人机位置。Update UAV positions
        if self.fix_uav_pos:
            pass
        elif self.cartesian_flight:
            velocity_vectors = _cartesian_flight_velocity(action[:, :2], self.v_max)
            self.uav_positions[:, :2] += velocity_vectors * self.Delta_t
            self.uav_positions[:, 0] = np.clip(self.uav_positions[:, 0], self.x_min_uav, self.x_max_uav)
            self.uav_positions[:, 1] = np.clip(self.uav_positions[:, 1], self.y_min_uav, self.y_max_uav)
        else:
            fly_action = action[:, :2] * np.array([2 * np.pi, self.v_max])  # 方向和速度都是0-1之间的数
            # 向量化更新位置
            directions = fly_action[:, 0]
            velocities = fly_action[:, 1]

            # 计算位移
            dx = velocities * self.Delta_t * np.cos(directions)
            dy = velocities * self.Delta_t * np.sin(directions)

            # 更新位置
            self.uav_positions[:, 0] += dx
            self.uav_positions[:, 1] += dy

            # 确保在边界内
            # self.uav_positions[:, :2] = np.clip(self.uav_positions[:, :2], 0, self.x_max)
            self.uav_positions[:, 0] = np.clip(self.uav_positions[:, 0], self.x_min_uav, self.x_max_uav)
            self.uav_positions[:, 1] = np.clip(self.uav_positions[:, 1], self.y_min_uav, self.y_max_uav)
        # 更新用户位置。 Update GU positions
        # Update ground user velocities and directions using Gauss-Markov Model
        moving_gu_ids = reward_active_ids if self.dynamic_md else np.arange(self.n_GUs)
        random_normal_vel = np.random.normal(0, 0.01 * self.std_dev_gaussian, len(moving_gu_ids))
        updated_gu_velocities = (
            self.alpha_gaussian * self.gu_velocities[moving_gu_ids]
            + (1 - self.alpha_gaussian) * self.mean_velocity
            + np.sqrt(1 - self.alpha_gaussian ** 2) * random_normal_vel
        )
        if (
            self.md_velocity_update_clip_min is not None
            or self.md_velocity_update_clip_max is not None
        ):
            clip_min = (
                -np.inf
                if self.md_velocity_update_clip_min is None
                else self.md_velocity_update_clip_min
            )
            clip_max = (
                np.inf
                if self.md_velocity_update_clip_max is None
                else self.md_velocity_update_clip_max
            )
            updated_gu_velocities = np.clip(
                updated_gu_velocities, clip_min, clip_max
            )
        self.gu_velocities[moving_gu_ids] = updated_gu_velocities
        random_normal_dir = np.random.normal(0, 0.01 * self.std_dev_gaussian, len(moving_gu_ids))
        self.gu_directions[moving_gu_ids] = (
            self.alpha_gaussian * self.gu_directions[moving_gu_ids]
            + (1 - self.alpha_gaussian) * self.gu_directions_0[moving_gu_ids]
            + np.sqrt(1 - self.alpha_gaussian ** 2) * random_normal_dir
        )
        # Update ground user positions
        self.gu_positions[moving_gu_ids, 0] += (
            self.gu_velocities[moving_gu_ids] * np.cos(self.gu_directions[moving_gu_ids]) * self.Delta_t
        )
        self.gu_positions[moving_gu_ids, 1] += (
            self.gu_velocities[moving_gu_ids] * np.sin(self.gu_directions[moving_gu_ids]) * self.Delta_t
        )

        # if self.x_max_gu == 300 and self.n_GUs == 40:
        #     # 在[0, 0]到[300, 300]的区域内有40个用户
        x_out_min = (self.gu_positions[:, 0] < self.x_min_all_gus) & reward_active_mask
        x_out_max = (self.gu_positions[:, 0] > self.x_max_all_gus) & reward_active_mask
        y_out_min = (self.gu_positions[:, 1] < self.y_min_all_gus) & reward_active_mask
        y_out_max = (self.gu_positions[:, 1] > self.y_max_all_gus) & reward_active_mask
        # 反射位置
        self.gu_positions[x_out_min, 0] = 2 * self.x_min_all_gus[x_out_min] - self.gu_positions[x_out_min, 0]
        self.gu_positions[x_out_max, 0] = 2 * self.x_max_all_gus[x_out_max] - self.gu_positions[x_out_max, 0]
        self.gu_positions[y_out_min, 1] = 2 * self.y_min_all_gus[y_out_min] - self.gu_positions[y_out_min, 1]
        self.gu_positions[y_out_max, 1] = 2 * self.y_max_all_gus[y_out_max] - self.gu_positions[y_out_max, 1]
        # else:
        #     # 向量化处理边界碰撞
        #     x_out_min = self.gu_positions[:, 0] < self.x_min_gu
        #     x_out_max = self.gu_positions[:, 0] > self.x_max_gu
        #     y_out_min = self.gu_positions[:, 1] < self.y_min_gu
        #     y_out_max = self.gu_positions[:, 1] > self.y_max_gu
        #
        #     # 反射位置
        #     self.gu_positions[x_out_min, 0] = 2 * self.x_min_gu - self.gu_positions[x_out_min, 0]
        #     self.gu_positions[x_out_max, 0] = 2 * self.x_max_gu - self.gu_positions[x_out_max, 0]
        #     self.gu_positions[y_out_min, 1] = 2 * self.y_min_gu - self.gu_positions[y_out_min, 1]
        #     self.gu_positions[y_out_max, 1] = 2 * self.y_max_gu - self.gu_positions[y_out_max, 1]

        # 处理方向反射
        hit_vertical = x_out_min | x_out_max
        hit_horizontal = y_out_min | y_out_max

        # 角落碰撞 - 两个方向都反射
        corner_hit = hit_vertical & hit_horizontal
        self.gu_directions[corner_hit] = (self.gu_directions[corner_hit] + np.pi) % (2 * np.pi)

        # 垂直墙壁碰撞
        vert_hit = hit_vertical & (~corner_hit)
        self.gu_directions[vert_hit] = (np.pi - self.gu_directions[vert_hit]) % (2 * np.pi)

        # 水平墙壁碰撞
        horiz_hit = hit_horizontal & (~corner_hit)
        self.gu_directions[horiz_hit] = (2 * np.pi - self.gu_directions[horiz_hit]) % (2 * np.pi)

        # 更新基准方向
        self.gu_directions_0[corner_hit | vert_hit | horiz_hit] = self.gu_directions[
            corner_hit | vert_hit | horiz_hit]

        if self.dynamic_md:
            self.dynamic_md_active_sum += reward_active_count
            if self.time_step > self.MAX_SIMULATION_TIME / 2:
                self.dynamic_md_active_sum_second_half += reward_active_count
            self._advance_dynamic_md_population()

        self._update_distance_matrices()

        # if self.env_id == -0.7050493998306726:
        # # if self.env_id:
        #     self.system_performance_all_uavs[self.time_step] = R_task_delay + R_task_energy + R_fly_energy

        R_collision = -1 * self.mu_r * (np.sum(self.uav_uav_distances_2d < self.Dis_min, axis=1) - 1)
        rewards = R_task_delay + R_task_energy + R_fly_energy + R_collision
        # has_neighbors = np.any((uav_uav_distances < self.neighbor_distance) &  (np.eye(self.n_UAVs) == 0), axis=1)
        # if np.sum(has_neighbors) < self.n_UAVs:
        #     print("有无人机孤立:",self.n_UAVs-np.sum(has_neighbors))

        self.cumulative_individual_reward_wo_cover += rewards
        self.cumulative_reward_wo_cover += np.mean(rewards) * np.ones_like(rewards)
        # if self.x_max_gu == 300 and self.n_GUs == 40:
        #     uav_distance_to_150 = np.linalg.norm(self.uav_positions[:, :2] - np.array([150, 150]), axis=1)
        #     rewards[uav_distance_to_150 > 220] -= uav_distance_to_150[uav_distance_to_150 > 220] / 640
        #     # (np.sqrt(2)*150+110)**2 = 103769.  np.sqrt(2)*150=213   (150, 150)到(600,600)的距离np.sqrt(2)*450=636

        # 一、加上未覆盖用户的惩罚。
        # if self.alpha_r != 0:
            # # 一、1 利用所有无人机、用户的真实位置计算奖励
            # uav_gu_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.gu_positions[:, :2].T[np.newaxis, :], axis=1)    #（n_UAVs, n_GUs）
            # coverd_gu = np.any(uav_gu_distances<=self.Cover_R, axis=0)      # (n_GUs,)
            # R_cover_all = -1 * self.alpha_r * (self.n_GUs - np.sum(coverd_gu)) / self.n_GUs
            # rewards += R_cover_all
            # # 一、2 根据{M_i^1}邻居之间通信，利用1跳无人机的s_{i,t} 估计覆盖的总用户数目
            # uav_gu_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.gu_positions[:, :2].T[np.newaxis, :], axis=1)  # （n_UAVs, n_GUs）
            # # update_drone_knowledge_id(self.drones, self.uav_positions[:, :2], self.Cover_R, self.neighbor_R)
            # update_drone_knowledge_id(self.drones, self.uav_positions[:, :2], self.Cover_R, self.neighbor_distance)
            # R_cover_all = np.zeros(self.n_UAVs)
            # for i in range(self.n_UAVs):
            #     knowledge_id_i = np.array(list(self.drones[i].drone_knowledge))
            #     num_knowledge_id_i = len(knowledge_id_i)
            #     # if num_knowledge_id_i <= 1:
            #     #     # 如果就没有一跳邻居。
            #     #     R_cover_all[i] = -1 * self.alpha_r
            #     # else:
            #     #     coverd_gu_i = np.any(uav_gu_distances[knowledge_id_i]<=self.Cover_R, axis=0)      # 邻居几个无人机真正覆盖的用户数目
            #     #     coverd_gu_estimated_i = min(np.sum(coverd_gu_i)*self.n_UAVs/num_knowledge_id_i, self.n_GUs)
            #     #     R_cover_all[i] = -1 * self.alpha_r * (self.n_GUs - coverd_gu_estimated_i) / self.n_GUs
            #     coverd_gu_i = np.any(uav_gu_distances[knowledge_id_i] <= self.Cover_R, axis=0)  # 邻居几个无人机真正覆盖的用户数目
            #     coverd_gu_estimated_i = min(np.sum(coverd_gu_i) * self.n_UAVs / num_knowledge_id_i, self.n_GUs)
            #     R_cover_all[i] = -1 * self.alpha_r * (self.n_GUs - coverd_gu_estimated_i) / self.n_GUs
            # rewards += R_cover_all
            # # 3. 用和邻居的覆盖信息来计算奖励
            # R_cover_all = np.zeros(self.n_UAVs)
            # for n in range(self.n_GUs):
            #     uav_indices = np.where(coverage_mask[:, n])[0]
            #     if len(uav_indices) > 0:
            #         R_cover_all[uav_indices] += self.alpha_r * 1 / len(uav_indices)
            # rewards += R_cover_all
            # # 4.新的全局覆盖奖励
            # uav_gu_distances = np.linalg.norm(self.uav_positions[:, :2, np.newaxis] - self.gu_positions[:, :2].T[np.newaxis, :], axis=1)    #（n_UAVs, n_GUs）
            # coverd_gu = np.any(uav_gu_distances<=self.Cover_R, axis=0)      # (n_GUs,)
            # R_cover_all = self.alpha_r * np.sum(coverd_gu)
            # rewards += R_cover_all

        # # 二、加上覆盖面积的惩罚
        # if self.epsilon_r != 0:
        #     # 二、1 利用真实位置计算奖励
        #     R_cover_areas = -1 * self.epsilon_r * (self.x_max**2 - calculate_coverage_area(self.Cover_R, self.x_max, true_positions = self.uav_positions[:, :2])) / (self.x_max**2)
        #     rewards += R_cover_areas
        #     # # 二、2 根据邻居之间通信，利用延迟的位置计算奖励。     （这些都不符合R(s_{M_i^1}, a_{M_i^1})或者不符合p(s_{i,t+1} | s_{M_i^1}, a_{M_i^1})）
        #     # update_drone_knowledge(self.drones, self.uav_positions[:, :2], self.time_step, self.neighbor_distance)
        #     # R_cover_areas = np.zeros(self.n_UAVs)
        #     # for i in range(self.n_UAVs):
        #     #     R_cover_areas[i] = -1 * self.epsilon_r * (self.x_max**2 - calculate_coverage_area(self.Cover_R, self.x_max, known_positions = self.drones[i].known_positions)) / (self.x_max**2)
        #     # rewards += R_cover_areas
        #     # # 二、3 根据{M_i^1}邻居之间通信，根据1跳邻居所知道的无人机位置 估计覆盖的总面积。
        #     # # update_drone_knowledge_id(self.drones, self.uav_positions[:, :2], self.Cover_R, self.neighbor_R)
        #     # update_drone_knowledge_id(self.drones, self.uav_positions[:, :2], self.Cover_R, self.neighbor_distance)
        #     # R_cover_areas = np.zeros(self.n_UAVs)
        #     # for i in range(self.n_UAVs):
        #     #     knowledge_position_i = self.uav_positions[np.array(list(self.drones[i].drone_knowledge)), :2]
        #     #     num_knwoledge_i = len(knowledge_position_i)
        #     #     # if num_knwoledge_i <= 1:
        #     #     #     # 如果就没有一跳邻居。
        #     #     #     R_cover_areas[i] = -1 * self.epsilon_r
        #     #     # else:
        #     #     #     coverage_area_estimated = min(calculate_coverage_area(self.Cover_R, self.x_max, true_positions = knowledge_position_i) * self.n_UAVs / num_knwoledge_i, self.x_max**2)
        #     #     #     R_cover_areas[i] = -1 * self.epsilon_r * (self.x_max**2 - coverage_area_estimated) / (self.x_max**2)
        #     #     coverage_area_estimated = min(calculate_coverage_area(self.Cover_R, self.x_max,
        #     #                                                           true_positions=knowledge_position_i) * self.n_UAVs / num_knwoledge_i,
        #     #                                   self.x_max ** 2)
        #     #     R_cover_areas[i] = -1 * self.epsilon_r * (self.x_max ** 2 - coverage_area_estimated) / (self.x_max ** 2)
        #     # rewards += R_cover_areas

        self.cumulative_individual_reward += rewards
        # 无人机角度出发每架无人机自己从服务用户获得的性能。 求和是system_performance。
        self.system_performance += np.sum(R_fly_energy) + np.sum(per_GU_task_reward)
        self.system_performance_individual += (R_task_delay + R_task_energy + R_fly_energy)
        # 用户角度出发计算的全局性能
        self.system_performance_coverd_GUs += np.sum(R_fly_energy) + np.sum(per_GU_task_reward)
        slot_system_performance = (
            np.sum(R_fly_energy)
            + np.sum(per_GU_task_reward)
            + np.sum(per_GU_task_reward_others)
        )
        self.system_performance_true_all_GUs += slot_system_performance
        self.system_performance_equivalent_full_GUs += slot_system_performance
        if self.dynamic_md:
            self.system_performance_equivalent_full_GUs += (
                self.n_GUs - reward_active_count
            ) * self.expected_inactive_md_local_reward
        metric_gu_count = max(reward_active_count, 1)
        self.delay_true_all_GUs += (
            np.sum(per_GU_delay_true) + np.sum(per_GU_delay_true_others)
        ) / metric_gu_count
        self.delay_true_coverd_GUs += np.array([[np.sum(per_GU_delay_true), np.sum(per_GU_delay_true != 0)]])
        self.energy_true_all_GUs += np.sum(per_GU_energy_true) + np.sum(per_GU_energy_true_others)
        self.energy_all_GUs_UAVs += np.sum(per_GU_energy_true) + np.sum(per_GU_energy_true_others) + np.sum(E_fly)
        self.complete_task_ratio += np.sum(self.complete_task[reward_active_mask]) / metric_gu_count
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
        if self.dynamic_md:
            transformed_action_components = self.process_actions(transformed_action_components)
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
            assert self.continuous_associate
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
                if self.ave_resource:
                    action_components = [
                        action_components[:, :2]  # Movement actions
                    ]
                    offloading_actions = None
                    bandwidth_actions = None
                    computation_actions = None
                    transformed_offloading = None
                elif self.ave_bandwidth:
                    action_components = [
                        action_components[:, :2],  # Movement actions
                        action_components[:, 2:2 + self.max_GUs_in_range]  # Computation resource allocation
                    ]
                    offloading_actions = None
                    bandwidth_actions = None
                    computation_actions = action_components[1]
                    # Initialize transformed arrays with zeros - preallocate memory once
                    transformed_offloading = None
                else:
                    action_components = [
                        action_components[:, :2],  # Movement actions
                        action_components[:, 2:2 + self.max_GUs_in_range],  # Bandwidth allocation
                        action_components[:, 2 + self.max_GUs_in_range:2 + 2 * self.max_GUs_in_range]  # Computation resource allocation
                    ]
                    offloading_actions = None
                    bandwidth_actions = action_components[1]
                    computation_actions = action_components[2]
                    # Initialize transformed arrays with zeros - preallocate memory once
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
                elif self.ave_bandwidth:
                    action_components = [
                        action_components[:, :2],  # Movement actions
                        action_components[:, 2:2 + self.max_GUs_in_range],  # Offloading decisions
                        action_components[:, 2 + self.max_GUs_in_range:]    # Computation resource allocation
                    ]
                    offloading_actions = action_components[1]
                    bandwidth_actions = None
                    computation_actions = action_components[2]
                    # Initialize transformed arrays with zeros - preallocate memory once
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
                if offloading_actions is not None:
                    transformed_offloading[uav_idx, valid_indices] = offloading_actions[uav_idx, valid_pos]
                if bandwidth_actions is not None:
                    transformed_bandwidth[uav_idx, valid_indices] = bandwidth_actions[uav_idx, valid_pos]
                if computation_actions is not None:
                    transformed_computation[uav_idx, valid_indices] = computation_actions[uav_idx, valid_pos]
        # Create and return result without copying the entire action_components array
        result = action_components.copy()  # Convert to list if it's not already
        if self.fix_uav_pos:
            result[0] = transformed_offloading
            result[1] = transformed_bandwidth
            result[2] = transformed_computation
        else:
            if self.nearest_associate:
                if self.ave_resource:
                    pass
                elif self.ave_bandwidth:
                    result[1] = transformed_computation
                else:
                    result[1] = transformed_bandwidth
                    result[2] = transformed_computation
            else:
                if self.ave_resource:
                    result[1] = transformed_offloading
                elif self.ave_bandwidth:
                    result[1] = transformed_offloading
                    result[2] = transformed_computation
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
        if self.neighbor_distance <= 0:
            return np.eye(self.n_UAVs, dtype=float)
        # 每个智能体i对邻居的权重为1/ (d(i)+1)
        distances = self.uav_uav_distances_2d

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
        if self.neighbor_distance <= 0:
            return np.eye(self.n_UAVs, dtype=float)

        # Calculate pairwise distances between UAVs using vectorization
        distances = self.uav_uav_distances_2d

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

    def get_critic_local_obs(self, actor_obs):
        """Remove actor-only blocks and expose one canonical Type-S encoding."""
        actor_prefix_dim = (
            int(self.ob_state_with_timestep)
            + (self.n_UAVs if self.ob_state_with_id else 0)
            + 2
        )
        if self.actor_only_obs_dim:
            actor_only_end = actor_prefix_dim + self.actor_only_obs_dim
            critic_obs = np.concatenate(
                (actor_obs[:, :actor_prefix_dim], actor_obs[:, actor_only_end:]),
                axis=-1,
            )
        else:
            critic_obs = actor_obs

        packet_prefix_dim = actor_prefix_dim + self.layout_context_dim + 1
        record_dim = self.max_GUs_in_range * self.gu_obs_features
        records = critic_obs[
            :, packet_prefix_dim:packet_prefix_dim + record_dim
        ].reshape(self.n_UAVs, self.max_GUs_in_range, self.gu_obs_features)
        record_valid = np.zeros(
            (self.n_UAVs, self.max_GUs_in_range), dtype=bool
        )
        session_ids = np.full(
            (self.n_UAVs, self.max_GUs_in_range), -1, dtype=np.int64
        )
        visible_count = np.count_nonzero(
            self.nearby_gus_of_uavs != -1, axis=1
        ).astype(np.int32)
        for sender in range(self.n_UAVs):
            count = min(int(visible_count[sender]), self.max_GUs_in_range)
            gu_ids = self.nearby_gus_of_uavs[sender, :count].astype(
                np.int64, copy=False
            )
            record_valid[sender, :count] = True
            session_ids[sender, :count] = (
                self.md_session_ids[gu_ids] if self.dynamic_md else gu_ids
            )
        self.current_type_s_packet = {
            "sender_ids": np.arange(self.n_UAVs, dtype=np.int32),
            "critic_prefix": critic_obs[:, :packet_prefix_dim].copy(),
            "record_features": records.copy(),
            "session_ids": session_ids,
            "record_valid": record_valid,
            "visible_count": visible_count,
        }
        if not self.critic_md_metadata:
            return critic_obs
        valid = record_valid.astype(np.float64)
        metadata = np.stack((valid, valid, np.zeros_like(valid)), axis=-1)
        return np.concatenate(
            (critic_obs, metadata.reshape(self.n_UAVs, -1)), axis=-1
        )

    def get_type_s_data(self):
        """Return the current structured Type-S state without exposing actor messages."""
        packet = self.current_type_s_packet
        geometric_mask = self.uav_uav_distances_2d <= self.neighbor_distance
        np.fill_diagonal(geometric_mask, True)
        return {
            "sender_ids": packet["sender_ids"].copy(),
            "critic_prefix": packet["critic_prefix"].copy(),
            "record_features": packet["record_features"].copy(),
            "session_ids": packet["session_ids"].copy(),
            "record_valid": packet["record_valid"].copy(),
            "visible_count": packet["visible_count"].copy(),
            "uav_positions": self.uav_positions[:, :2].copy(),
            "reception_mask": self.last_state_reception_mask.copy(),
            "geometric_mask": geometric_mask,
            "payload_schema_bits": np.asarray(
                self.type_s_schema_bits, dtype=np.int32
            ),
        }

    def get_actor_message_block(self):
        """Build radius-gated sender-local summaries for the flight policy.

        Packet layout per other UAV is:
        [sender_x, sender_y, active_count, centroid_x, centroid_y,
         mean_velocity_x, mean_velocity_y, spread, hard_task_fraction, mask].
        Under ``absolute_raw_v2`` the first eight quantities use raw physical
        units and only the mask bypasses observation normalization. The legacy
        contract retains relative/scaled values for old checkpoints.
        """
        messages = np.zeros(
            (self.n_UAVs, self.n_UAVs - 1, self.actor_message_features),
            dtype=np.float32,
        )
        if self.actor_message_mode == "zero" or self.neighbor_distance <= 0.0:
            self._record_actor_message_diagnostics(messages)
            return messages.reshape(self.n_UAVs, -1)

        legacy_contract = self.actor_message_contract == "relative_scaled_v1"
        map_scale = max(
            self.x_max_uav - self.x_min_uav,
            self.y_max_uav - self.y_min_uav,
            1.0,
        ) if legacy_contract else 1.0
        velocity_scale = max(float(self.v_max), 1e-6) if legacy_contract else 1.0
        sender_payloads = np.zeros((self.n_UAVs, 7), dtype=np.float32)
        if self.actor_message_mode == "task_summary":
            # A sender broadcasts the same local task summary to every
            # reachable receiver. Compute it once per sender instead of once
            # per directed communication edge.
            for sender_id in range(self.n_UAVs):
                sorted_ids = self.nearby_gus_of_uavs[sender_id]
                in_range_count = int(np.count_nonzero(sorted_ids != -1))
                local_ids = sorted_ids[:self.max_GUs_in_range]
                local_ids = local_ids[local_ids != -1].astype(np.int64)
                sender_payloads[sender_id, 0] = (
                    in_range_count / float(max(self.n_GUs, 1))
                    if legacy_contract else in_range_count
                )
                if not local_ids.size:
                    continue
                local_positions = self.gu_positions[local_ids, :2]
                centroid = np.mean(local_positions, axis=0)
                sender_payloads[sender_id, 1:3] = (
                    (centroid - self.uav_positions[sender_id, :2]) / self.Cover_R
                    if legacy_contract else centroid
                )
                velocity_vectors = np.column_stack((
                    self.gu_velocities[local_ids]
                    * np.cos(self.gu_directions[local_ids]),
                    self.gu_velocities[local_ids]
                    * np.sin(self.gu_directions[local_ids]),
                ))
                sender_payloads[sender_id, 3:5] = (
                    np.mean(velocity_vectors, axis=0) / velocity_scale
                )
                sender_payloads[sender_id, 5] = np.sqrt(np.mean(np.sum(
                    (local_positions - centroid) ** 2, axis=1
                ))) / (self.Cover_R if legacy_contract else 1.0)
                tasks = self.gu_tasks[local_ids]
                sender_payloads[sender_id, 6] = np.mean(
                    tasks[:, 1] / self.F_n > tasks[:, 2]
                )

        for receiver_id in range(self.n_UAVs):
            slot = 0
            for sender_id in range(self.n_UAVs):
                if sender_id == receiver_id:
                    continue
                if (
                    self.uav_uav_distances_2d[receiver_id, sender_id]
                    > self.neighbor_distance
                ):
                    slot += 1
                    continue

                sender_position = (
                    self.uav_positions[sender_id, :2]
                    - self.uav_positions[receiver_id, :2]
                ) / map_scale if legacy_contract else self.uav_positions[
                    sender_id, :2
                ]
                messages[receiver_id, slot, :2] = sender_position
                messages[receiver_id, slot, 9] = 1.0

                if self.actor_message_mode == "task_summary":
                    messages[receiver_id, slot, 2:9] = sender_payloads[sender_id]
                slot += 1
        self._record_actor_message_diagnostics(messages)
        return messages.reshape(self.n_UAVs, -1)

    def _record_actor_message_diagnostics(self, messages):
        mask = messages[:, :, 9]
        self.actor_message_neighbor_fraction_sum += float(np.mean(mask))
        if self.actor_message_mode == "task_summary" and np.any(mask):
            task_payload = messages[:, :, 2:9]
            active_payload = task_payload[mask > 0]
            payload_norm = float(np.mean(np.linalg.norm(active_payload, axis=-1)))
        else:
            payload_norm = 0.0
        self.actor_message_payload_norm_sum += payload_norm
        self.actor_message_observation_count += 1

    def get_local_obs(self):
        """
        添加get_local_obs()方法，每个无人机的局部obs为
        [总无人机数目、总用户数目、区域总长度、
        自身无人机位置；
        与自身距离小于Cover_R的邻居无人机的数目；
        距离由近到远的前max_UAVs_in_neighbor架邻居无人机的位置；
        与自身距离小于Cover_R的地面用户的数目；
        距离由近到远的前self.max_GUs_in_range个地面用户的位置、信道增益和计算任务的信息]。
        """
        # Pre-calculate all UAV-to-UAV distances at once (n_UAVs × n_UAVs matrix)
        uav_uav_distances = self.uav_uav_distances_2d
        # uav_gu_distances = self.uav_gu_distances_2d

        # # obsdim应该不变了。直接就是s_{i,t}已经不动了。
        # if self.concat_neighbor_obs:
        #     obs_dim = self.obs_dim // self.max_UAVs_obs_concat
        # else:
        #     obs_dim = self.obs_dim

        if self.use_atten_actor:
            obs_dim = self.obs_dim // self.max_UAVs_obs_concat
        else:
            obs_dim = self.obs_dim

        local_obs = np.zeros((self.n_UAVs, obs_dim))
        actor_messages = (
            self.get_actor_message_block()
            if self.actor_message_dim
            else None
        )

        idx = 0
        if self.ob_state_with_id:
            agent_id = np.eye(self.n_UAVs)
        for i in range(self.n_UAVs):
            if self.ob_state_with_timestep:
                local_obs[i, idx] = self.time_step
                idx += 1
            if self.ob_state_with_id:
                local_obs[i, idx:idx + self.n_UAVs] = agent_id[i]
                idx += self.n_UAVs

            # # local_obs[i, idx:idx + 3] = np.array([self.n_UAVs, self.n_GUs, self.x_max])
            # local_obs[i, idx:idx + 3] = np.array([self.n_UAVs, self.n_GUs, self.x_max_gu])
            # idx += 3
            # 只有一位的用户中心
            # local_obs[i, idx] = self.x_max_gu/2
            # idx += 1

            # 2. UAV's own position
            local_obs[i, idx:idx + 2] = self.uav_positions[i, :2]
            idx += 2

            if self.actor_neighbor_obs:
                for neighbor_id in range(self.n_UAVs):
                    if neighbor_id == i:
                        continue
                    if uav_uav_distances[i, neighbor_id] <= self.neighbor_R:
                        relative_position = (
                            self.uav_positions[neighbor_id, :2] - self.uav_positions[i, :2]
                        ) / self.neighbor_R
                        local_obs[i, idx:idx + 2] = relative_position
                        local_obs[i, idx + 2] = 1.0
                    idx += 3

            if actor_messages is not None:
                local_obs[i, idx:idx + self.actor_message_dim] = actor_messages[i]
                idx += self.actor_message_dim

            # Common reset-time task prior, not a UAV-to-UAV packet. Keep it
            # outside actor_only_obs_dim so the critic retains the same context.
            if self.layout_context_dim:
                local_obs[
                    i, idx:idx + self.layout_context_dim
                ] = self.episode_layout_context
                idx += self.layout_context_dim

            # # 3. Find neighboring UAVs (excluding self)
            # # neighbor_mask = (uav_uav_distances[i] <= self.Cover_R) & (np.arange(self.n_UAVs) != i)
            # neighbor_mask = (uav_uav_distances[i] <= self.neighbor_distance) & (np.arange(self.n_UAVs) != i)
            # neighbor_indices = np.where(neighbor_mask)[0]
            # neighbor_count = len(neighbor_indices)
            # # 4. Number of neighboring UAVs
            # local_obs[i, idx] = neighbor_count
            # idx += 1
            # # 5. Get closest neighbors
            # if neighbor_count > 0:
            #     # Sort neighbors by distance
            #     neighbor_distances = uav_uav_distances[i, neighbor_indices]
            #     sorted_idx = np.argsort(neighbor_distances)
            #     closest_neighbors = neighbor_indices[sorted_idx[:self.max_UAVs_in_neighbor]]
            #     # Add positions of closest neighbors
            #     for j, neighbor_idx in enumerate(closest_neighbors):
            #         if j < self.max_UAVs_in_neighbor:
            #             local_obs[i, idx:idx + 2] = self.uav_positions[neighbor_idx, :2]
            #             idx += 2
            # # Pad with zeros if there are fewer than max_UAVs_in_neighbor
            # padding_neighbors = self.max_UAVs_in_neighbor - min(neighbor_count, self.max_UAVs_in_neighbor)
            # idx += padding_neighbors * 2

            in_range_count = np.sum(self.nearby_gus_of_uavs[i] != -1)
            in_range_indices = self.nearby_gus_of_uavs[i, :in_range_count]
            # 7. Number of GUs within coverage
            local_obs[i, idx] = in_range_count
            idx += 1
            # 8. Sort GUs by distance and take closest max_GUs_in_range
            if in_range_count > 0:
                closest_gus = in_range_indices[:self.max_GUs_in_range]
                for j, gu_idx in enumerate(closest_gus):
                    if j < self.max_GUs_in_range:
                        local_obs[i, idx:idx + 2] = self.gu_positions[gu_idx, :2]
                        idx += 2
                        if self.dynamic_md:
                            local_obs[i, idx:idx + 4] = np.array([
                                self.gu_velocities[gu_idx],
                                self.gu_directions[gu_idx],
                                self.gu_directions_0[gu_idx],
                                self.md_remaining_lifetime[gu_idx],
                            ])
                            idx += 4
                            local_obs[i, idx] = self.channel_gains[i, gu_idx]
                            idx += 1
                            local_obs[i, idx:idx + 3] = self.gu_tasks[gu_idx]
                            idx += 3
                        else:
                            local_obs[i, idx] = self.gu_directions[gu_idx]
                            idx += 1
                            local_obs[i, idx] = self.channel_gains[i, gu_idx]
                            idx += 1
                            local_obs[i, idx:idx + 3] = self.gu_tasks[gu_idx]
                            idx += 3
                            local_obs[i, idx] = gu_idx
                            idx += 1
                            local_obs[i, idx] = (
                                0 if self.gu_tasks[gu_idx, 2] > self.gu_tasks[gu_idx, 1] / self.F_n else 1
                            )
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
        uav_gu_distances = self.uav_gu_distances_2d
        local_avail_actions = np.zeros((self.n_UAVs, self.max_GUs_in_range), dtype=np.int32)
        if self.nearest_avail_actions:
            # 重叠覆盖的用户只能卸载到最近的无人机。
            # 提取候选区域 (n_UAVs, max_gus)
            candidate_matrix = self.nearby_gus_of_uavs[:, :self.max_GUs_in_range]
            # 创建有效用户掩码 (n_UAVs, max_gus)
            valid_mask = candidate_matrix != -1
            if not np.any(valid_mask):
                return local_avail_actions
            # 获取所有有效的(uav_id, pos, user_id)组合
            uav_indices, pos_indices = np.where(valid_mask)
            user_ids = candidate_matrix[uav_indices, pos_indices]
            # 为每个唯一用户找到所有候选无人机
            unique_users = np.unique(user_ids)
            # 对每个唯一用户进行向量化处理
            for user_id in unique_users:
                # 找到所有包含该用户的位置
                user_mask = user_ids == user_id
                candidate_uav_indices = uav_indices[user_mask]
                candidate_pos_indices = pos_indices[user_mask]
                # 获取该用户到所有候选无人机的距离
                user_distances = uav_gu_distances[candidate_uav_indices, user_id]
                # 找到最近的无人机
                closest_idx = np.argmin(user_distances)
                closest_uav = candidate_uav_indices[closest_idx]
                closest_pos = candidate_pos_indices[closest_idx]
                # 在avail_actions中标记连接
                local_avail_actions[closest_uav, closest_pos] = 1
            return local_avail_actions
        else:
            # 每架无人机直接可以卸载范围内最近的max_GUs_in_range个用户。
            for i in range(self.n_UAVs):
                in_range_count = np.sum(self.nearby_gus_of_uavs[i] != -1)
                if in_range_count >= self.max_GUs_in_range:
                    local_avail_actions[i] = 1
                else:
                    local_avail_actions[i, :in_range_count] = 1
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
        # 使用缓存的距离和掩码
        uav_gu_distances = self.uav_gu_distances_2d
        coverage_mask = self.coverage_mask

        if self.distance_only_user_sort:
            for i in range(self.n_UAVs):
                in_range_indices = np.flatnonzero(coverage_mask[i])
                if len(in_range_indices):
                    order = np.argsort(uav_gu_distances[i, in_range_indices])
                    nearby_users_sorted[i, :len(in_range_indices)] = in_range_indices[order]
            return nearby_users_sorted

        can_finish = self.gu_tasks[:, 1] / self.F_n <= self.gu_tasks[:, 2]
        for i in range(self.n_UAVs):
            # 把不能完成任务的放在前面。
            cannot_finish__in_range_mask = coverage_mask[i] & (~can_finish)
            cannot_finish__in_range_indices = np.where(cannot_finish__in_range_mask)[0]
            if len(cannot_finish__in_range_indices) != 0:
                cannot_finish__sorted_indices = cannot_finish__in_range_indices[np.argsort(uav_gu_distances[i][cannot_finish__in_range_indices])]
                # Fill in the sorted user IDs for this UAV (up to the number of users in range)
                nearby_users_sorted[i, :len(cannot_finish__in_range_indices)] = cannot_finish__sorted_indices
            can_finish__in_range_mask = coverage_mask[i] & can_finish
            can_finish__in_range_indices = np.where(can_finish__in_range_mask)[0]
            if len(can_finish__in_range_indices) != 0:
                can_finish__sorted_indices = can_finish__in_range_indices[np.argsort(uav_gu_distances[i][can_finish__in_range_indices])]
                # Fill in the sorted user IDs for this UAV (up to the number of users in range)
                nearby_users_sorted[i, len(cannot_finish__in_range_indices):len(cannot_finish__in_range_indices)+len(can_finish__in_range_indices)] \
                    = can_finish__sorted_indices
            # # 直接就按照距离排序，不管能不能完成。
            # in_range_mask = uav_gu_distances[i] <= self.Cover_R
            # # Get users that are within range
            # in_range_indices = np.where(in_range_mask)[0]
            # # If no users in range, continue to next UAV
            # if len(in_range_indices) == 0:
            #     continue
            # # Sort the indices by their distances
            # sorted_indices = in_range_indices[np.argsort(uav_gu_distances[i][in_range_indices])]
            # # Fill in the sorted user IDs for this UAV (up to the number of users in range)
            # nearby_users_sorted[i, :len(sorted_indices)] = sorted_indices
        return nearby_users_sorted

    def get_info(self):
        return {'n_agents': self.n_agents,
                'obs_space': self.observation_space.shape[0],
                'state_space': self.state_space.shape[0],
                'action_space': self.action_space}

    def render(self, mode="human", acts=None, timestep=None, title=None):
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_xlim(0, self.x_max)
        # ax.set_ylim(0, self.x_max)
        # ax.set_xlabel('X (m)')
        # ax.set_ylabel('Y (m)')
        # # Plot UAVs and annotate their IDs
        # for i in range(self.n_UAVs):
        #     ax.scatter(self.uav_positions[i, 0], self.uav_positions[i, 1], c='r', label='UAV' if i == 0 else "")
        #     ax.annotate(f'UAV {i}', (self.uav_positions[i, 0], self.uav_positions[i, 1]))
        #     # Draw service range
        #     circle = plt.Circle((self.uav_positions[i, 0], self.uav_positions[i, 1]), self.Cover_R, color='r',
        #                         fill=False, linestyle='--')
        #     ax.add_patch(circle)
        #
        # # Plot GUs and annotate their IDs
        # for j in range(self.n_GUs):
        #     if acts is not None:
        #         if self.complete_task[j] == 1:
        #             ax.scatter(self.gu_positions[j, 0], self.gu_positions[j, 1], c='b', label='GU' if j == 0 else "")
        #         else:
        #             ax.scatter(self.gu_positions[j, 0], self.gu_positions[j, 1], c='k', label='GU' if j == 0 else "")
        #
        #         column = acts[:, j]
        #         if np.any(column == 1):  # 检查是否有1
        #             # 如果是无人机帮忙卸载的任务。 记录无人机的id。
        #             serving_uavs = np.argmax(column == 1)  # 找到第一个1的位置
        #             annotation_text = f'{serving_uavs}'
        #             if self.self_complete_task[j] == 0: # 如果是本来不能完成任务的。这样就太好了
        #                 annotation_text += f'!'
        #             ax.annotate(annotation_text, (self.gu_positions[j, 0], self.gu_positions[j, 1]))
        #     else:
        #         ax.scatter(self.gu_positions[j, 0], self.gu_positions[j, 1], c='b', label='GU' if j == 0 else "")
        #         annotation_text = f'{j}'
        #         ax.annotate(annotation_text, (self.gu_positions[j, 0], self.gu_positions[j, 1]))
        # if title is not None:
        #     plt.title(title)
        #
        # ax.legend()
        # if timestep is not None:
        #     if title is not None:
        #         save_path = title + '-timestep' + str(timestep) + ".png"
        #     else:
        #         save_path = str(timestep) + ".png"
        #     plt.savefig(save_path, dpi=200, bbox_inches='tight')
        #     print(f"图表已保存到: {save_path}")
        #     plt.close()
        # else:
        #     plt.show()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, self.x_max_uav)
        ax.set_ylim(0, self.y_max_uav)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        # Add grid with light color
        ax.grid(True, alpha=0.3, linestyle='--')
        # Define colors for each UAV
        # colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # Blue, Green, Red, Purple
        colors = [
            '#1f77b4',  # Blue (original)
            '#2ca02c',  # Green (original)
            '#d62728',  # Red (original)
            '#9467bd',  # Purple (original)
            '#ff7f0e',  # Orange
            '#17becf',  # Cyan
            '#e377c2',  # Pink
            '#bcbd22',  # Lime
            '#8c564b',  # Brown
            '#ff9e1f',  # Golden Yellow
            '#1a237e',  # Navy Blue
            '#006064',  # Dark Teal
            '#c2185b',  # Magenta
            '#33691e',  # Forest Green
            '#ff5722',  # Deep Orange
            '#7b1fa2'  # Violet
        ]
        # Plot UAVs
        for i, (x, y) in enumerate(self.uav_positions[:, :2]):
            # Draw coverage area
            coverage = Circle((x, y), self.Cover_R, color=colors[i], alpha=0.1)
            ax.add_patch(coverage)

            # Draw UAV with a black edge instead of using PathEffects
            ax.scatter(x, y, marker='X', color=colors[i], s=200)

            # Add UAV label with a white background instead of using PathEffects
            ax.text(x + 2*i, y + 2*i, f"UAV{i + 1}", color=colors[i], size=12)
                    # bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2',
                    #           edgecolor='none')
        # Plot ground users
        render_gu_ids = np.flatnonzero(self.active_md_mask) if self.dynamic_md else np.arange(self.n_GUs)
        for j in render_gu_ids:
            x, y = self.gu_positions[j, :2]
            if acts is not None:
                column = acts[:, j]
                if np.any(column == 1):  # 检查是否有1
                    serving_uavs = np.argmax(column == 1)  # 找到第一个1的位置
                    ax.scatter(x, y, marker='o', color=colors[serving_uavs], s=50)
                else:
                    # Unserved users are black
                    ax.scatter(x, y, marker='o', color='black', s=50)
                # if self.self_complete_task[j] == 0:
                #     annotation_text = f'0'
                #     if self.complete_task[j] == 1:
                #         annotation_text += f'!'
                #     ax.text(x + 2, y + 2, annotation_text)
                # ax.text(300, 100, f"Not C{np.sum(self.complete_task==0)}", fontweight='bold', color=colors[i],
                #     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2',
                #               edgecolor='none'))
            else:
                ax.scatter(x, y, marker='o', color='black', s=50)
        # Create legend entries manually
        legend_elements = []
        if timestep == 0:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor='black', markersize=8,
                                              label='MDs'))
        else:
            for i in range(4):
                # Add UEs served by each UAV
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=colors[i], markersize=8,
                                                  label=f'MDs served by UAV {i + 1}'))
            # Add locally computing users to legend
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor='black', markersize=8,
                                              label='MDs compute locally'))
        # Add legend with shadow effect
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.6)

        # if title is not None:
        #     plt.title(title+'-UAV Service Assignment', fontsize=16, pad=20)
        plt.tight_layout()

        if timestep is not None:
            if title is not None:
                save_path = title + '-timestep' + str(timestep)
            else:
                save_path = str(timestep)
            plt.savefig(save_path+".png", dpi=300, bbox_inches='tight', pad_inches=0.05)
            # plt.savefig(save_path+".pdf", bbox_inches="tight", pad_inches=0.05)
            # plt.savefig(save_path + ".eps", bbox_inches="tight", pad_inches=0.05, format='eps')
            print(f"图表已保存到: {save_path}")
            plt.close()
        else:
            plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    class Args:
        def __init__(self):
            self.n_UAVs = 25
            self.n_GUs = 200
            self.x_max = 900
            self.max_UAVs_obs_concat = 25
            self.neighbor_distance = 195.0
            self.d_optimal = 210
            self.perform_with_local_state = False
            self.state_is_k_hops = True
            self.local_reward = True
            self.alpha_r = 32
            self.beta_r = 0.5
            self.gamma_r = 4
            self.delta_r = 2.0
            self.lambda_r = 0.0001
            self.epsilon_r = 0.0
            self.mu_r = 64
            self.all_uav_k_hops = True
            self.use_atten_actor = False
            self.discrete_associate = False
            self.continuous_associate = True
            self.nearest_associate = False
            self.nearest_avail_actions = False
            self.not_process_action = True
            self.fix_uav_pos = False
            self.ave_resource = False
            self.ave_bandwidth = False
            self.not_served_rew_to_ave = False
            self.not_served_rew_to_nearest = True
            self.average_neighbor_advantage = False
            self.neighbor_R = 300

            self.max_UAVs_in_neighbor = 3

            self.B = 30 * 10 ** 6  # 信道带宽，Hz，20MHz
            self.H_UAV = 90  # m，无人机服务器固定飞行高度
            self.H_GU = 1  # m，地面用户高度
            # 边缘计算参数
            self.F_m = 20 * 10 ** 9  # 15 Gigacycles，无人机服务器的最大可用计算资源
            self.F_n = 1.5 * 10 ** 9  # 1.5 Gigacycles，地面用户的最大可用计算资源
            # 任务参数
            self.D_min = 2 * 10 ** 6  # 1 MB，生成任务的最小数据量
            self.D_max = 4 * 10 ** 6  # 3 MB，生成任务的最大数据量
            self.C_min = 700 * 10 ** 6  # 300 Megacycles，生成任务的最小资源需求
            self.C_max = 800 * 10 ** 6  # 500 Megacycles，生成任务的最大资源需求
            self.delay_min = 0.499  # 250 ms，生成任务的最小延迟要求
            self.delay_max = 0.5  # 300 ms，生成任务的最大延迟要求
            # 这几个参数自己设置的
            self.Dis_min = 8  # 20m, 无人机服务器之间的最小防碰撞距离
            self.Cover_R = 90  # 120m, 无人机服务器二维的通信覆盖半径，在Evolutionary Multi-Objective中有最大仰角45度的限制，覆盖范围等于无人机高度。
            self.Delta_t = 0.5  # 0.5s, 无人机服务器的最小时间间隔
            self.episode_length = 400  # 50步数，仿真的最大步数
            # self.MAX_SIMULATION_TIME = args.episode_length  # 50步数，仿真的最大步数，在config.py里边有episode_length
            self.w1 = 20  # 计算奖励时时延部分的权重
            self.w2 = 1  # 计算奖励时能耗部分的权重
            self.p3 = 500  # 计算奖励时防碰撞部分的惩罚值
            self.v_max = 30  # 20m/s，无人机飞行的最大速度
            self.mean_velocity = 15.0  # 10m/s，地面用户的平均移动速度
            self.ob_state_with_timestep = True
            self.max_GUs_in_range = 10
            self.q1 = 1
            self.q2 = 1
            self.q3 = 1
            self.q4 = 1
            self.q5 = 1

    args = Args()
    env = MEC(args)
    start = time.time()
    # Reset the environment
    for j in range(10):
        obs, state, avail_actions, Metropolis_weights, attention_active_mask = env.reset()
        print(j)
        for i in range(400):
            # print("Initial Observation:", obs)
            # print("Initial State:", state)
            # print("Initial Available Actions:", avail_actions)
            action = np.concatenate([np.random.uniform(low=0, high=1, size=(env.n_UAVs, 2)),
                      np.random.uniform(low=0, high=1, size=(env.n_UAVs, env.max_GUs_in_range)),
                      np.random.uniform(low=0, high=1, size=(env.n_UAVs, env.max_GUs_in_range)),
                      np.random.uniform(low=0, high=1, size=(env.n_UAVs, env.max_GUs_in_range))], axis=-1)

            obs, rewards, dones, state, avail_actions, info, Metropolis_weights, attention_active_mask = env.step(action)
            if i % 50==0:
                print(i)
        # print("Observation after step:", obs)
        # print("Rewards after step:", rewards)
        # print("Dones after step:", dones)
        # print("State after step:", state)
        # print("Available Actions after step:", avail_actions)
        # print("Info after step:", info)
    end = time.time()
    print("time:", end-start)

