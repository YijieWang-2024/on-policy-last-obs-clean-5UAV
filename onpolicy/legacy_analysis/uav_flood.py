import numpy as np
from scipy.spatial.distance import cdist


class OptimizedUAVNetwork:
    def __init__(self, positions, observations, neighbor_distance, max_iterations=10):
        """
        优化的UAV网络初始化.

        Args:
            positions: 形状为(n_rollout_threads, n_UAVs, 2)的数组，包含UAV最终位置
            observations: 形状为(n_UAVs, episode_length, n_rollout_threads, 1)的数组，包含局部观测
            # # observations: 形状为(episode_length, n_rollout_threads, n_UAVs, 1)的数组，包含局部观测
            neighbor_distance: 两个UAV被视为邻居的最大距离
            max_iterations: 通信迭代的最大次数(k)
        """
        self.positions = positions
        # 处理新的observations形状 (n_UAVs, episode_length, n_rollout_threads, 1)
        # 转换为内部一致的表示形式 (episode_length, n_rollout_threads, n_UAVs, 1)
        self.original_observations = np.transpose(observations, (1, 2, 0, 3))
        # self.original_observations = observations
        self.neighbor_distance = neighbor_distance
        self.max_iterations = max_iterations

        self.n_rollout_threads = positions.shape[0]
        self.n_UAVs = positions.shape[1]
        self.episode_length = self.original_observations.shape[0]

        # 使用更高效的方式计算邻居关系
        self.neighbor_masks = self._compute_neighbors_vectorized()
        # 改进的知识库结构：使用稀疏存储并分离值和标记
        # Shape: (n_rollout_threads, n_UAVs, episode_length, n_UAVs, 1)
        self.knowledge_base = np.full((self.n_rollout_threads, self.n_UAVs,
                                       self.episode_length, self.n_UAVs, 1), np.nan)
        # 使用位掩码跟踪观测状态，比布尔数组更高效
        # 0: 没有数据, 1: 有数据但已发送, 2: 有数据且需要发送
        self.observation_status = np.zeros((self.n_rollout_threads, self.n_UAVs,
                                            self.episode_length, self.n_UAVs), dtype=np.int8)
        # 初始化每个UAV自己的观测
        for thread in range(self.n_rollout_threads):
            for uav in range(self.n_UAVs):
                self.knowledge_base[thread, uav, :, uav, 0] = self.original_observations[:, thread, uav, 0]
                # 标记为"有数据且需要发送"
                self.observation_status[thread, uav, :, uav] = 2

        # 预计算每个线程中UAV的总邻居数，用于性能统计
        self.total_neighbors = {t: sum(self.neighbor_masks[t].sum(axis=1)) for t in range(self.n_rollout_threads)}

    def _compute_neighbors_vectorized(self):
        """使用向量化操作计算邻居关系，显著提高效率."""
        neighbor_masks = {}
        for thread in range(self.n_rollout_threads):
            # 使用cdist一次性计算所有UAV之间的距离
            distances = cdist(self.positions[thread], self.positions[thread])
            # 创建掩码：距离小于neighbor_distance且不是自身的即为邻居
            mask = (distances <= self.neighbor_distance) & (distances > 0)
            neighbor_masks[thread] = mask
        return neighbor_masks

    def run_flooding(self):
        """执行优化的洪泛通信协议，确保同步通信."""
        message_counts = np.zeros(self.max_iterations, dtype=np.int64)
        for iteration in range(self.max_iterations):
            messages_this_iteration = 0
            updated_uavs = False
            # 创建缓冲区存储这轮迭代中的所有更新，确保同步通信
            # 结构: {thread: {receiver: [(timestep, source, value), ...], ...}, ...}
            updates_buffer = {t: {r: [] for r in range(self.n_UAVs)} for t in range(self.n_rollout_threads)}
            # 跟踪哪些发送者的状态需要从2更新为1
            senders_status_updates = {t: {} for t in range(self.n_rollout_threads)}
            # 一次性处理所有线程的所有UAV
            for thread in range(self.n_rollout_threads):
                # 查找需要发送数据的UAV（状态为2的）
                senders_with_data = np.any(self.observation_status[thread] == 2, axis=(1, 2))
                if not np.any(senders_with_data):
                    continue
                senders_status_updates[thread] = {}

                for sender in np.where(senders_with_data)[0]:
                    # 获取此发送者的所有邻居
                    receivers = np.where(self.neighbor_masks[thread][sender])[0]
                    if len(receivers) == 0:
                        continue
                    # 查找发送者需要发送的数据
                    data_to_send = self.observation_status[thread, sender] == 2
                    if not np.any(data_to_send):
                        continue
                    # 记录需要将该发送者的哪些数据状态从2更新为1
                    senders_status_updates[thread][sender] = data_to_send.copy()
                    # 获取需要发送的时间步和源UAV
                    timesteps, sources = np.where(data_to_send)
                    messages_this_iteration += len(timesteps)
                    # 遍历每个接收者
                    for receiver in receivers:
                        # 筛选接收者尚未收到的观测
                        receiver_status = self.observation_status[thread, receiver]
                        # 收集需要更新的数据
                        for i in range(len(timesteps)):
                            timestep, source = timesteps[i], sources[i]
                            if receiver_status[timestep, source] == 0:  # 接收者没有此数据
                                # 将更新添加到缓冲区，而不是直接更新
                                value = self.knowledge_base[thread, sender, timestep, source, 0]
                                updates_buffer[thread][receiver].append((timestep, source, value))
                                updated_uavs = True

            # 在所有发送者都处理完后，统一应用更新
            for thread in range(self.n_rollout_threads):
                # 1. 先更新所有接收者的状态
                for receiver, updates in updates_buffer[thread].items():
                    for timestep, source, value in updates:
                        self.knowledge_base[thread, receiver, timestep, source, 0] = value
                        # 将接收者的观测状态更新为"有数据且需要发送"
                        self.observation_status[thread, receiver, timestep, source] = 2
                # 2. 更新所有发送者的状态（从2到1）
                for sender, mask in senders_status_updates[thread].items():
                    self.observation_status[thread, sender][mask] = 1
            # 记录此迭代发送的消息数
            message_counts[iteration] = messages_this_iteration
            if not updated_uavs:
                print(f"洪泛通信在{iteration + 1}次迭代后完成！")
                message_counts = message_counts[:iteration + 1]
                break
        # 计算覆盖率统计
        coverage = self._calculate_coverage()
        return coverage, message_counts

    def get_final_observations(self):
        """
        返回所有UAV收集到的最终观测数据，格式与输入保持一致
        输出形状为 (n_UAVs, episode_length, n_rollout_threads, n_UAVs, 1)
        使用向量化操作提高效率
        """
        # 创建结果数组
        final_observations = np.full((self.n_UAVs, self.episode_length, self.n_rollout_threads, self.n_UAVs, 1), np.nan)
        # 首先创建一个转置后的视图，无需复制数据
        # 从 (n_rollout_threads, n_UAVs, episode_length, n_UAVs, 1)
        # 到 (n_UAVs, episode_length, n_rollout_threads, n_UAVs, 1)
        transposed_view = np.transpose(self.knowledge_base, (1, 2, 0, 3, 4))
        # 直接复制有效数据（非NaN的值）
        # 使用布尔掩码避免循环
        mask = ~np.isnan(transposed_view)
        final_observations[mask] = transposed_view[mask]

        return final_observations

    def _calculate_coverage(self):
        """计算每个UAV已接收到的观测百分比."""
        # 状态为1或2的表示已有数据
        has_observation = (self.observation_status > 0)
        total_possible = self.episode_length * self.n_UAVs
        coverage = np.sum(has_observation, axis=(2, 3)) / total_possible
        return coverage

    def get_network_stats(self):
        """返回网络统计信息，包括连接性分析."""
        stats = {
            "network_density": {},
            "avg_neighbors": {},
            "isolated_uavs": {},
            "max_neighbors": {},
        }

        for thread in range(self.n_rollout_threads):
            # 计算网络密度（实际连接数/可能的最大连接数）
            possible_connections = self.n_UAVs * (self.n_UAVs - 1)
            actual_connections = self.total_neighbors[thread]
            stats["network_density"][
                thread] = actual_connections / possible_connections if possible_connections > 0 else 0

            # 平均邻居数
            stats["avg_neighbors"][thread] = actual_connections / self.n_UAVs if self.n_UAVs > 0 else 0

            # 没有邻居的UAV数量（孤立节点）
            neighbors_per_uav = self.neighbor_masks[thread].sum(axis=1)
            stats["isolated_uavs"][thread] = np.sum(neighbors_per_uav == 0)

            # 最多的邻居数
            stats["max_neighbors"][thread] = np.max(neighbors_per_uav) if len(neighbors_per_uav) > 0 else 0

        return stats


# 示例用法
if __name__ == "__main__":
    # 示例参数
    n_UAVs = 25
    episode_length = 400
    n_rollout_threads = 4
    neighbor_distance = 390
    max_iterations = 20

    # 生成随机位置和观测用于演示
    np.random.seed(42)
    positions = np.random.rand(n_rollout_threads, n_UAVs, 2) * 900
    # observations = np.random.rand(episode_length, n_rollout_threads, n_UAVs, 1)
    # 注意观测数据的形状现在是 (n_UAVs, episode_length, n_rollout_threads, 1)
    observations = np.random.rand(n_UAVs, episode_length, n_rollout_threads, 1)
    mean_observations = np.mean(observations, axis=0, keepdims=True)

    # 创建并运行网络
    import time

    start_time = time.time()

    network = OptimizedUAVNetwork(positions, observations, neighbor_distance, max_iterations)
    coverage, message_counts = network.run_flooding()
    flooding_observations = network.get_final_observations()
    flooding_observations_mean = np.nanmean(flooding_observations, axis=-2)     # 每个无人机内部 对所有无人机的观测取均值
    flooding_errors = np.mean(np.abs(flooding_observations_mean - mean_observations) / np.abs(observations - mean_observations))
    print('flooding_errors:', flooding_errors)

    elapsed_time = time.time() - start_time

    print(f"\n执行时间: {elapsed_time:.3f}秒")
    print(f"\n每次迭代的消息数量: {message_counts}")
    print(f"总消息数: {np.sum(message_counts)}")

    print("\n最终覆盖率（按UAV和线程）:")
    print(coverage)
    print(f"所有UAV和线程的平均覆盖率: {np.mean(coverage) * 100:.2f}%")

    # 显示网络统计信息
    stats = network.get_network_stats()
    print("\n网络统计:")
    for thread in range(n_rollout_threads):
        print(f"线程 {thread}:")
        print(f"  网络密度: {stats['network_density'][thread]:.4f}")
        print(f"  平均邻居数: {stats['avg_neighbors'][thread]:.2f}")
        print(f"  孤立UAV数: {stats['isolated_uavs'][thread]}")
        print(f"  最大邻居数: {stats['max_neighbors'][thread]}")