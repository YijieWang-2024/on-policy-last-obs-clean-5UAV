import time
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

def run_consensus_algorithm(positions, local_observations, neighbor_distance, max_iterations):
    """
    Run the consensus algorithm to share local observations among UAVs.

    Args:
        positions: UAV positions with shape (n_rollout_threads, n_UAVs, 2)
        local_observations: Local observations with shape (n_UAVs, episode_length, n_rollout_threads, 1)
        neighbor_distance: Maximum distance for two UAVs to be neighbors
        max_iterations: Maximum number of consensus iterations

    Returns:
        tuple: (updated_observations, message_counts)
            - updated_observations: Observations after consensus with shape (n_UAVs, episode_length, n_rollout_threads, 1)
            - message_counts: Array with number of messages exchanged in each iteration
    """
    n_UAVs, episode_length, n_rollout_threads, _ = local_observations.shape

    # Reorganize observation data for more direct computation
    # New shape: (n_rollout_threads, n_UAVs, episode_length, 1)
    obs = np.transpose(local_observations, (2, 0, 1, 3))
    # 初始化共识估计值
    consensus_estimates = np.copy(obs)

    # Compute adjacency matrices using cdist for consistency
    adjacency_matrices = np.zeros((n_rollout_threads, n_UAVs, n_UAVs))
    for env_idx in range(n_rollout_threads):
        # Calculate pairwise distances using cdist for consistency with flooding algorithm
        distances = cdist(positions[env_idx], positions[env_idx])
        # Create adjacency matrix (1 if neighbors, 0 otherwise)
        adjacency = (distances <= neighbor_distance).astype(np.float32)
        np.fill_diagonal(adjacency, 0)  # Remove self-connections
        adjacency_matrices[env_idx] = adjacency

    # Compute weight matrices
    weight_matrices = np.zeros_like(adjacency_matrices)
    for env_idx in range(n_rollout_threads):
        adjacency_matrix = adjacency_matrices[env_idx]
        degrees = np.sum(adjacency_matrix, axis=1)
        # Compute Metropolis weights
        max_degrees_matrix = np.maximum.outer(degrees, degrees)
        weights_matrix = np.where(adjacency_matrix > 0, 1.0 / (1.0 + max_degrees_matrix), 0.0)
        # Self-weights for row stochasticity
        diagonal_values = 1.0 - np.sum(weights_matrix, axis=1)
        np.fill_diagonal(weights_matrix, diagonal_values)
        weight_matrices[env_idx] = weights_matrix

    # Track message counts for each iteration (for consistency with flooding algorithm)
    message_counts = np.zeros(max_iterations, dtype=np.int64)

    # Run consensus iterations
    for iteration in range(max_iterations):
        messages_this_iteration = 0
        for env_idx in range(n_rollout_threads):
            weights = weight_matrices[env_idx]  # (n_UAVs, n_UAVs)
            if iteration>0:
                adjacency = adjacency_matrices[env_idx]
                messages_this_iteration += np.sum(np.any(adjacency, axis=1))*episode_length
            else:
                messages_this_iteration += n_UAVs * episode_length
            # 使用矩阵乘法执行共识迭代
            thread_estimates = consensus_estimates[env_idx]  # (n_UAVs, episode_length, 1)
            # 使用矩阵乘法进行更新 (n_UAVs, n_UAVs) @ (n_UAVs, episode_length, 1)
            # 重塑为2D进行矩阵乘法，然后恢复原始形状
            reshaped_estimates = thread_estimates.reshape(n_UAVs, -1)  # (n_UAVs, episode_length*1)
            updated_estimates = weights @ reshaped_estimates  # (n_UAVs, episode_length*1)
            consensus_estimates[env_idx] = updated_estimates.reshape(n_UAVs, episode_length, 1)

        message_counts[iteration] = messages_this_iteration

    # Transpose back to original format: (n_UAVs, episode_length, n_rollout_threads, 1)
    result = np.transpose(consensus_estimates, (1, 2, 0, 3))
    return result, message_counts


# def run_flooding_algorithm(positions, local_observations, neighbor_distance, max_iterations):
#     """
#     Run optimized flooding communication protocol with synchronous communication.
#
#     Args:
#         positions: UAV positions with shape (n_rollout_threads, n_UAVs, 2)
#         local_observations: Local observations with shape (n_UAVs, episode_length, n_rollout_threads, 1)
#         neighbor_distance: Maximum distance for two UAVs to be neighbors
#         max_iterations: Maximum number of flooding iterations
#
#     Returns:
#         tuple: (updated_observations, message_counts)
#             - updated_observations: Final observations after flooding with shape (n_UAVs, episode_length, n_rollout_threads, 1)
#             - message_counts: Array with number of messages exchanged in each iteration
#     """
#     n_UAVs, episode_length, n_rollout_threads, _ = local_observations.shape
#     # Convert to internally consistent representation (n_rollout_threads, n_UAVs, episode_length, 1)
#     obs_transposed = np.transpose(local_observations, (2, 0, 1, 3))
#
#     # Precompute neighbor relationships
#     neighbor_masks = np.zeros((n_rollout_threads, n_UAVs, n_UAVs), dtype=bool)
#     for thread in range(n_rollout_threads):
#         distances = cdist(positions[thread], positions[thread])
#         # True if UAVs are neighbors but not self
#         neighbor_masks[thread] = (distances <= neighbor_distance) & (distances > 0)
#
#     # 改进的知识库结构：使用稀疏存储并分离值和标记
#     # Shape: (n_rollout_threads, n_UAVs, episode_length, n_UAVs, 1)
#     knowledge_base = np.full((n_rollout_threads, n_UAVs, episode_length, n_UAVs, 1), np.nan)
#
#     # 0: 没有数据, 1: 有数据但已发送, 2: 有数据且需要发送
#     observation_status = np.zeros((n_rollout_threads, n_UAVs, episode_length, n_UAVs), dtype=np.int8)
#
#     # Initialize with each UAV's own observations
#     for thread in range(n_rollout_threads):
#         for uav in range(n_UAVs):
#             knowledge_base[thread, uav, :, uav, 0] = obs_transposed[thread, uav, :, 0]
#             observation_status[thread, uav, :, uav] = 2  # Mark as "to send"
#
#     message_counts = np.zeros(max_iterations, dtype=np.int64)
#
#     for iteration in range(max_iterations):
#         messages_this_iteration = 0
#         updated_uavs = False
#
#         # Synchronous communication buffer
#         updates_buffer = {t: {r: [] for r in range(n_UAVs)} for t in range(n_rollout_threads)}
#         # 跟踪哪些发送者的状态需要从2更新为1
#         senders_status_updates = {t: {} for t in range(n_rollout_threads)}
#
#         # Process all threads and UAVs
#         for thread in range(n_rollout_threads):
#             # 查找需要发送数据的UAV（状态为2的）
#             senders_with_data = np.any(observation_status[thread] == 2, axis=(1, 2))
#             if not np.any(senders_with_data):
#                 continue
#
#             for sender in np.where(senders_with_data)[0]:
#                 data_to_send = observation_status[thread, sender] == 2
#                 if not np.any(data_to_send):
#                     continue
#                 # 记录需要将该发送者的哪些数据状态从2更新为1
#                 senders_status_updates[thread][sender] = data_to_send.copy()
#                 timesteps, sources = np.where(data_to_send)
#                 messages_this_iteration += len(timesteps)
#                 # Find all neighbors of this sender
#                 receivers = np.where(neighbor_masks[thread, sender])[0]
#                 if len(receivers) == 0:
#                     continue
#                 # For each receiver, collect updates
#                 for receiver in receivers:
#                     receiver_status = observation_status[thread, receiver]
#                     for i in range(len(timesteps)):
#                         timestep, source = timesteps[i], sources[i]
#                         if receiver_status[timestep, source] == 0:  # 接收者没有此数据
#                             value = knowledge_base[thread, sender, timestep, source, 0]
#                             updates_buffer[thread][receiver].append((timestep, source, value))
#                             updated_uavs = True
#
#         # Apply all updates synchronously
#         for thread in range(n_rollout_threads):
#             # 1. 先更新所有接收者的状态
#             for receiver, updates in updates_buffer[thread].items():
#                 for timestep, source, value in updates:
#                     knowledge_base[thread, receiver, timestep, source, 0] = value
#                     # 将接收者的观测状态更新为"有数据且需要发送"
#                     observation_status[thread, receiver, timestep, source] = 2
#             # 2. 更新所有发送者的状态（从2到1）
#             for sender, mask in senders_status_updates[thread].items():
#                 observation_status[thread, sender][mask] = 1
#
#         # Record message count for this iteration
#         message_counts[iteration] = messages_this_iteration
#
#         # If no new information was shared, we can terminate early
#         if not updated_uavs:
#             print(f"洪泛通信在{iteration + 1}次迭代后完成！")
#             message_counts = message_counts[:iteration + 1]
#             break
#     # 创建结果数组
#     final_observations = np.full((n_UAVs, episode_length, n_rollout_threads, n_UAVs, 1), np.nan)
#     # 首先创建一个转置后的视图，无需复制数据
#     # 从 (n_rollout_threads, n_UAVs, episode_length, n_UAVs, 1)
#     # 到 (n_UAVs, episode_length, n_rollout_threads, n_UAVs, 1)
#     transposed_view = np.transpose(knowledge_base, (1, 2, 0, 3, 4))
#     # 直接复制有效数据（非NaN的值）
#     # 使用布尔掩码避免循环
#     mask = ~np.isnan(transposed_view)
#     final_observations[mask] = transposed_view[mask]
#     return final_observations, message_counts

def run_flooding_algorithm(positions, local_observations, neighbor_distance, max_iterations):
    """
    Run highly optimized flooding communication protocol with synchronous communication.
    Uses vectorized operations for significant performance improvements.

    Args:
        positions: UAV positions with shape (n_rollout_threads, n_UAVs, 2)
        local_observations: Local observations with shape (n_UAVs, episode_length, n_rollout_threads, 1)
        neighbor_distance: Maximum distance for two UAVs to be neighbors
        max_iterations: Maximum number of flooding iterations

    Returns:
        tuple: (updated_observations, message_counts)
    """
    n_UAVs, episode_length, n_rollout_threads, _ = local_observations.shape

    # Reshape observations to (n_rollout_threads, n_UAVs, episode_length)
    obs_values = np.transpose(local_observations, (2, 0, 1, 3))[:, :, :, 0]

    # Compute neighbor masks (vectorized) - shape: (n_rollout_threads, n_UAVs, n_UAVs)
    neighbor_masks = np.zeros((n_rollout_threads, n_UAVs, n_UAVs), dtype=bool)
    for thread in range(n_rollout_threads):
        distances = cdist(positions[thread], positions[thread])
        neighbor_masks[thread] = (distances <= neighbor_distance) & (distances > 0)

    # Knowledge tracking with boolean arrays:
    # knowledge[thread, receiver, timestep, source] = True means receiver knows source's data at timestep
    knowledge = np.zeros((n_rollout_threads, n_UAVs, episode_length, n_UAVs), dtype=bool)

    # Initialize: each UAV knows its own observations
    for thread in range(n_rollout_threads):
        for uav in range(n_UAVs):
            knowledge[thread, uav, :, uav] = True

    # Create a 5D array to store all observation values
    # Shape: (n_rollout_threads, n_UAVs, episode_length, n_UAVs)
    # This stores which values each UAV knows about each other UAV's observations
    all_values = np.full((n_rollout_threads, n_UAVs, episode_length, n_UAVs), np.nan)

    # Initialize with own observations
    for thread in range(n_rollout_threads):
        for uav in range(n_UAVs):
            all_values[thread, uav, :, uav] = obs_values[thread, uav, :]

    message_counts = np.zeros(max_iterations, dtype=np.int64)

    # Track what knowledge needs to be sent in the next iteration
    to_send = knowledge.copy()

    for iteration in range(max_iterations):
        # Count messages in this iteration
        messages_this_iteration = np.sum(to_send)
        message_counts[iteration] = messages_this_iteration

        if messages_this_iteration == 0:
            print(f"Flooding completed after {iteration} iterations!")
            message_counts = message_counts[:iteration]
            break

        # Prepare next round buffer
        next_to_send = np.zeros_like(knowledge, dtype=bool)

        # Vectorized message passing
        for thread in range(n_rollout_threads):
            for sender in range(n_UAVs):
                # Skip if nothing to send
                if not np.any(to_send[thread, sender]):
                    continue

                # Get all receivers (neighbors)
                receivers = np.where(neighbor_masks[thread, sender])[0]
                if len(receivers) == 0:
                    continue

                # For each receiver
                for receiver in receivers:
                    # Find what sender knows that receiver doesn't
                    new_knowledge = to_send[thread, sender] & ~knowledge[thread, receiver]

                    if np.any(new_knowledge):
                        # Get timesteps and sources for new knowledge
                        timesteps, sources = np.where(new_knowledge)

                        # Update receiver's knowledge
                        knowledge[thread, receiver, timesteps, sources] = True

                        # Transfer the actual values
                        all_values[thread, receiver, timesteps, sources] = all_values[
                            thread, sender, timesteps, sources]

                        # Mark to send in next iteration
                        next_to_send[thread, receiver, timesteps, sources] = True

        # Update to_send for next iteration
        to_send = next_to_send

        # Check if we're done (no new information to spread)
        if not np.any(next_to_send):
            print(f"Flooding completed after {iteration + 1} iterations!")
            message_counts = message_counts[:iteration + 1]
            break

    # Construct the final observation array - OPTIMIZED VECTORIZED VERSION
    final_observations = np.full((n_UAVs, episode_length, n_rollout_threads, n_UAVs, 1), np.nan)
    # Reshape knowledge and all_values to match final_observations shape
    # Transpose from (n_rollout_threads, n_UAVs(receiver), episode_length, n_UAVs(source))
    # to match (n_UAVs(receiver), episode_length, n_rollout_threads, n_UAVs(source))
    knowledge_transposed = np.transpose(knowledge, (1, 2, 0, 3))
    all_values_transposed = np.transpose(all_values, (1, 2, 0, 3))
    # Use broadcasting to fill final_observations in one vectorized operation
    # We need to add a dimension at the end to match the shape of final_observations
    final_observations[knowledge_transposed] = all_values_transposed[knowledge_transposed][..., np.newaxis]

    return final_observations, message_counts

# 示例用法
if __name__ == "__main__":
    # 示例参数
    # n_UAVs = 5
    # episode_length = 400
    # n_rollout_threads = 100
    neighbor_distance = 500
    max_iterations = 26
    all_consensus_errors = np.zeros(max_iterations)

    # # ************************************************** 25无人机，1000m，固定位置，看通信量的多少**************************
    # positions = np.array([[100, 100], [300, 100], [500, 100], [700, 100], [900, 100],
    #                       [100, 300], [300, 300], [500, 300], [700, 300], [900, 300],
    #                       [100, 500], [300, 500], [500, 500], [700, 500], [900, 500],
    #                       [100, 700], [300, 700], [500, 700], [700, 700], [900, 700],
    #                       [100, 900], [300, 900], [500, 900], [700, 900], [900, 900]])
    # positions = np.tile(positions[np.newaxis, ...], (128, 1, 1))
    # observations = np.load(r'F:\1.移动边缘计算\on-policy-separated-last-obs\onpolicy\scripts\train\plot_flood_consensus_data\synthetic_A25.npy')
    # n_rollout_threads = positions.shape[0]
    # mean_observations = np.mean(observations, axis=0, keepdims=True)
    # # (1, episode_length, n_rollout_threads, 1)
    # # 创建并运行网络
    # # start_time = time.time()
    # flooding_observations, flooding_message_counts = run_flooding_algorithm(positions, observations, neighbor_distance, max_iterations)
    # flooding_observations_mean = np.nanmean(flooding_observations, axis=-2)  # 每个无人机内部 对所有无人机的观测取均值
    # # (n_UAVs, episode_length, n_rollout_threads, 1)
    # flooding_errors = np.mean(np.linalg.norm(flooding_observations_mean - mean_observations, axis=0) / np.linalg.norm(
    #     observations - mean_observations, axis=0))
    # print('flooding_errors:', flooding_errors)
    #
    # for iteration in range(1, max_iterations):
    #     print(iteration, '次迭代：')
    #     start_time = time.time()
    #     consensus_observations, consensus_message_counts = run_consensus_algorithm(positions, observations, neighbor_distance, iteration)
    #     consensus_errors = np.mean(np.linalg.norm(consensus_observations - mean_observations, axis=0) / np.linalg.norm(observations - mean_observations, axis=0))
    #     all_consensus_errors[iteration] = consensus_errors
    #     elapsed_time = time.time() - start_time
    #     print(f"\n执行时间: {elapsed_time:.3f}秒, 消息数量: {consensus_message_counts}, 误差为：{consensus_errors:.6f}")
    # print(np.round(all_consensus_errors, 6))
    # 记录误差。25架无人机，1000m的范围，260的通信距离
    # [0.46707505, 0.3302556 , 0.26487994, 0.22407117, 0.19484623, 0.17219657, 0.1537441 , 0.13820253, 0.12481209, 0.11309216,
    #  0.10272033, 0.0934685 , 0.08516777, 0.07768796, 0.0709255 , 0.06479577, 0.05922821, 0.05416299, 0.0495487 , 0.04534071,
    #  0.0414999 , 0.03799173, 0.03478555, 0.03185397, 0.02917245, 0.02671889, 0.02447332, 0.02241771, 0.02053565, 0.01881227,
    #  0.01723401, 0.01578853, 0.01446455, 0.01325179, 0.01214087, 0.01112318, 0.01019088, 0.00933678, 0.0085543 , 0.00783743,
    #  0.00718066, 0.00657895, 0.00602766, 0.00552259, 0.00505984, 0.00463587, 0.00424743, 0.00389154, 0.00356547,]
    # 记录误差。25架无人机，1000m的范围，520的通信距离
    # [0.287176, 0.159823, 0.104013, 0.071206, 0.049721, 0.035045, 0.024826, 0.017640, 0.012557, 0.008948,
    # 0.006381, 0.004553, 0.003249, 0.002319, 0.001655, 0.001182, 0.000844, 0.000602, 0.000430, 0.000307,
    # 0.000219, 0.000156, 0.000112, 0.000080, 0.000057]
    # 记录误差。25架无人机，1000m的范围，780的通信距离

    # ********************上述数据画图
    plt.figure(figsize=(10, 6), dpi=100)
    consensus_errors_1 = np.array([1, 0.46707505, 0.3302556 , 0.26487994, 0.22407117, 0.19484623, 0.17219657, 0.1537441 , 0.13820253, 0.12481209, 0.11309216,
     0.10272033, 0.0934685 , 0.08516777, 0.07768796, 0.0709255 , 0.06479577, 0.05922821, 0.05416299, 0.0495487 , 0.04534071,
     0.0414999 , 0.03799173, 0.03478555, 0.03185397, 0.02917245, 0.02671889, 0.02447332, 0.02241771, 0.02053565, 0.01881227,
     0.01723401, 0.01578853, 0.01446455, 0.01325179, 0.01214087, 0.01112318, 0.01019088, 0.00933678])
    # 0.0085543 , 0.00783743
     # 0.00718066, 0.00657895, 0.00602766, 0.00552259, 0.00505984, 0.00463587, 0.00424743, 0.00389154, 0.00356547
    consensus_message_counts_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                                  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38])
    # 39, 40
    consensus_errors = np.array([1, 0.287176, 0.159823, 0.104013, 0.071206, 0.049721, 0.035045, 0.024826, 0.017640, 0.012557, 0.008948])
     # 0.006381, 0.004553, 0.003249, 0.002319, 0.001655, 0.001182, 0.000844, 0.000602, 0.000430, 0.000307,
     # 0.000219, 0.000156, 0.000112, 0.000080, 0.000057
    consensus_message_counts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25

    plt.plot(consensus_message_counts_1, consensus_errors_1, marker='*', markersize=6, linewidth=2, color='#e377c2',
             label='Consensus error (250m)')
    plt.plot(consensus_message_counts, consensus_errors, marker='o', markersize=6, linewidth=2, color='#1f77b4', label='Consensus error (500m)')

    plt.scatter(25, 0, marker='^', s=80, color='#d62728', label='Flooding error')
    plt.scatter(25, 0, marker='^', s=400, color='#d62728')
    # Set labels and title
    plt.xlabel('Communication data ($\\times \, b_y$ Bytes)', fontsize=14)
    plt.ylabel('Accuracy error', fontsize=14)
    # Set limits for better visualization
    plt.xlim(0, consensus_message_counts_1.max() * 1.05)
    plt.ylim(0, consensus_errors_1.max() * 1.1)

    # Add legend and grid
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)  # Explicitly add grid
    # Save the figure
    plt.tight_layout()
    plt.savefig('consensus_error3.png', dpi=300, bbox_inches='tight')
    # plt.savefig('consensus_error_analysis.pdf', bbox_inches='tight')
    print("图像已保存为 consensus_error3.png 和 consensus_error_analysis.pdf")
    plt.show()



    # # ****************************************************生成随机位置和观测用于演示***************************************
    # np.random.seed(42)
    # positions = np.random.rand(n_rollout_threads, n_UAVs, 2) * 600
    # observations = np.random.rand(n_UAVs, episode_length, n_rollout_threads, 1)
    # mean_observations = np.mean(observations, axis=0, keepdims=True)
    # *****共识
    # for iteration in range(1, max_iterations):
    #     start_time = time.time()
    #     consensus_observations, consensus_message_counts = run_consensus_algorithm(positions, observations, neighbor_distance, iteration)
    #     # (n_UAVs, episode_length, n_rollout_threads, 1)
    #     consensus_errors = np.mean(np.linalg.norm(consensus_observations - mean_observations, axis=0) / np.linalg.norm(observations - mean_observations, axis=0))
    #     elapsed_time = time.time() - start_time
    #     print(f"\n{iteration}次迭代的总执行时间: {elapsed_time:.3f}秒。误差为：{consensus_errors:.3f}。总消息数：{np.sum(consensus_message_counts):.3f}")
    # ****flood
    # start_time = time.time()
    # flooding_observations, flooding_message_counts = run_flooding_algorithm(positions, observations, neighbor_distance, max_iterations)
    # flooding_observations_mean = np.nanmean(flooding_observations, axis=-2)  # 每个无人机内部 对所有无人机的观测取均值
    # # (n_UAVs, episode_length, n_rollout_threads, 1)
    # flooding_errors = np.mean(np.linalg.norm(flooding_observations_mean - mean_observations, axis=0) / np.linalg.norm(observations - mean_observations, axis=0))
    # elapsed_time = time.time() - start_time
    # # print(f"\n{iteration}次迭代的总执行时间: {elapsed_time:.3f}秒。误差为：{flooding_errors:.3f}。总消息数：{np.sum(flooding_message_counts):.3f}")
    # print(f"\n误差为：{flooding_errors:.3f}。总消息数：{np.sum(flooding_message_counts):.3f}")

    # # *******************************************************读数据，然后计算通信的计算量和误差。****************************
    # all_flooding_errors = np.full(57, np.nan)
    # all_flooding_message_counts = np.full(57, np.nan)
    # for i, episode in enumerate(range(0, 285, 5)):
    #     positions = np.load(r'F:\1.移动边缘计算\on-policy-separated-last-obs\onpolicy\scripts\results\mec\mappo\check\run165' + '\\uav_positions_' + str(episode) + '.npy')
    #     observations = np.load(r'F:\1.移动边缘计算\on-policy-separated-last-obs\onpolicy\scripts\results\mec\mappo\check\run165' + '\local_advantages_' + str(episode) + '.npy')
    #     n_rollout_threads = positions.shape[0]
    #
    #     mean_observations = np.mean(observations, axis=0, keepdims=True)
    #     # (1, episode_length, n_rollout_threads, 1)
    #     # 创建并运行网络
    #     # start_time = time.time()
    #     flooding_observations, flooding_message_counts = run_flooding_algorithm(positions, observations, neighbor_distance, max_iterations)
    #     flooding_observations_mean = np.nanmean(flooding_observations, axis=-2)     # 每个无人机内部 对所有无人机的观测取均值
    #     # (n_UAVs, episode_length, n_rollout_threads, 1)
    #     flooding_errors = np.mean(np.linalg.norm(flooding_observations_mean - mean_observations, axis=0) / np.linalg.norm(observations - mean_observations, axis=0))
    #     # elapsed_time = time.time() - start_time
    #
    #     all_flooding_errors[i] = flooding_errors
    #     all_flooding_message_counts[i] = np.sum(flooding_message_counts) / n_rollout_threads
    #     print('第'+str(episode)+'个episode的flooding_errors:', flooding_errors)
    #     # print('flooding_errors:', flooding_errors)
    #     # print(f"\n执行时间: {elapsed_time:.3f}秒")
    #     # print(f"\n每次迭代的消息数量: {flooding_message_counts}")
    #     # print(f"总消息数: {np.sum(flooding_message_counts)}")
    # print('总的平均每个episode内flooding_errors:', np.nanmean(all_flooding_errors))
    # print('平均的平均每个episode内消息量message_counts:', np.nanmean(all_flooding_message_counts))
    #
    #
    # all_consensus_errors = np.full((max_iterations, 57), np.nan)
    # all_consensus_message_counts = np.full((max_iterations, 57), np.nan)
    # for iteration in range(1, max_iterations):
    #     print('第iteration:', iteration, '次迭代的数据：')
    #     for i, episode in enumerate(range(0, 285, 5)):
    #         positions = np.load(r'F:\1.移动边缘计算\on-policy-separated-last-obs\onpolicy\scripts\results\mec\mappo\check\run165' + '\\uav_positions_' + str(episode) + '.npy')
    #         observations = np.load(r'F:\1.移动边缘计算\on-policy-separated-last-obs\onpolicy\scripts\results\mec\mappo\check\run165' + '\local_advantages_' + str(episode) + '.npy')
    #         n_rollout_threads = positions.shape[0]
    #         mean_observations = np.mean(observations, axis=0, keepdims=True)
    #
    #         # start_time = time.time()
    #         consensus_observations, consensus_message_counts = run_consensus_algorithm(positions, observations, neighbor_distance, iteration)
    #         # (n_UAVs, episode_length, n_rollout_threads, 1)
    #         consensus_errors = np.mean(np.linalg.norm(consensus_observations - mean_observations, axis=0) / np.linalg.norm(observations - mean_observations, axis=0))
    #         # elapsed_time = time.time() - start_time
    #         all_consensus_errors[iteration, i] = consensus_errors
    #         all_consensus_message_counts[iteration, i] = np.sum(consensus_message_counts) / n_rollout_threads
    #         # print('第' + str(episode) + '个episode的consensus_errors:', all_consensus_errors[iteration, i])
    #         # print('consensus_errors:', consensus_errors)
    #         # print(f"\n执行时间: {elapsed_time:.3f}秒")
    #         # print(f"\n每次迭代的消息数量: {consensus_message_counts}")
    #         # print(f"总消息数: {np.sum(consensus_message_counts)}")
    #     print('每个episode内平均误差为：',np.nanmean(all_consensus_errors[iteration]))
    #     print('平均每个episode内消息量message_counts:', np.nanmean(all_consensus_message_counts[iteration]))


    # consensus_errors = np.array([0.37834, 0.27318, 0.22046, 0.18437, 0.15708, 0.13552, 0.11808, 0.10375,
    #                              0.09182, 0.08180, 0.07332, 0.06609, 0.05990, 0.05456, 0.04993, 0.045918,
    #                              0.04241, 0.03933, 0.03663, 0.03425, 0.03214, 0.03028, 0.02862, 0.02714])
    # consensus_message_counts = np.array([10000.0, 19981.25, 29962.5, 39943.75, 49925.0, 59906.25, 69887.5, 79868.75,
    #                                      89850.0, 99831.25, 109812.5, 119793.75, 129775.0, 139756.25, 149737.5, 159718.75,
    #                                      169700.0, 179681.25, 189662.5, 199643.75, 209625.0, 219606.25, 229587.5, 239568.75])
    # flooding_error = 0.01227
    # flooding_message_counts = 249005.208


    # **********************************************有通信量的数据了，最后的画图。*****************************
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import matplotlib as mpl
    #
    # # Set Chinese font support
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # # Data from the question
    # consensus_message_counts = np.array([0, 10000.        ,  19996.49122807,  29992.98245614,
    #     39989.47368421,  49985.96491228,  59982.45614035,  69978.94736842,
    #     79975.43859649,  89971.92982456,  99968.42105263, 109964.9122807 ,
    #    119961.40350877, 129957.89473684, 139954.38596491, 149950.87719298,
    #    159947.36842105, 169943.85964912, 179940.35087719, 189936.84210526,
    #    199933.33333333, 209929.8245614 , 219926.31578947, 229922.80701754,
    #    239919.29824561])
    # consensus_errors = np.array([1.0, 0.43273902, 0.31392401, 0.24847274, 0.20240464,
    #    0.16727261, 0.13953277, 0.11721161, 0.09903189, 0.08409623,
    #    0.07174319, 0.06147013, 0.05288704, 0.04568669, 0.03962429,
    #    0.03450298, 0.03016332, 0.02647536, 0.02333263, 0.0206475 ,
    #    0.01834757, 0.01637278, 0.01467316, 0.01320701, 0.01193942])
    #
    # flooding_error = 0.002579
    # flood_message_counts = 249784.978
    #
    # # Create a clean figure without using seaborn styles
    # plt.figure(figsize=(10, 6), dpi=100)
    #
    # # Create the main plot
    # plt.plot(consensus_message_counts, consensus_errors,
    #          marker='o', markersize=6, linewidth=2, color='#1f77b4',
    #          label='Consensus error')
    #
    # # # Add the horizontal flooding error line
    # # plt.axhline(y=flooding_error, linestyle='--', color='#d62728', linewidth=1.5,
    # #             label=f'Flooding error ({flooding_error})')
    # plt.axhline(y=flooding_error, linestyle='--', color='#d62728', linewidth=1.5,
    #             label=f'Flooding error')
    # plt.axvline(x=flood_message_counts, linestyle='--', color='k', linewidth=1.5,
    #             label=f'Flooding communication data/3')
    #
    # # Shade the region between the curves
    # plt.fill_between(consensus_message_counts,
    #                  consensus_errors,
    #                  flooding_error,
    #                  where=(consensus_errors > flooding_error),
    #                  interpolate=True, color='#1f77b4', alpha=0.2)
    #
    # # Set labels and title
    # plt.xlabel('Communication data', fontsize=14)
    # plt.ylabel('Accuracy error', fontsize=14)
    # plt.title('Consensus Algorithm Accuracy vs. Communication Data', fontsize=16)
    #
    # # Format x-axis ticks for better readability
    # plt.xticks(np.linspace(0, 250000, 6),
    #            labels=[f'{int(x / 1000)}k' for x in np.linspace(0, 250000, 6)],
    #            fontsize=12)
    # plt.yticks(fontsize=12)
    #
    # # Set limits for better visualization
    # plt.xlim(0, consensus_message_counts.max() * 1.05)
    # plt.ylim(0, consensus_errors.max() * 1.1)
    #
    # # Highlight the intersection point (if any)
    # for i in range(len(consensus_message_counts) - 1):
    #     if (consensus_errors[i] > flooding_error and consensus_errors[i + 1] < flooding_error) or \
    #             (consensus_errors[i] < flooding_error and consensus_errors[i + 1] > flooding_error):
    #         # Linear interpolation to find intersection
    #         x1, x2 = consensus_message_counts[i], consensus_message_counts[i + 1]
    #         y1, y2 = consensus_errors[i], consensus_errors[i + 1]
    #         x_intersect = x1 + (x2 - x1) * (flooding_error - y1) / (y2 - y1)
    #         plt.plot(x_intersect, flooding_error, 'o', markersize=8, color='red')
    #         plt.annotate(f'({x_intersect:.0f}, {flooding_error})',
    #                      xy=(x_intersect, flooding_error),
    #                      xytext=(x_intersect + 10000, flooding_error + 0.02),
    #                      arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
    #                      fontsize=10)
    # # Add legend and grid
    # plt.legend(fontsize=12, loc='upper right')
    # plt.grid(True, linestyle='--', alpha=0.7)  # Explicitly add grid
    # # # Add text annotation for summary
    # # plt.text(0.02, 0.02,
    # #          "随着通信数据量增加，共识算法精确度提高\n但始终高于Flooding算法精确度",
    # #          transform=plt.gca().transAxes, fontsize=10,
    # #          bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    # # Save the figure
    # plt.tight_layout()
    # plt.savefig('consensus_error_analysis.png', dpi=300, bbox_inches='tight')
    # # plt.savefig('consensus_error_analysis.pdf', bbox_inches='tight')
    # print("图像已保存为 consensus_error_analysis.png 和 consensus_error_analysis.pdf")
    # plt.show()
