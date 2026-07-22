import numpy as np
import time


def calculate_adjacency_matrix(positions, neighbor_distance):
    """
    Calculate adjacency matrix based on UAVs' positions and neighbor distance.

    Args:
        positions: UAV positions with shape (n_rollout_threads, n_UAVs, 2)
        neighbor_distance: Maximum distance for two UAVs to be neighbors

    Returns:
        Adjacency matrices with shape (n_rollout_threads, n_UAVs, n_UAVs)
    """
    n_rollout_threads, n_UAVs, _ = positions.shape
    adjacency_matrices = np.zeros((n_rollout_threads, n_UAVs, n_UAVs))

    for env_idx in range(n_rollout_threads):
        env_positions = positions[env_idx]  # (n_UAVs, 2)

        # Calculate pairwise distances using broadcasting
        diff = env_positions[:, np.newaxis, :] - env_positions[np.newaxis, :, :]  # (n_UAVs, n_UAVs, 2)
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (n_UAVs, n_UAVs)

        # Create adjacency matrix (1 if neighbors, 0 otherwise)
        adjacency = (distances <= neighbor_distance).astype(float)
        np.fill_diagonal(adjacency, 0)  # Remove self-connections

        adjacency_matrices[env_idx] = adjacency

    return adjacency_matrices


def compute_metropolis_weights(adjacency_matrices):
    """
    Compute Metropolis weights based on adjacency matrices.

    Args:
        adjacency_matrices: Adjacency matrices with shape (n_rollout_threads, n_UAVs, n_UAVs)

    Returns:
        Weight matrices with shape (n_rollout_threads, n_UAVs, n_UAVs)
    """
    n_rollout_threads, n_UAVs, _ = adjacency_matrices.shape
    weight_matrices = np.zeros((n_rollout_threads, n_UAVs, n_UAVs))

    for env_idx in range(n_rollout_threads):
        adjacency_matrix = adjacency_matrices[env_idx]
        # Calculate degree (number of neighbors) for each UAV
        degrees = np.sum(adjacency_matrix, axis=1)
        # Compute Metropolis weights
        max_degrees_matrix = np.maximum.outer(degrees, degrees)
        weights_matrix = np.where(adjacency_matrix > 0,
                                  1.0 / (1.0 + max_degrees_matrix),
                                  0.0)
        # Calculate self-weights to ensure row stochasticity
        diagonal_values = 1.0 - np.sum(weights_matrix, axis=1)
        np.fill_diagonal(weights_matrix, diagonal_values)

        weight_matrices[env_idx] = weights_matrix
    return weight_matrices

def run_consensus_algorithm(positions, local_observations, neighbor_distance, max_iterations):
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
    n_UAVs, episode_length, n_rollout_threads, _ = local_observations.shape

    # 重新组织观测数据，使计算更直接
    # 新形状: (n_rollout_threads, n_UAVs, episode_length, 1)
    obs = np.transpose(local_observations, (2, 0, 1, 3))
    # 初始化共识估计值
    consensus_estimates = np.copy(obs)

    # Compute adjacency matrices and weight matrices
    adjacency_matrices = calculate_adjacency_matrix(positions, neighbor_distance)
    weight_matrices = compute_metropolis_weights(adjacency_matrices)
    # Calculate true global average for reference (not used in algorithm, just for evaluation)
    global_average = np.mean(consensus_estimates, axis=1, keepdims=True)  # (n_rollout_threads, 1, episode_length, 1)

    # 原始误差
    original_error = np.mean(np.linalg.norm(consensus_estimates - global_average, axis=1))
    print(f"原始误差Error: {original_error:.6f}")
    # Run consensus iterations
    for iteration in range(max_iterations):
        for env_idx in range(n_rollout_threads):
            weights = weight_matrices[env_idx]  # (n_UAVs, n_UAVs)
            # 使用矩阵乘法执行共识迭代
            thread_estimates = consensus_estimates[env_idx]  # (n_UAVs, episode_length, 1)

            # 使用矩阵乘法进行更新 (n_UAVs, n_UAVs) @ (n_UAVs, episode_length, 1)
            # 重塑为2D进行矩阵乘法，然后恢复原始形状
            reshaped_estimates = thread_estimates.reshape(n_UAVs, -1)  # (n_UAVs, episode_length*1)
            updated_estimates = weights @ reshaped_estimates  # (n_UAVs, episode_length*1)
            thread_estimates = updated_estimates.reshape(n_UAVs, episode_length, 1)

            consensus_estimates[env_idx] = thread_estimates

        # Calculate convergence error (for monitoring purposes)
        consensus_errors = np.mean(np.linalg.norm(consensus_estimates - global_average, axis=1) / np.linalg.norm(obs - global_average, axis=1))
        errors = np.mean(np.linalg.norm(consensus_estimates - global_average, axis=1))
        print(f"Iteration {iteration + 1}/{max_iterations}, 相对原始的误差比Average Error: {consensus_errors:.6f}, 实际误差Error: {errors:.6f}")
    # 转置回原始格式: (n_UAVs, episode_length, n_rollout_threads, 1)
    result = np.transpose(consensus_estimates, (1, 2, 0, 3))
    return result


def verify_consensus(original_obs, consensus_results):
    """
    验证共识结果与真实全局平均值的接近程度。

    Args:
        original_obs: 原始观测值，形状为 (n_UAVs, episode_length, n_rollout_threads, 1)
        consensus_results: 共识后的结果，具有相同形状

    Returns:
        共识结果与真实全局平均值之间的平均误差
    """
    # 计算UAVs之间的真实全局平均值
    true_average = np.mean(original_obs, axis=0, keepdims=True)  # shape: (episode_length, n_rollout_threads, 1)

    # 计算所有UAVs的平均误差
    error = np.mean(np.abs(consensus_results - true_average))

    return error, true_average


# Example usage
if __name__ == "__main__":
    # Example parameters
    n_UAVs = 25
    episode_length = 400
    n_rollout_threads = 4
    neighbor_distance = 390
    max_iterations = 20

    # 生成随机位置和观测用于演示
    np.random.seed(42)
    positions = np.random.rand(n_rollout_threads, n_UAVs, 2) * 900
    # 注意观测数据的形状现在是 (n_UAVs, episode_length, n_rollout_threads, 1)
    observations = np.random.rand(n_UAVs, episode_length, n_rollout_threads, 1)
    mean_observations = np.mean(observations, axis=0, keepdims=True)

    start_time = time.time()
    consensus_results = run_consensus_algorithm(
        positions, observations, neighbor_distance, max_iterations)
    execution_time = time.time() - start_time

    # 计算所有UAVs的原始误差
    error = np.mean(np.linalg.norm(observations - mean_observations, axis=0))
    print(f"原始误差: {error:.6f}")

    # 验证结果
    avg_error, true_average = verify_consensus(observations, consensus_results)
    print(f"平均误差: {avg_error:.6f}")
    print(f"执行时间: {execution_time:.6f} 秒")


    # 检查网络连通性
    disconnected_counts = []
    for thread in range(n_rollout_threads):
        pos = positions[thread]
        sq_dists = np.sum((pos[:, np.newaxis, :] - pos[np.newaxis, :, :]) ** 2, axis=2)
        distances = np.sqrt(sq_dists)
        adjacency = (distances < neighbor_distance) & (distances > 0)
        degrees = np.sum(adjacency, axis=1)
        disconnected = np.where(degrees == 0)[0]
        disconnected_counts.append(len(disconnected))
        if len(disconnected) > 0:
            print(f"线程 {thread} 有断开连接的UAV: {len(disconnected)}个")

    print(f"断开连接的UAV平均数: {np.mean(disconnected_counts):.2f}")