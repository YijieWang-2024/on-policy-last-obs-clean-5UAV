import numpy as np

n_UAVs = 25
neighbor_distance = 390
positions = np.array(
                    [[90, 90], [270, 90], [450, 90], [630, 90], [810, 90],
                     [90, 270], [270, 270], [450, 270], [630, 270], [810, 270],
                     [90, 450], [270, 450], [450, 450], [630, 450], [810, 450],
                     [90, 630], [270, 630], [450, 630], [630, 630], [810, 630],
                     [90, 810], [270, 810], [450, 810], [630, 810], [810, 810]], dtype=np.float)
diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape: (n_UAVs, n_UAVs, 2)
distances = np.linalg.norm(diff, axis=2)  # shape: (n_UAVs, n_UAVs)

# Create adjacency matrix based on neighbor_distance
adjacency_matrix = (distances <= neighbor_distance) & (np.eye(n_UAVs) == 0)
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

# Calculate singular values of the weights matrix
singular_values = np.linalg.svd(weights_matrix, compute_uv=False)
# Sort singular values in descending order (should already be sorted)
sorted_singular_values = np.sort(singular_values)[::-1]
# The second largest singular value
second_largest_singular_value = sorted_singular_values[1]
print(f"The second largest singular value of the weights matrix is: {second_largest_singular_value}")
# You can also analyze spectral properties further
print(f"All singular values: {sorted_singular_values}")
print(f"Spectral gap (1 - second largest singular value): {1 - second_largest_singular_value}")

all_advantages = np.random.random((n_UAVs,1))
advantages_0_mean = np.mean(all_advantages)
error_0 = np.linalg.norm(all_advantages - advantages_0_mean)
updated_advantages = all_advantages.copy()
for step in range(1, 20):
    updated_advantages = np.matmul(weights_matrix, updated_advantages)
    # 记录下平均的进度的代码。
    error_step = np.linalg.norm(updated_advantages - advantages_0_mean)
    print('第',step,'步:', error_step/error_0)
