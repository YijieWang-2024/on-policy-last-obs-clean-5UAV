import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Set random seed for reproducibility
np.random.seed(42)


# Define the plot area and data
def visualize_uav_service(uav_positions, gu_positions, acts):
    """
    Visualize UAV service assignment without using PathEffects

    Parameters:
    uav_positions: numpy array of shape [4, 2] - positions of 4 UAVs
    gu_positions: numpy array of shape [40, 2] - positions of 40 ground users
    acts: numpy array of shape [4, 40] - service matrix (1 if UAV i serves user j)
    """
    Cover_R = 100
    x_max_uav = 650
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, x_max_uav)
    ax.set_ylim(0, x_max_uav)
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')

    # Add grid with light color
    ax.grid(True, alpha=0.3, linestyle='--')

    # Define colors for each UAV
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # Blue, Green, Red, Purple

    # Plot UAVs
    for i, (x, y) in enumerate(uav_positions):
        # Draw coverage area
        coverage = Circle((x, y), Cover_R, color=colors[i%4], alpha=0.1)
        ax.add_patch(coverage)

        # Draw UAV with a black edge instead of using PathEffects
        ax.scatter(x, y, marker='X', color=colors[i%4], s=300)

        # Add UAV label with a white background instead of using PathEffects
        ax.text(x + 10, y + 10, f"UAV{i + 1}", fontweight='bold', color=colors[i%4],
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2',
                          edgecolor='none'))

    # Plot ground users
    for j, (x, y) in enumerate(gu_positions):
        if acts is not None:
            served = False
            for i in range(4):
                if acts[i, j] == 1:
                    # Draw user with the color of its serving UAV
                    ax.scatter(x, y, marker='o', color=colors[i%4], s=80)
                    served = True
                    break
            if not served:
                # Unserved users are black
                ax.scatter(x, y, marker='o', color='black', s=80)
        else:
            ax.scatter(x, y, marker='o', color='black', s=80)

    # Create legend entries manually
    legend_elements = []
    for i in range(4):
        # Add UEs served by each UAV
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=colors[i%4], markersize=10,
                                          label=f'UEs served by UAV {i + 1}'))
    # Add locally computing users to legend
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='black', markersize=8,
                                      label='UEs performing local computation'))

    # Add legend with shadow effect
    # ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    ax.legend(handles=legend_elements, framealpha=0.9)

    # Add title
    plt.title('UAV Service Assignment',
              fontsize=16, pad=20)
    plt.tight_layout()
    return fig


def generate_sample_data():
    # Sample UAV positions
    uav_positions = np.array([
        [100, 100],  # UAV 1
        [250, 100],  # UAV 2
        [100, 250],  # UAV 3
        [250, 250]  # UAV 4
    ])
    uav_positions = np.array([
        [75, 150],  # UAV 1
        [225, 150],  # UAV 2
        [600, 500],  # UAV 3
        [500, 600],  # UAV 4
        [230, 230],  # UAV 4
        [230+165, 230+165],  # UAV 4
        [230+165*2, 230+165*2],  # UAV 4
    ])
    n_UAVs = uav_positions.shape[0]
    x_max = 300
    n_GUs= 30
    gu_positions = np.random.uniform(0, x_max, (n_GUs, 2))

    # # Generate ground user positions
    # gu_positions = np.zeros((n_GUs, 2))
    # # Users around different UAVs
    # for i in range(4):
    #     start_idx = i * 10
    #     end_idx = start_idx + 10
    #     center_x, center_y = uav_positions[i]
    #     spread = 80
    #
    #     for j in range(start_idx, end_idx):
    #         gu_positions[j] = [
    #             np.random.normal(center_x, spread),
    #             np.random.normal(center_y, spread)
    #         ]
    # # Ensure all points are within bounds
    # gu_positions = np.clip(gu_positions, 50, 850)

    # Create service assignments based on proximity
    acts = np.zeros((n_UAVs, n_GUs))

    for j in range(n_GUs):
        user_pos = gu_positions[j]
        distances = [np.linalg.norm(user_pos - uav_pos) for uav_pos in uav_positions]

        # Assign to closest UAV if within range
        min_idx = np.argmin(distances)
        if distances[min_idx] < 200 or np.random.random() < 0.7:  # Some chance to be served even if far
            acts[min_idx, j] = 1

    return uav_positions, gu_positions, acts


# Generate and visualize data
if __name__ == "__main__":
    uav_positions, gu_positions, acts = generate_sample_data()
    fig = visualize_uav_service(uav_positions, gu_positions, acts)

    try:
        # Save figure
        plt.savefig('uav_service_assignment.png', dpi=300, bbox_inches='tight')
        print("图像保存成功!")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    plt.show()