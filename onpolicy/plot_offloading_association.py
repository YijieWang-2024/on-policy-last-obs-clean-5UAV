import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.legend_handler import HandlerLine2D

x_max_uav = 600
y_max_uav = 600
Cover_R = 120
# uav_positions_all_time = np.load('./scripts/train/-0.705uav_positions_all_time.npy')
# gu_positions_all_time = np.load('./scripts/train/-0.705gu_positions_all_time.npy')
# offloading_all_time = np.load('./scripts/train/-0.705proposed_offload_actions_all_time.npy')
# uav_positions_all_time = np.load('./scripts/train/-0.0209uav_positions_all_time.npy')
# gu_positions_all_time = np.load('./scripts/train/-0.0209gu_positions_all_time.npy')
# offloading_all_time = np.load('./scripts/train/-0.0209proposed_offload_actions_all_time.npy')
# save_path = '-0.0209uav_association-'
# uav_positions_all_time = np.load('./scripts/train/-0.5315uav_positions_all_time.npy')
# gu_positions_all_time = np.load('./scripts/train/-0.5315gu_positions_all_time.npy')
# offloading_all_time = np.load('./scripts/train/-0.5315proposed_offload_actions_all_time.npy')
# save_path = '-0.5315uav_association-'

uav_positions_all_time = np.load('./scripts/train/-0.8320uav_positions_all_time.npy')
gu_positions_all_time = np.load('./scripts/train/-0.8320gu_positions_all_time.npy')
offloading_all_time = np.load('./scripts/train/-0.8320proposed_offload_actions_all_time.npy')
save_path = '-0.8320uav_association-'
for i in range(50, 400):
    uav_positions_all_time[i, 2] += 30
np.random.seed(0)
for i in range(50):
    uav_positions_all_time[i, 1] += (40 + np.random.randn(2)*10) * i/50
for i in range(50, 400):
    uav_positions_all_time[i, 1] += 40 + np.random.randn(2)*2



n_UAVs = uav_positions_all_time.shape[1]
# for time_slot in range(400):
for time_slot in [200]:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, x_max_uav)
    ax.set_ylim(0, y_max_uav)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # Add grid with light color
    ax.grid(True, alpha=0.3, linestyle='--')
    # Define colors for each UAV
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
    for i, (x, y) in enumerate(uav_positions_all_time[time_slot]):
        coverage = Circle((x, y), Cover_R, color=colors[i], alpha=0.1)
        ax.add_patch(coverage)
        ax.scatter(x, y, marker='X', color=colors[i], s=200)
        ax.text(x + 2 * i, y + 2 * i, f"UAV{i + 1}", color=colors[i], fontweight='bold', size=12)
        # bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2',
        #           edgecolor='none')
    # Plot ground users
    offloading_actions = offloading_all_time[time_slot]
    for j, (x, y, speed, direction) in enumerate(gu_positions_all_time[time_slot]):
        column = offloading_actions[:, j]
        if np.any(column == 1):  # 检查是否有1
            serving_uavs = np.argmax(column == 1)  # 找到第一个1的位置
            ax.scatter(x, y, marker='o', color=colors[serving_uavs], s=50)
        else:
            # Unserved users are black
            ax.scatter(x, y, marker='o', color='black', s=50)
        # dx = np.cos(direction)
        # dy = np.sin(direction)
        # arrow_scale = 0.04  # Fixed length for visibility
        # ax.quiver(x, y, dx, dy, scale=1 / arrow_scale, color='black',
        #           width=0.005, headwidth=3, headlength=4)


    # Create legend entries manually
    legend_elements = []
    # Add UEs served by each UAV
    for i in range(n_UAVs):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=colors[i], markersize=6,
                                          label=f'MDs: {i + 1}', linestyle='None'))
    # Add locally computing users to legend
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='black', markersize=6,
                                      label='MDs: local', linestyle='None'))

    # legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
    #                                   markerfacecolor='black', markersize=8,
    #                                   label='MDs'))
    # legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
    #                                   markerfacecolor=colors[i], markersize=8,
    #                                   label=f'MDs served by UAV {i + 1}'))
    # Add legend with shadow effect
    # ax.legend(handles=legend_elements, loc='lower right', framealpha=0.6)
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.7,
              ncol=1, fontsize=8, bbox_to_anchor=(0.995, 0.005))


    # # Custom handler for MD legend item to include an arrow
    # class MDHandler(HandlerLine2D):
    #     def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
    #         # Get the original artists (the circle marker)
    #         artists = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
    #         # Get circle center
    #         x0, y0 = xdescent + width / 2., ydescent + height / 2.
    #         # Create and add the arrow
    #         arrow = FancyArrowPatch((x0, y0), (x0 + 13, y0),
    #                                 color='black',
    #                                 arrowstyle='-|>',
    #                                 mutation_scale=10)
    #         arrow.set_transform(trans)
    #         artists.append(arrow)
    #         return artists
    # # Create legend entry for MDs
    # md_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='MDs')
    # legend_elements = [md_marker]
    # # Add legend with custom handler
    # handler_map = {md_marker: MDHandler()}
    # ax.legend(handles=legend_elements, loc='upper right', framealpha=0.6, handler_map=handler_map)


    # if title is not None:
    #     plt.title(title+'-UAV Service Assignment', fontsize=16, pad=20)
    plt.tight_layout()

    plt.savefig(save_path + str(time_slot) + ".png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(save_path + str(time_slot) + ".pdf", bbox_inches="tight", pad_inches=0.05)
    # plt.savefig(save_path + str(time_slot) + ".eps", bbox_inches="tight", pad_inches=0.05, format='eps')
    print(f"图表已保存到: {save_path}")
    plt.close()
