import matplotlib.pyplot as plt
import numpy as np

x_max = 600

# uav_positions = np.array([[140,140], [300,140], [460, 140], [220, 260], [380, 260], [140, 380],[300, 380],[460, 380], [220, 500], [380,500]])
uav_positions = np.array([[120, 120], [300,120], [480, 120], [120, 300], [300, 300], [480, 300],[120, 480],[300, 480], [480, 480], [210,390]])
# uav_positions = np.array([[120, 120], [120,360], [120, 480], [360, 120], [360, 360], [480, 480]])
# uav_positions = np.array([[120, 120], [480, 120], [120, 480],[480, 480]])
# uav_positions = np.array([[250, 250 ], [750, 250 ], [500, 500 ],
#                                                [250, 750 ], [750, 750 ]], dtype=np.float)
# uav_positions = np.array([[190, 190 ], [500, 190], [810, 190],
#                                                [128, 500], [376, 500], [628, 500], [872, 500],
#                                                [190, 810], [500, 810], [810, 810]], dtype=np.float)
# uav_positions = np.array(
#                     [[128, 128], [376, 128], [628, 128], [872, 128],
#                      [128, 376], [376, 376], [628, 376], [872, 376],
#                      [128, 628], [376, 628], [628, 628], [872, 628],
#                      [190, 872],[500, 872], [810, 872]],
#                     dtype=np.float)
# uav_positions = np.array([[100, 128], [300, 128], [500, 128], [700, 128], [900, 128],
#                                                [100, 376], [300, 376],[500, 376],[700, 376],[900, 376],
#                                                [100, 628], [300, 628], [500, 628], [700, 628], [900, 628],
#                                                [100, 872], [300, 872], [500, 872], [700, 872], [900, 872]],
#                                               dtype=np.float)
# uav_positions = np.array(
#     [[100, 100], [300, 100], [500, 100], [700, 100],
#      [900, 100],
#      [100, 300], [300, 300], [500, 300], [700, 300],
#      [900, 300],
#      [100, 500], [300, 500], [500, 500], [700, 500],
#      [900, 500],
#      [100, 700], [300, 700], [500, 700], [700, 700],
#      [900, 700],
#      [100, 900], [300, 900], [500, 900], [700, 900],
#      [900, 900]],
#     dtype=np.float)
n_UAVs = uav_positions.shape[0]
Cover_R = 120
timestep = None
title = None
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0, x_max)
ax.set_ylim(0, x_max)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
# Plot UAVs and annotate their IDs
for i in range(n_UAVs):
    ax.scatter(uav_positions[i, 0], uav_positions[i, 1], c='r', label='UAV' if i == 0 else "")
    ax.annotate(f'UAV {i}', (uav_positions[i, 0], uav_positions[i, 1]))
    # Draw service range
    circle = plt.Circle((uav_positions[i, 0], uav_positions[i, 1]), Cover_R, color='r',
                        fill=False, linestyle='--')
    ax.add_patch(circle)

# # Plot GUs and annotate their IDs
# for j in range(self.n_GUs):
#     ax.scatter(self.gu_positions[j, 0], self.gu_positions[j, 1], c='b', label='GU' if j == 0 else "")
#     # Basic annotation
#     # annotation_text = f'GU {j}'
#     annotation_text = f'{j}'
#     if acts is not None:
#         serving_uavs = [i for i in range(n_UAVs) if acts[i, j] == 1]
#         if serving_uavs:
#             # annotation_text += f' (UAV {serving_uavs[0]})'
#             annotation_text += f'/ {serving_uavs[0]}'
#
#     ax.annotate(annotation_text, (self.gu_positions[j, 0], self.gu_positions[j, 1]))
# if title is not None:
#     plt.title(title)

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