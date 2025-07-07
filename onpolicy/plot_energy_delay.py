import matplotlib.pyplot as plt
import numpy as np

# 数据
weight_w2 = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
energy_consumption = np.array([842.454, 766.697, 757.262, 752.520, 752.585, 750.543, 752.320])
execution_delay = np.array([400, 401.5, 405, 407, 405.5, 408.83, 413])

# 创建图形和第一个y轴
fig, ax1 = plt.subplots(figsize=(8, 6))

# 绘制能耗曲线（蓝色，左Y轴）
line1 = ax1.plot(weight_w2, energy_consumption, 'bo-', linewidth=2, markersize=6, label='Energy consumption')
ax1.set_xlabel('Weight w₂', fontsize=12)
ax1.set_ylabel('Energy consumption (J)', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(750, 848)
ax1.grid(True, alpha=0.3)

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制执行延迟曲线（红色，右Y轴）
line2 = ax2.plot(weight_w2, execution_delay, 'r*-', linewidth=2, markersize=8, label='Execution delay')
ax2.set_ylabel('Execution delay (ms)', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(395, 415)

# 设置X轴范围
ax1.set_xlim(0.01, 0.30)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
