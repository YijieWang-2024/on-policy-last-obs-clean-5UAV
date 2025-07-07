import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['10', '15', '20', '25']  # x轴标签 (Number of UAVs)
algorithms = ['DNC-PPO', 'MAPPO']

# 每个算法在不同UAV数量下的Total cost数据
# 这里的数据是从图中估计的，你需要替换为实际数据
data = {
    'DNC-PPO': [-15668.086,-7684.103,165.956,6461.670],
    'MAPPO': [-14587.296,-4718.286,1679.660,10339.516]
}

# 设置图形大小
plt.figure(figsize=(10, 6))

# 设置柱子的宽度和位置
x = np.arange(len(categories))
width = 0.22  # 柱子宽度
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
colors = ['#1f77b4', '#ff7f0e']

# 绘制每个算法的柱状图
for i, (algorithm, values) in enumerate(data.items()):
    offset = (i - len(algorithms)/2 + 0.5) * width
    plt.bar(x + offset, values, width, label=algorithm, color=colors[i])

# 设置图表属性
plt.xlabel('Number of UAVs', fontsize=12)
plt.ylabel('System performance', fontsize=12)
plt.xticks(x, categories)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3, axis='y')

# 设置y轴范围
# plt.ylim(0, 35)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()