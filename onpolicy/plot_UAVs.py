import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['3', '4', '5', '6', '7']  # x轴标签 (Number of UAVs)
algorithms = ['MAPPO', 'DNC-PPO', 'F-PPO', 'IPPO']

# 每个算法在不同UAV数量下的Total cost数据
MAPPO_3UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run338.txt'))
MAPPO_4UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run339.txt'))
MAPPO_5UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run313.txt'))
MAPPO_6UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run340.txt'))
MAPPO_7UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run341.txt'))

DC_PPO_3UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_3UAVs.txt'))
DC_PPO_4UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_4UAVs.txt'))
DC_PPO_5UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_5UAVs.txt'))
DC_PPO_6UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_6UAVs.txt'))
DC_PPO_7UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_7UAVs.txt'))

FPPO_3UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run354.txt'))
FPPO_4UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run355.txt'))
FPPO_5UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run344.txt'))
FPPO_6UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run356.txt'))
FPPO_7UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run357.txt'))

IPPO_3UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run342.txt'))
IPPO_4UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run343.txt'))
IPPO_5UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run316.txt'))
IPPO_6UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run346.txt'))
IPPO_7UAVs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run347.txt'))

data = {
'MAPPO': [MAPPO_3UAVs, MAPPO_4UAVs,MAPPO_5UAVs, MAPPO_6UAVs, MAPPO_7UAVs],
'DC-PPO': [DC_PPO_3UAVs, DC_PPO_4UAVs, DC_PPO_5UAVs, DC_PPO_6UAVs, DC_PPO_7UAVs],
'F-PPO': [FPPO_3UAVs, FPPO_4UAVs,FPPO_5UAVs, FPPO_6UAVs, FPPO_7UAVs],
'IPPO': [IPPO_3UAVs, IPPO_4UAVs,IPPO_5UAVs, IPPO_6UAVs, IPPO_7UAVs],
}

# 设置图形大小
plt.figure(figsize=(10, 6.6))

# 设置柱子的宽度和位置
x = np.arange(len(categories))
width = 0.22  # 柱子宽度
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#d62728']

# 绘制每个算法的柱状图
for i, (algorithm, values) in enumerate(data.items()):
    offset = (i - len(algorithms)/2 + 0.5) * width
    plt.bar(x + offset, values, width, label=algorithm, color=colors[i])

# 设置图表属性
plt.xlabel('Number of UAVs', fontsize=12)
plt.ylabel('System gain', fontsize=12)
plt.xticks(x, categories)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

plt.savefig('./figures/UAVs_comparison.eps', dpi=300, bbox_inches='tight', format='eps')
plt.savefig('./figures/UAVs_comparison.png', dpi=300, bbox_inches='tight')
print(f"图表已保存到: ./figures/UAVs_comparison.png")
# 显示图表
plt.show()
