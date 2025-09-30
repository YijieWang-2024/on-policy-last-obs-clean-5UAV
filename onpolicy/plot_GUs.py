import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['40', '50', '60', '70', '80']  # x轴标签 (Number of UAVs)
algorithms = ['MAPPO', 'DNC-PPO', 'F-PPO', 'IPPO']

# 每个算法在不同UAV数量下的Total cost数据
MAPPO_40GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run348.txt'))
MAPPO_50GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run349.txt'))
MAPPO_60GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run313.txt'))
MAPPO_70GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run350.txt'))
MAPPO_80GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run351.txt'))

DC_PPO_40GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_40GUs.txt'))
DC_PPO_50GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_50GUs.txt'))
DC_PPO_60GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_60GUs.txt'))
DC_PPO_70GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_70GUs.txt'))
DC_PPO_80GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_80GUs.txt'))

FPPO_40GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run364.txt'))
FPPO_50GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run365.txt'))
FPPO_60GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run344.txt'))
FPPO_70GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run360.txt'))
FPPO_80GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run361.txt'))

IPPO_40GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run358.txt'))
IPPO_50GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run359.txt'))
IPPO_60GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run316.txt'))
IPPO_70GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run362.txt'))
IPPO_80GUs = np.mean(np.loadtxt('./scripts/train/plot_data/system_gain_run363.txt'))

data = {
'MAPPO': [MAPPO_40GUs, MAPPO_50GUs, MAPPO_60GUs, MAPPO_70GUs, MAPPO_80GUs],
'DC-PPO': [DC_PPO_40GUs, DC_PPO_50GUs, DC_PPO_60GUs, DC_PPO_70GUs, DC_PPO_80GUs],
'F-PPO': [FPPO_40GUs, FPPO_50GUs, FPPO_60GUs, FPPO_70GUs, FPPO_80GUs],
'IPPO': [IPPO_40GUs, IPPO_50GUs, IPPO_60GUs, IPPO_70GUs, IPPO_80GUs],
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
plt.xlabel('Number of GUs', fontsize=12)
plt.ylabel('System gain', fontsize=12)
plt.xticks(x, categories)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

plt.savefig('./figures/GUs_comparison.eps', dpi=300, bbox_inches='tight', format='eps')
plt.savefig('./figures/GUs_comparison.png', dpi=300, bbox_inches='tight')
print(f"图表已保存到: ./figures/GUs_comparison.png")
# 显示图表
plt.show()
