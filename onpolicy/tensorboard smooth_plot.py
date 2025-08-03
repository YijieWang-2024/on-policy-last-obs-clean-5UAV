import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.external.husl import lab_k
from tensorboard.backend.event_processing import event_accumulator
from scipy.signal import savgol_filter
import matplotlib


def load_tensorboard_data(log_dir, tag='system_performance'):
    # 查找事件文件
    # event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*.user1'))
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"警告: 在 {log_dir} 中未找到事件文件")
        return [], []

    # 加载事件文件
    ea = event_accumulator.EventAccumulator(event_files[0], size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        tag = 'agent0/'+tag
    # 检查是否包含所需标签
    if tag not in ea.Tags()['scalars']:
        print(f"警告: 在 {log_dir} 中未找到标签 '{tag}'")
        return [], []

    # 获取数据
    events = ea.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    if tag.split('/')[-1] == 'delay_true_all_GUs':
        print(log_dir, np.mean(values[-200:]) * 1000)
        # steps = steps[-200:]
        # values = values[-200:]
    else:
        # print(log_dir, np.mean(values[-30:])) # UAVs
        print(log_dir, np.mean(values[-200:])) # 带宽
    return steps, values


def smooth_data(values, method='savgol', window_length=51, polyorder=3, window_size=20):
    """
    使用指定方法平滑数据
    参数:
        values: 要平滑的数据
        method: 'savgol' 或 'moving_average'
        window_length: Savitzky-Golay过滤器窗口长度
        polyorder: Savitzky-Golay多项式阶数
        window_size: 移动平均窗口大小
    返回:
        平滑后的数据
    """
    if len(values) < max(window_length, window_size):
        if len(values) < 5:
            return values  # 数据点太少，不进行平滑处理
        # 调整窗口大小以适应数据量
        window_length = min(len(values) - 2, 11)  # 确保窗口长度是奇数且小于数据点数
        window_length = window_length - 1 if window_length % 2 == 0 else window_length
        polyorder = min(2, window_length - 1)
        window_size = min(len(values) // 2, 10)
    if method == 'savgol':
        # Savitzky-Golay滤波平滑
        return savgol_filter(values, window_length, polyorder)
    else:
        # 移动平均平滑
        return pd.Series(values).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values


def plot_smooth_comparison(log_dirs, experiment_names=None, tag='system_performance',
                           smooth_method='savgol', title='系统性能对比',
                           figsize=(12, 8), save_path=None):

    cn_font = matplotlib.font_manager.FontProperties(fname=".\SourceHanSansSC-Bold.otf")
    if experiment_names is None:
        experiment_names = [os.path.basename(path.rstrip('/')) for path in log_dirs]

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    # 存储所有数据，用于确定y轴范围
    all_values = []
    linestyles = ['-', '--', '-.', ':']
    for i, (log_dir, exp_name) in enumerate(zip(log_dirs, experiment_names)):
        steps, values = load_tensorboard_data(log_dir, tag)
        if not steps or not values:
            continue
        # if tag == 'delay_true_all_GUs':
        #     steps = np.array(steps)
        #     steps = steps.reshape(-1, 5).mean(axis=1)
        #     values = np.array(values) * 1000
        #     values = values.reshape(-1, 5).mean(axis=1)
        all_values.extend(values)
        # 随机选择不同的颜色
        color = plt.cm.tab10(i % 10)
        # if tag == 'delay_true_all_GUs':
        #     plt.plot(values, 'o', color=color, alpha=0.2, markersize=3)  # 绘制原始数据点(半透明)
        # else:
        #     plt.plot(steps, values, 'o', color=color, alpha=0.2, markersize=3)      # 绘制原始数据点(半透明)
        plt.plot(steps, values, 'o', color=color, alpha=0.2, markersize=3)      # 绘制原始数据点(半透明)
        smoothed_values = smooth_data(values, method=smooth_method)     # 添加平滑曲线
        current_linestyle = linestyles[i % len(linestyles)]
        # if tag == 'delay_true_all_GUs':
        #     plt.plot(smoothed_values,linestyle=current_linestyle, color=color, linewidth=2.5, label=exp_name)
        # else:
        #     plt.plot(steps, smoothed_values,linestyle=current_linestyle, color=color, linewidth=2.5, label=exp_name)
        plt.plot(steps, smoothed_values,linestyle=current_linestyle, color=color, linewidth=2.5, label=exp_name)

    # 设置图表样式
    plt.title(title, fontsize=16, fontproperties=cn_font)
    plt.xlabel('训练步数', fontsize=14, fontproperties=cn_font)
    plt.ylabel(tag, fontsize=14, fontproperties=cn_font)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best', prop=cn_font)
    # 优化y轴范围
    if all_values:
        min_val, max_val = min(all_values), max(all_values)
        margin = (max_val - min_val) * 0.1
        plt.ylim(min_val - margin, max_val + margin)
    plt.tight_layout()
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 指定TensorBoard日志目录
    # ************************* 利用actions[2]和[3]处理[1]，给mappo添加观测之后的算法对比 *************************
    log_dirs = [
        #
        # "./scripts/results/mec/mappo/check/run1/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run2/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run3/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run5/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run6/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run55/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run56/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run9/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run10/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run11/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run12/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run13/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run14/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run15/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run16/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run17/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run18/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run19/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run22/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run23/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run24/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run25/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run26/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run27/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run28/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run31/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run34/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run35/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run36/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run37/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run38/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run39/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run40/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run41/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run42/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run43/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run44/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run45/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run46/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run47/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run48/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run50/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run51/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run52/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run53/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run54/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run55/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run56/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run57/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run58/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run59/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run60/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run61/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run62/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run63/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run64/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run65/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run66/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run67/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run68/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run69/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run71/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run72/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run73/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run74/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run75/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run76/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run77/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run78/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run79/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run80/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run81/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run82/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run83/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run84/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run85/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run86/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run87/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run88/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run89/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run94/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run95/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run96/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run102/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run103/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run104/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run105/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run106/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run107/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run108/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run109/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run110/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run111/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run112/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run119/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run120/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run121/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run122/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run123/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run124/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run125/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run126/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run127/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run128/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run129/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run130/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run131/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run132/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run133/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run134/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run135/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run136/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run137/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run138/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run139/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run140/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run141/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run142/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run143/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run144/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run145/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run146/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run147/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run148/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run149/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run150/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run151/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run152/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run153/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run154/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run155/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run156/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run157/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run158/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run159/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run160/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run161/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run162/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run163/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run164/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run165/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run166/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run167/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run168/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run169/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run170/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run171/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run172/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run173/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run174/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run175/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run176/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run177/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run178/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run179/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run180/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run181/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run182/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run183/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run184/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run185/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run186/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run187/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run188/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run189/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run190/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run201/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run19/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run22/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run23/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run86/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run87/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run88/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run89/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run94/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run95/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run96/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run102/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run103/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run104/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run105/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run119/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run120/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run121/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run122/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run143/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run155/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run161/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run163/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run164/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run166/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run69/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run73/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run109/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run94/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run95/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run96/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run143/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run155/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run161/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run163/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run164/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run166/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",

        # "./scripts/results/mec/mappo/check/run19/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run22/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run23/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run66/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run67/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run68/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run69/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run71/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run72/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run73/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run74/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run75/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run76/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run77/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run78/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run79/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run80/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run81/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run82/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run83/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run84/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run85/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run86/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run87/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run88/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run89/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run94/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run95/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run96/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run102/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run103/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run104/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run105/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run106/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run107/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run108/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run109/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run110/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run111/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run112/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run119/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run120/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run121/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run122/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run201/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run202/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run203/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",

        # "./scripts/results/mec/mappo/check/run86/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run87/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run88/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run89/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run94/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run95/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run96/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run102/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run103/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run104/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run105/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run106/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run107/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run108/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run109/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run110/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run111/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run112/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run86/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run87/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run88/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run89/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run94/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run95/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run96/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run17/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run18/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run19/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run22/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run23/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run24/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run25/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run26/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run27/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run13/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run14/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run34/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run37/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run38/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run39/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run40/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run42/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run46/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run28/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run62/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run63/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run64/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run65/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run161/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run162/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run163/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run164/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run165/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run166/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run167/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run168/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run169/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run170/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    ]
    experiment_names = [
        # "19proposed",
        # "1",
        # "2",
        # "3",
        # "5",
        # "6",
        # # 随机种子为2
        # "55proposed",
        # # 随机种子为3
        # "63proposed",
        # # 只有飞行动作，只有not-coverd惩罚。
        # "28",
        # 只有飞行动作，正常计算奖励
        # "31_only_fly_16",
        # 平均资源处理动作
        # "34ave_proc_a_8",
        # "35ave_proc_a_16",
        # "36ave_proc_a_32",
        # 平均资源不处理动作
        # "37ave_n_proc_4",
        # "38ave_n_proc_8",
        # "39ave_n_proc_16",
        # "40ave_n_proc_32",
        # 所有动作，处理动作
        # "41all_proc_a_4",
        # "42all_proc_a_8",
        # "43all_proc_a_16",
        # "44all_proc_a_32",
        # 所有动作，不处理动作
        # "45all_n_proc_4",
        # "46all_n_proc_8",
        # "47all_n_proc_16",
        # "48all_n_proc_32",
        # # 平均未被服务用户的奖励。  把未被服务用户的奖励也给加进来了。
        # "50AveRew_averesou_1",
        # "51AveRew_averesou_2",
        # "52AveRew_averesou_4",
        # "53AveRew_allact_1",
        # "54AveRew_allact_2",
        # "55AveRew_allact_4",
        # 未被服务用户的奖励给到就近无人机。   把未被服务用户的奖励也给加进来了。
        # "56NeRew_averesou_1",
        # "57NeRew_averesou_2",
        # "58NeRew_averesou_4",
        # "59NeRew_allact_1",
        # "60NeRew_allact_2",
        # "61NeRew_allact_4",
        # "62",
        # "63",
        # "64",
        # "65",
        # # 全局计算未被覆盖区域
        # "66epsilon_0.5",
        # "67epsilon_1",
        # "68epsilon_2",
        # "69epsilon_4",
        # # 全局计算未被覆盖用户个数
        # "70alpha_0.5",
        # "71alpha_1",
        # "72alpha_2",
        # "73alpha_4",
        # # 分布式估计未被覆盖区域
        # "74epsilon_0.5",
        # "75epsilon_1",
        # "76epsilon_2",
        # "77epsilon_4",
        # 分布式估计未被覆盖用户个数
        # "78alpha_0.5",
        # "79alpha_1",
        # "80alpha_2",
        # "81alpha_4",
        # 从头开始简单。平均资源，全局未覆盖面积。
        # "82epsilon_32",
        # "83epsilon_64",
        # "84epsilon_128",
        # "85epsilon_200",
        # epsilon_32就很好。然后调整delta值。全局未覆盖面积的惩罚
        # "86delta_1",
        # "87delta_2",
        # "88delta_4",
        # "89delta_6",
        # alpha_32。520米。平均动作。全局未覆盖用户的惩罚
        # "94delta_1",
        # "95delta_2",
        # "96delta_4",
        # "97delta_6",
        # 260米，平均动作，全局覆盖惩罚
        # "102delta_1",
        # "103delta_2",
        # "104delta_4",
        # "105delta_6",
        # 260米，平均动作，局部估计覆盖惩罚
        # "106delta_1",
        # "107delta_2",
        # "108delta_4",
        # "109delta_6",
        # 260米，加了一位obs，卸载动作也加了0.5阈值。 平均带宽，全局覆盖惩罚。
        # "119delta_1",
        # "120delta_2",
        # "121delta_4",
        # "122delta_6",
        # 520米，全局动作咯，全局惩罚。
        # "110delta_1",
        # "111delta_2",
        # "112delta_4",
        # 取消了processed-action。试一下新改的mec-runner对不对。和103，104比较
        # "123delta_2",
        # "124delta_4",
        # 260米，加了一位obs，卸载动作也加了0.5阈值。 平均带宽，全局覆盖惩罚。  狄利克雷分布
        # "12*delta_1",
        # "12*delta_2",
        # "12*delta_4",
        # "127",    # 狄利克雷分布平均带宽仅把熵系数为0.。还没有全分资源
        # "129",    # 是103平均动作的熵系数为0版本          (103和129好)
        # "134",    # 熵系数为0了，logp乘的系数：0.01。全分资源
        # "135",    # 熵系数为0了，logp乘的系数：0.0001。全分资源
        # "136",    # 熵系数为0了，logp乘的系数：0.1。全分资源
        # "137",    # 熵系数为0了，logp乘的系数：0.1。全分资源..带act.py的0.5改变avail-actions (137好)
        # "141_gaussian",
        # 修改了local-obs的顺序,把大任务的用户放在前
        # "143_ave_resource",
        # 熵系数都为0,        ave_bandwidth..
        # "144",            # avail-actions=0的经验保留,logp乘的系数：0.1×××××××××××××××
        # "145_0.1",        #avail-actions=0的经验屏蔽掉,logp乘的系数：0.1
        # "146_0.1",                #avail-actions=0的经验屏蔽掉,logp乘的系数：0.1     all_action
        # "147_0.05",       #avail-actions=0的经验屏蔽掉,logp乘的系数：0.05
        # "148_0.2",        #avail-actions=0的经验屏蔽掉,logp乘的系数：0.2
        # "149_0.4",        #avail-actions=0的经验屏蔽掉,logp乘的系数：0.4
        # "150_0.2",                #avail-actions=0的经验屏蔽掉,logp乘的系数：0.2     all_action
        # "151_0.4",                #avail-actions=0的经验屏蔽掉,logp乘的系数：0.4     all_action
        # "152_0.8",                #avail-actions=0的经验屏蔽掉,logp乘的系数：0.8     all_action
        # "153_1",                  #avail-actions=0的经验屏蔽掉,logp乘的系数：1     all_action
        # 总时延大于0.5的就不要卸载了.
        # "154_ave_bandwidth",
        # "155_all",
        # 仅仅看处理时延大于0.5的就不要卸载了.
        # "156_ave_bandwidth",
        # "157_all",
        # individual-state不要无人机位置了
        # "158",
        # 估计全局奖励
        # "159",
        # "160",  # 估计时不要邻居惩罚。
                    # 权重0.01的场景。
        # "161",      # 平均资源
        # "162",      # 利用全部资源的 平均带宽。
        # "163",      # 利用全部资源的 所有动作。
                # 权重0.1的场景。
        # "164",  # 平均资源
        # "165",  # 利用全部资源的 平均带宽。
        # "166",  # 利用全部资源的 所有动作。
        # "167",      # 部分资源的 平均带宽。
        # "168",      # 部分资源的 所有动作。
        # "169",  # 部分资源的 平均带宽。
        # "170",  # 部分资源的 所有动作。
        # 带平均A的不同k
        # "175",
        # "171",
        # "172",
        # "173",
        # "174",
        # 不带平均A的不同k
        # "175",
        # "176",    # 1跳
        # "177",    # 2跳
        # "178",    # 3跳
        # 不带平均A的不同k。 inidvidual-state中不要无人机
        # "179",
        # "180",    # 1跳
        # "181",    # 2跳
        # "182",    # 3跳
        # # 初始在小范围的位置中随机找一个
        # "183",
        # "184",      # 1跳
        # "185",      # 2跳
        # "186",      # 3跳
        # 初始4角随机
        "187",
        "188",  # 1跳
        "189",  # 2跳
        "190",  # 3跳


        # 看103的轨迹图

        # 9无人机，600m，80用户。260m感知范围，全局动作。
        # "201_local_delta2",
        # "202_local_delta4",
        # "203_global_delta2",
        # "204_global_delta4",
    ]

    # 绘制平滑对比图，指定中文字体
    plot_smooth_comparison(
        log_dirs=log_dirs,
        experiment_names=experiment_names,
        tag=log_dirs[0].split('/')[-1],
        smooth_method='moving_average',  # 'savgol' 或 'moving_average'
        title=log_dirs[0].split('/')[-1],
        figsize=(12, 8),
        save_path=log_dirs[0].split('/')[-1] + str(experiment_names) + '.png'
    )
