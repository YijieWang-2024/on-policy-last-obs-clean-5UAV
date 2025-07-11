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
        steps = steps[-200:]
        values = values[-200:]
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
        if tag == 'delay_true_all_GUs':
            steps = np.array(steps)
            steps = steps.reshape(-1, 5).mean(axis=1)
            values = np.array(values) * 1000
            values = values.reshape(-1, 5).mean(axis=1)
        all_values.extend(values)
        # 随机选择不同的颜色
        color = plt.cm.tab10(i % 10)
        if tag == 'delay_true_all_GUs':
            plt.plot(values, 'o', color=color, alpha=0.2, markersize=3)  # 绘制原始数据点(半透明)
        else:
            plt.plot(steps, values, 'o', color=color, alpha=0.2, markersize=3)      # 绘制原始数据点(半透明)
        smoothed_values = smooth_data(values, method=smooth_method)     # 添加平滑曲线
        current_linestyle = linestyles[i % len(linestyles)]
        if tag == 'delay_true_all_GUs':
            plt.plot(smoothed_values,linestyle=current_linestyle, color=color, linewidth=2.5, label=exp_name)
        else:
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

        # "./scripts/results/mec/mappo/check/run19/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run22/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run23/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run69/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run73/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # # "./scripts/results/mec/mappo/check/run109/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",

        # "./scripts/results/mec/mappo/check/run19/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run22/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run23/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",

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
        # "9",
        # "10",
        # "11",
        # "12",
        # "13",
        # "14",
        # "15",
        # "16",
        # "17",
        # "18",
        # "19",
        # "20",
        # "21",
        # "22",
        # "23",
        # "24",
        # "25",
        # "26",
        # "27",
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
