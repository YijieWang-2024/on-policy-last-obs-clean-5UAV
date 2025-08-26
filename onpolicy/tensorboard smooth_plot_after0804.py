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
    if tag == 'system_utility':
        gamma_r = 4
        lambda_r = 0.0001
        n_GUs = 40
        episode_length = 400
        all_values = []
        for tag_unity in ['delay_true_all_GUs', 'energy_all_GUs_UAVs']:
            log_dir_unity = log_dir.split('agent')[0] + 'agent0/' + tag_unity + '/agent0/' + tag_unity
            event_files = glob.glob(os.path.join(log_dir_unity, 'events.out.tfevents.*'))
            if not event_files:
                print(f"警告: 在 {log_dir} 中未找到事件文件")
                return [], []
                # 加载事件文件
            ea = event_accumulator.EventAccumulator(event_files[0], size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload()
            if tag_unity not in ea.Tags()['scalars']:
                tag_unity = log_dir.split('/')[-2] + '/' + tag_unity
            # 检查是否包含所需标签
            if tag_unity not in ea.Tags()['scalars']:
                print(f"警告: 在 {log_dir} 中未找到标签 '{tag_unity}'")
                return [], []
                # 获取数据
            events = ea.Scalars(tag_unity)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            all_values.append(values)
        all_values = np.array(all_values)
        all_values = gamma_r * (0.5-all_values[0]) *n_GUs*episode_length - lambda_r * all_values[1]*episode_length
        return steps, all_values.tolist()
    # event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*.user1'))
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"警告: 在 {log_dir} 中未找到事件文件")
        return [], []

    # 加载事件文件
    ea = event_accumulator.EventAccumulator(event_files[0], size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        tag = log_dir.split('/')[-2] + '/' +tag
    # 检查是否包含所需标签
    if tag not in ea.Tags()['scalars']:
        print(f"警告: 在 {log_dir} 中未找到标签 '{tag}'")
        return [], []

    # 获取数据
    events = ea.Scalars(tag)
    steps = [event.step for event in events]
    if tag.split('/')[-1] == 'cumulative_individual_reward_wo_cover':
        values = [4*event.value for event in events]
    else:
        values = [event.value for event in events]
    # # 查看数据。取前5e-7步数
    # steps = steps[:996]
    # values = values[:996]

    if tag.split('/')[-1] == 'delay_true_all_GUs':
        print(log_dir, np.mean(values[-200:]) * 1000)
        # steps = steps[-200:]
        # values = values[-200:]
    else:
        # print(log_dir, np.mean(values[-30:])) # UAVs
        # print(log_dir, np.mean(values[-200:])) # 带宽
        print(log_dir, np.mean(values[-3:])) # 随意
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
        # # 移动平均平滑
        # return pd.Series(values).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        s = pd.Series(values)
        result = np.zeros_like(values, dtype=float)
        for i in range(len(values)):
            # 计算窗口的起始和结束位置
            half = window_size // 2
            start = max(0, i - half)
            end = min(len(values), i + half + 1)
            # 使用实际可用的数据计算平均值
            result[i] = s.iloc[start:end].mean()
        return result


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
        # "./scripts/results/mec/mappo/check/run206/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run207/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run208/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
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
        # "./scripts/results/mec/mappo/check/run28/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run24/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run25/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run26/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run27/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run29/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run30/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run31/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run32/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run33/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run34/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run35/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run36/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run37/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run38/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run39/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run40/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run42/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run43/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run44/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run45/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run46/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run47/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run48/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run49/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
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
        # "./scripts/results/mec/mappo/check/run90/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run91/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run92/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run93/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run96/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run95/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run98/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run99/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run100/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run101/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
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
        # "./scripts/results/mec/mappo/check/run113/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run114/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run115/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run116/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run117/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run118/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
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
        # "./scripts/results/mec/mappo/check/run185/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run186/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run187/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run188/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run189/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run193/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run195/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run196/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run197/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run198/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run199/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run200/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run201/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run202/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run203/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run204/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run205/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run206/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run207/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run208/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run209/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run210/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run211/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run356/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run150/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run143/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run155/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run60/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run61/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run62/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run63/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run64/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run65/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run102/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run103/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run104/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run105/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run106/logs/agent0/system_utility/agent0/system_utility",
        # "./scripts/results/mec/mappo/check/run160/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run182/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # # "./scripts/results/mec/mappo/check/run183/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # # "./scripts/results/mec/mappo/check/run184/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run185/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run186/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run187/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run188/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run189/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run190/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run191/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run192/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run193/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
        # "./scripts/results/mec/mappo/check/run194/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",


        # "./scripts/results/mec/mappo/check/run164/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run166/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run163/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run164/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run166/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",

        # "./scripts/results/mec/mappo/check/run201/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run202/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run203/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",

        # "./scripts/results/mec/mappo/check/run96/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/energy_true_all_GUs/agent0/energy_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run102/logs/agent0/cumulative_individual_reward_wo_cover/agent0/cumulative_individual_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run103/logs/agent0/cumulative_individual_reward_wo_cover/agent0/cumulative_individual_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run104/logs/agent0/cumulative_individual_reward_wo_cover/agent0/cumulative_individual_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run105/logs/agent0/cumulative_individual_reward_wo_cover/agent0/cumulative_individual_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run106/logs/agent0/cumulative_individual_reward_wo_cover/agent0/cumulative_individual_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run109/logs/agent0/cumulative_reward_wo_cover/agent0/cumulative_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run110/logs/agent0/cumulative_reward_wo_cover/agent0/cumulative_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run111/logs/agent0/cumulative_reward_wo_cover/agent0/cumulative_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run112/logs/agent0/cumulative_reward_wo_cover/agent0/cumulative_reward_wo_cover",
        # "./scripts/results/mec/mappo/check/run113/logs/agent0/cumulative_reward_wo_cover/agent0/cumulative_reward_wo_cover",

        # "./scripts/results/mec/mappo/check/run16/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run17/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run18/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run19/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run60/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run61/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run62/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run63/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run64/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run65/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run66/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run67/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run102/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run103/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run104/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run105/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run106/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run109/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run110/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run111/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run112/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run113/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run141/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run142/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run143/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run144/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run145/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run171/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run160/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run151/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run143/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run172/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run173/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run174/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run175/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run176/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run177/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run178/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run179/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run185/logs/agent0/cumulative_reward/agent0/cumulative_reward",

        # "./scripts/results/mec/mappo/check/run155/logs/agent0/system_performance_individual/agent0/system_performance_individual",
        # "./scripts/results/mec/mappo/check/run155/logs/agent1/system_performance_individual/agent1/system_performance_individual",
        # "./scripts/results/mec/mappo/check/run155/logs/agent2/system_performance_individual/agent2/system_performance_individual",
        # "./scripts/results/mec/mappo/check/run155/logs/agent3/system_performance_individual/agent3/system_performance_individual",

        # "./scripts/results/mec/mappo/check/run163/logs/agent0/system_performance_individual/agent0/system_performance_individual",
        # "./scripts/results/mec/mappo/check/run163/logs/agent1/system_performance_individual/agent1/system_performance_individual",
        # "./scripts/results/mec/mappo/check/run163/logs/agent2/system_performance_individual/agent2/system_performance_individual",
        # "./scripts/results/mec/mappo/check/run163/logs/agent3/system_performance_individual/agent3/system_performance_individual",
    ]
    experiment_names = [
        # # 和上个一样。DC-PPO，不平均。
        # "205",      # 1跳
        # "206",      # 2跳
        # "207",      # 3跳
        # "208",      # 4跳
        # 减小覆盖范围，增加无人机数目。
        # "9",        # self
        # "10",       # 1跳
        # "11",       # 2跳
        # "12",       # 4跳
        # "13",       # 8跳
        # 加上平均的跳
        # "14",       # 1跳
        # "15",       # 2跳

        # # 改网络结构
        # "16-self-不平均",       # self
        # "17_1-hops-不平均",       # 1跳,L1
        # "18_2-hops-不平均",       # 2跳,L1
        # "19_3-hops-不平均",       # 3跳,L2
        # "20_4-hops",       # 4跳,L3
        # "21_5-hops",       # 5跳,L3
        # # local+ave_adv的一跳
        # "22",
        # # 平均V
        # "23",

        # 改网络结构的local+ave_adv
        # "28_1-hops-buffer平均",        # 1跳,L1
        # "24_2-hops-buffer平均",        # 2跳,L1
        # "25_3-hops-buffer平均",        # 3跳,L2
        # "26_4-hops",        # 4跳,L3
        # "27_5-hops",        # 5跳,L3
        # # 改进了下网络，看4能不能更好。。（结论：没用）
        # "32_4-hops",        # 4跳,L2——384
        # "33_4-hops",        # 4跳,L3——384
        # self。但是改代码也为local+ave_adv，对比下16
        # "34-self-buffer平均",           # self

        # # 改网络结构的local+True-mean
        # "35-self-local+True-mean",       # self
        # "36_1-hops-local+True",       # 1跳,L1
        # "37_2-hops-local+True",       # 2跳,L1
        # "38_3-hops-local+Tru",       # 3跳,L2
        # "39_4-hops",       # 4跳,L3
        # "40_5-hops",       # 5跳,L3

        # # 2跳邻居。要测试不同T的影响了
        # "42-T2",
        # "43-4",
        # "44-T8",
        # "45-16",
        # # 1跳邻居。测试不同T的影响
        # "46-8",
        # "47-20",
        # "48-64",
        # "49-150",

        # 不平均×2和Local+True-mean
        # # 195m，
        # "50-not-ave1",
        # "51-Local+True",
        # # 585m，
        # "52-not-ave3",
        # "53-Local+True",
        # # 1300m，
        # "54-not-ave5",
        # "55-Local+True",
        # # 585m的情况，不平均加声
        # "56-0.1",
        # "57-0.4",
        # "58-2",
        # "59-4",

        # # 改网络结构的。 rew+np.mean(rews)。（结论：无用！）
        # "29_1-hops",       # 1跳,L1
        # "30_2-hops",       # 2跳,L1
        # "31_3-hops",       # 3跳,L2

        # "155-U0",
        # "155-U1",
        # "155-U2",
        # "155-U3",


        # # Mec.py中把计算覆盖奖励的正确了。测试K
        # "60-1-hops",
        # "61-2-hops",
        # "62-3-hops",
        # "63-4-hops",
        # "64-5-hops",
        # "65-all-hops",

        # # 3跳加声
        # "66+0.8",
        # "67+1.6"

        # # 4架无人机
        # "68-self",       # self
        # "69-1-hops_L+M",       # 一跳的Local+True-mean
        # "70-1-hops_N",       # 一跳的不平均
        # "71-mappo",       # MAPPO的不平均
        # "72-ave_res",       # 一跳的不平均。平均资源
        # "73-fix-uav",       # 一跳的不平均。固定位置

        # # shuffle的4无人机
        # "74-1-hops",  #
        # "75-mappo",  #
        # "76-ave_res",  #
        # "77-fix-uav",  #
        # # 不要normer、不要shuffle.mlp-after-atten改为1
        # "78-1-hops",       #
        # "79-mappo",       #
        # "80-ave_res",       #
        # "81-fix-uav",       #
        # # Obs也不改顺序，不把自己提前了。
        # "82-1-hops",  #
        # "83-mappo",  #
        # "84-ave_res",  #
        # "85-fix-uav",  #
        # 观测所有，环境不shuffle，Obs不改顺序。 不优化处理动作。全局覆盖奖励
        # "86-1-hops",  #
        # "87-mappo",  #
        # "88-ave_res",  #
        # "89-fix-uav",  #
        #
        # "301-7月ave_res-1",
        # "150-7月1-hops",

        # 找到问题了，重新跑。4架无人机
        # "90-self",       # self
        # "91-1-hops_L+M",       # 一跳的Local+True-mean
        # "92-mappo_L+M",       # MAPPO
        # "93-ave_res",       # 一跳的不平均。平均资源
        # "96-fix-uav",       # 一跳的不平均。固定位置
        # "95-FPPO",          #

        # "97-roll-64",
        # "98-roll-32",

        # "99-lr-5e-4",
        # "100-lr-3e-5"

        # 并行64环境，4e-4的学习率
        # "101-1-hops_L+M",       # 一跳的不平均。
        # "102-1-hops_N",       # 一跳的不平均。
        # "103-mappo_N",       # MAPPO
        # "104-ave_res-coef0.01",       # 一跳的不平均。平均资源
        # "105-fix-uav-coef0.001",       # 一跳的不平均。固定位置
        # "106-FPPO",          #

        # "107-ave_res-coef0",
        # "108-fix-uav-coef0",

        # "109-1-hops_N",  # 一跳的不平均。
        # "110-mappo_N",       # MAPPO
        # "111-ave_res-coef0.01",  # 一跳的不平均。平均资源
        # "112-fix-uav-coef0.001",  # 一跳的不平均。固定位置
        # "113-FPPO",          #

        # 修改n_mini_batch和lr
        # "114-1-hops_N",  # 一跳的不平均。
        # "115-mappo_N",  # MAPPO
        # "116-ave_res-coef0.01",  # 一跳的不平均。平均资源
        # "117-fix-uav-coef0.01",  # 一跳的不平均。固定位置
        # "118-FPPO",  #
        # "119-ave_res-coef0",  # 一跳的不平均。平均资源
        # "120-fix-uav-coef0",  # 一跳的不平均。固定位置
        # "121-ave_res-coef0-lmda",  # 一跳的不平均。平均资源
        # "122-fix-uav-coef0-lmda",  # 一跳的不平均。固定位置

        # 改回3层网络（多观察、一次处理动作）
        # "123-1-hops_L+M",  # 一跳的不平均。
        # "124-mappo_L+M",  # MAPPO
        # "125-ave_res-coef0.01",  # 一跳的不平均。平均资源
        # "126-fix-uav-coef0.001",  # 一跳的不平均。固定位置
        # "127-FPPO",  #
        # "128-ave_res-coef0",  # 一跳的不平均。平均资源
        # "129-fix-uav-coef0",  # 一跳的不平均。固定位置
        # "130-ave_res-coef0-lmda",  # 一跳的不平均。平均资源
        # "131-fix-uav-coef0-lmda",  # 一跳的不平均。固定位置

        # 学习率也改回1e-4
        # "132-1-hops_Buf",  # 一跳的buffer平均。
        # "133-mappo_Buf",  # MAPPO
        # "134-ave_res-coef0.01",  # 一跳的buffer平均。平均资源00
        # "135-fix-uav-coef0.01",  # 一跳的buffer平均。固定位置
        # "136-FPPO",  #
        # "137-ave_res-coef0",  # 一跳的buffer平均。平均资源
        # "138-fix-uav-coef0",  # 一跳的buffer平均。固定位置
        # "139-ave_res-coef0-lmda",  # 一跳的buffer平均。平均资源
        # "140-fix-uav-coef0-lmda",  # 一跳的buffer平均。固定位置
        #
        # 自己的不平均。 F-PPO刚开始平均一下参数
        # "141-1-hops_N",  # 一跳的不平均。
        # "142-mappo_N",  # MAPPO的不平均。
        # "143-ave_res_N-coef0.01",  # 一跳的不平均。。平均资源       # 143和150基本一样。比150好一丢丢丢。 0000
        # "144-fix-uav_N-coef0.001",  # 一跳的不平均。。固定位置
        # "145-FPPO",  #
        # "146-FPPO-coef0.01",
        # "147-fix-uav-coef0.001",  # 一跳的buffer平均。。固定位置00

        # "148-FPPO-coef0.001",
        # 0.4声
        # "149-FPPO",
        # "150-ave_res_N-coef0.01",
        # 0.8声
        # "151-FPPO",   # 000000
        # "152-ave_res_N-coef0.01",
        # 1.2声
        # "153-ave_res_N-coef0.01",


        # 重新加声
        # 0.4声
        # "154-ave_res_N-coef0",
        # "155-fix-uav-N-coef0",
        # 0.8声
        # "156-ave_res_N-coef0",
        # "157-fix-uav-N-coef0",
        # 1.2声
        # "158-ave_res_N-coef0",
        # "159-fix-uav-N-coef0",

        # 自己最好的/
        # "160-1-hops_N",  # 一跳的不平均。141-1     # 00000
        # "161-mappo_N",  # MAPPO的不平均。142-m       # 00000
        # # 一跳的不平均。平均资源
        # "162-ave_res_N-coef0",
        #
        # # 上传两次修改动作，画示意图
        # "163-1-hops_N2",  # 一跳的不平均。
        # "164-ave_res_N-coef0-2",。和162一样

        # 163的各个无人机的奖励曲线对比
        # "UAV1",
        # "UAV2",
        # "UAV3",
        # "UAV4",

        # 重新加系数
        # # "166-ave_res_N-coef0.01",
        # # "167-fix-uav-N-coef0.01",
        # # "168-ave_res_N-coef0.001",
        # # "169-fix-uav-N-coef0.001",
        # # "170-mappo-N",
        # "171-mappo-L+M",
        # "160-1-hops_N",  # 一跳的不平均。141-1     # 00000
        # "151-FPPO",  # 000000
        # "143-ave_res_N-coef0.01",  # 一跳的不平均。。平均资源       # 143和150基本一样。比150好一丢丢丢。 0000
        # # "147-fix-uav-coef0.001",  # 一跳的buffer平均。。固定位置00

        # "172",    # 无覆盖奖励、无引导L+M
        # "173",      # 有覆盖奖励、无引导L+M
        # "174",      # 有覆盖奖励、有引导L+M
        # "175",      # 有覆盖奖励、有引导N
        # "176",    # 无覆盖奖励、有引导N

        # 不要覆盖奖励了，不符合系统gain。改了代码只有 飞行动作的熵计入训练。任务改为650-800，碰撞距离5m。
        # "177",      # 无引导N，熵0.005，有碰撞惩罚。纯探索     (探索不到)
        # "178",      # 无引导N，熵0.005，无碰撞惩罚。纯探索     (探索不到)
        # "179",      # 有引导N，熵0，有碰撞惩罚。
        # 观测信息多了无人机位置。在act.py中重新计算logp和entropy。主要是logp。任务回到700-800
        # "180",      # 无引导N，熵0，无碰撞惩罚。纯探索。     (探索不到)
        # "181",      # 有引导N，熵0，有碰撞惩罚。


        # # 重新回到均匀分布。没有覆盖奖励，完成任务有加了2*gamma_r的奖励。任务分布是675到800。
        # "182",      # delta2
        # # "183",      # delta8
        # # "184",      # delta16
        # "185",      # delta32
        # # 有碰撞。带覆盖奖励。Gammar的额外奖励也减为1.
        # "186",      # 新的局部覆盖奖励
        # "187",      # 新的全局覆盖奖励。
        # "188",      # 旧的全局惩罚

        # 还是不要覆盖奖励。 有碰撞，回到1*gamma-r的奖励，任务分布675-800
        # "189",      # delta2
        # "190",      # delta4
        # "191",      # delta8
        # "192",      # delta16
        # "193",      # delta32.        0000得是这个
        # "194",      # delta64

        # 开始测试K..16UAVs，800m，观测120m
        # "195-1-hops",      # k=1
        # "196-2-hops",      # k=2
        # "197-3-hops",      # k=3
        # "198-4-hops",      # k=4
        # "199-5-hops",      # k=5
        # # 加观测、不shuffle的简单。
        # "200-1-hops",      # k=1
        # "201-2-hops",      # k=2
        # "202-3-hops",      # k=3
        # "203-5-hops",      # k=5

        # # 9UAVs，650m，观测120m..多观测，不shuffle
        # "204-1-hops",      # k=1
        # "205-2-hops",      # k=2
        # "206-3-hops",      # k=3
        # "207-4-hops",      # k=4

        # 9UAVs， 650m，120m的少观测，shuffle。角落起飞
        "208-1-hops",
        "209-2-hops",
        "210-3-hops",
        "211-4-hops",


        #

        # "356",
        # "143",
        # "155",
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
