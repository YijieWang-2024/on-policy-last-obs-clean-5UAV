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
    # log_dirs = [
    #     # "./scripts/results/mec/mappo/check/run64/logs/system_performance/system_performance",
    #     # "./scripts/results/mec/mappo/check/run65/logs/agent0/system_performance/agent0/system_performance",
    #     "./scripts/results/mec/mappo/check/run66/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run67/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run68/logs/agent0/system_performance/agent0/system_performance",
    #     "./scripts/results/mec/mappo/check/run69/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run70/logs/agent0/system_performance/agent0/system_performance",
    # ]
    #
    # # 可选: 指定实验名称
    # experiment_names = [
    #     # "共享网络。局部观测，全局状态、全局奖励的mappo1",
    #     # "独立网络。局部观测，全局状态、全局奖励的mappo2",
    #     "独立网络。局部观测和局部状态、局部奖励的ippo3",
    #     # "独立网络。局部观测和局部状态、局部奖励的ippo4。添加邻居观测",
    #     # "独立网络。局部观测和局部状态、局部奖励的ippo5。添加优势函数平均",
    #     "独立网络。局部观测和局部状态、局部奖励的ippo6。添加邻居观测和优势函数平均",
    #     # "独立网络。局部观测和全局状态、独立奖励的mappo3",
    # ]

    # log_dirs = [
    #     # "./scripts/results/mec/mappo/check/run73/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run74/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run75/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run76/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run77/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run78/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run79/logs/system_performance/system_performance",
    #     # "./scripts/results/mec/mappo/check/run80/logs/system_performance/system_performance",
    #     # "./scripts/results/mec/mappo/check/run84/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run85/logs/agent0/system_performance/agent0/system_performance",
    #     "./scripts/results/mec/mappo/check/run86/logs/agent0/system_performance/agent0/system_performance",
    #     "./scripts/results/mec/mappo/check/run87/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run88/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run89/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run90/logs/agent0/system_performance/agent0/system_performance",
    # ]
    # # 可选: 指定实验名称
    # experiment_names = [
    #     # "my",
    #     # "74",       # 不能放。
    #     # "75",       # 75和76是调参，学习率、衰减，kl散度。都不行。
    #     # "76",
    #     # "77",     # 77和78修改了奖励权重，为了看能不能覆盖。（奖励权重不一样，对比曲线没有意义。）
    #     # "78",
    #     # "共享网络，局部奖励mappo79",     # 79和80共享网络，局部奖励/全局奖励的MAPPO。
    #     # "共享网络，全局奖励mappo80",
    #     # "84",     # 84和85独立网络，局部奖励/全局奖励的MAPPO。(多算了飞行惩罚，没有对比意义)
    #     # "85",
    #     "86",     # 86和87，重构奖励函数。1处理动作和0处理动作。
    #     "87",
    #     # "独立网络，局部奖励mappo88",     # 88和89独立网络，局部奖励/全局奖励的MAPPO。
    #     # "独立网络，全局奖励mappo89",
    #     # "90",
    # ]

    # 环境调参!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # q1，q3，alpha_r
    # log_dirs = [
    #     # "./scripts/results/mec/mappo/check/run91/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run92/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run93/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run98/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run99/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run94/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run95/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run96/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run97/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run100/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run101/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run102/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run103/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run104/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run105/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run106/logs/agent0/system_performance/agent0/system_performance",
    #
    #     # "./scripts/results/mec/mappo/check/run91/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run92/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run93/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run98/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run99/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run107/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run108/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run109/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run110/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run94/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run95/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run96/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run97/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run100/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run101/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run102/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run103/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run104/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run105/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run106/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run111/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run112/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run113/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #
    #     # "./scripts/results/mec/mappo/check/run91/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run92/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run93/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run98/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run99/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run107/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run108/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run109/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run110/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run94/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run95/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run96/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run97/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run100/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run101/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run102/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run103/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run104/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run105/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run106/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run111/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run112/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run113/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    # ]
    # experiment_names = [
    #     # "91---q1_1---q2_1--q3_1--q4_1--q5_1",
    #     # "92---q1_3---q2_1--q3_1--q4_1--q5_1",
    #     # "93---q1_5---q2_1--q3_1--q4_1--q5_1",
    #     # "98---q1_8---q2_1--q3_1--q4_1--q5_1",
    #     # "99---q1_11--q2_1--q3_1--q4_1--q5_1",
    #     # "107--q1_14--q2_1--q3_1--q4_1--q5_1",
    #     # "108--q1_17--q2_1--q3_1--q4_1--q5_1",
    #     # "109--q1_20--q2_1--q3_1--q4_1--q5_1",
    #     # "110--q1_25--q2_1--q3_1--q4_1--q5_1",
    #
    #     # "94--q1_1--q2_3--q3_1--q4_1--q5_1 ",
    #     # "95--q1_1--q2_5--q3_1--q4_1--q5_1",
    #
    #     # "96--q1_1--q2_1--q3_3--q4_1--q5_1",
    #     # "97--q1_1--q2_1--q3_5--q4_1--q5_1",
    #     # "100--q1_1--q2_1--q3_8--q4_1--q5_1",
    #     # "101--q1_1--q2_1--q3_11--q4_1--q5_1",
    #     # "102--q1_1--q2_1--q3_14--q4_1--q5_1",
    #
    #     # "103--q1_5--q2_1--q3_1--q4_1--q5_1--alpha_r_1.3",
    #     # "104--q1_5--q2_1--q3_1--q4_1--q5_1--alpha_r_1.5",
    #     # "105--q1_5--q2_1--q3_1--q4_1--q5_1--beta_r_0.3",
    #     # "106--q1_5--q2_1--q3_1--q4_1--q5_1--mu_r_3",
    #
    #     # "111-3Neighbor",
    #     # "112-7UAVs-3Neighbor",
    #     # "113-固定",
    # ]

    # #************************* q2，mu_r *************************
    # log_dirs = [
    #     # "./scripts/results/mec/mappo/check/run114/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run115/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run116/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run117/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run118/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run119/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run120/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run121/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run122/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run123/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run124/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run125/logs/agent0/system_performance/agent0/system_performance",
    #
    #     # "./scripts/results/mec/mappo/check/run114/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run115/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run116/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run117/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run118/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run119/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run120/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run121/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run122/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run123/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run124/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run125/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #
    #     # "./scripts/results/mec/mappo/check/run114/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run115/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run116/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run117/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run118/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run119/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run120/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run121/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run122/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run123/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run124/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run125/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    # ]
    # experiment_names = [
    #     # "114--q1_14--q2_1--q3_5--q4_1--q5_1--alpha_r_1.3--mu_r_1",
    #     # "115--q1_14--q2_1--q3_5--q4_1--q5_1--alpha_r_1.3--mu_r_4",
    #     # "116--q1_14--q2_1--q3_5--q4_1--q5_1--alpha_r_1.3--mu_r_8",
    #     # "117--q1_14--q2_1--q3_5--q4_1--q5_1--alpha_r_1.3--mu_r_16",
    #     # "118--q1_14--q2_1--q3_5--q4_1--q5_1--alpha_r_1.3--mu_r_32",
    #     # "119--q1_14--q2_1--q3_5--q4_1--q5_1--alpha_r_1.3--mu_r_64",
    #     # "120--q1_14--q2_1--q3_5--q4_1--q5_1--alpha_r_1.3--mu_r_128",
    #
    #     # "121--q1_14--q2_2--q3_5--q4_1--q5_1--alpha_r_1.3",
    #     # "122--q1_14--q2_4--q3_5--q4_1--q5_1--alpha_r_1.3",
    #     # "123--q1_14--q2_8--q3_5--q4_1--q5_1--alpha_r_1.3",
    #     # "124--q1_14--q2_16--q3_5--q4_1--q5_1--alpha_r_1.3",
    #     # "125--q1_14--q2_32--q3_5--q4_1--q5_1--alpha_r_1.3",
    # ]
    #
    # # ************************* 算法对比 *************************
    # log_dirs = [
    #     # "./scripts/results/mec/mappo/check/run126/logs/system_performance/system_performance",
    #     # "./scripts/results/mec/mappo/check/run127/logs/system_performance/system_performance",
    #     # "./scripts/results/mec/mappo/check/run135/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run129/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run130/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run131/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run132/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run133/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run134/logs/agent0/system_performance/agent0/system_performance",
    #     # "./scripts/results/mec/mappo/check/run136/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     # "./scripts/results/mec/mappo/check/run137/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     # "./scripts/results/mec/mappo/check/run138/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     # "./scripts/results/mec/mappo/check/run139/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     # "./scripts/results/mec/mappo/check/run140/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     # "./scripts/results/mec/mappo/check/run141/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     # "./scripts/results/mec/mappo/check/run144/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     # "./scripts/results/mec/mappo/check/run145/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     # "./scripts/results/mec/mappo/check/run146/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #
    #     # "./scripts/results/mec/mappo/check/run126/logs/n_GUs_by_coverd/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run127/logs/n_GUs_by_coverd/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run135/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run129/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run130/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run131/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run132/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run133/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run134/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run136/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run137/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run138/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run139/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run140/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run141/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run144/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run145/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #     # "./scripts/results/mec/mappo/check/run146/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
    #
    #     # "./scripts/results/mec/mappo/check/run126/logs/cumulative_reward/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run127/logs/cumulative_reward/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run135/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run129/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run130/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run131/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run132/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run133/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run134/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     "./scripts/results/mec/mappo/check/run136/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run137/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run138/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run139/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run140/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run141/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     # "./scripts/results/mec/mappo/check/run144/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     "./scripts/results/mec/mappo/check/run145/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    #     "./scripts/results/mec/mappo/check/run146/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    # ]
    # experiment_names = [
    #     # "126共享网络，局部观测，全局状态、全局奖励的mappo",
    #     # "127共享网络，局部观测，全局状态、局部奖励的mappo",
    #     # "135独立网络，局部观测，全局状态、全局奖励的mappo",
    #     # "129独立网络，局部观测，全局状态、局部奖励的mappo",
    #     # "130独立网络，局部观测，局部状态、局部奖励的ippo",
    #     # "131本文算法",
    #     # "132max_UAVs_obs_concat_3",
    #     # "133max_UAVs_obs_concat_2",
    #     # "134max_UAVs_obs_concat_1",
    #     "136本文算法，新性能记录",
    #     # "137atten网络",
    #     # "138简化了atten网络",
    #     # "139简化后网络max_UAVs_obs_concat_5",
    #     # "140简化后网络max_UAVs_obs_concat_6",
    #     # "141简化后网络max_UAVs_obs_concat_7",
    #     # "144修改了sigmoid",
    #     "145修改optimal",
    #     "146修改process_actions",
    # ]

    # ************************* 利用actions[2]和[3]处理[1]，给mappo添加观测之后的算法对比 *************************
    log_dirs = [
        # "./scripts/results/mec/mappo/check/run19/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run20/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run21/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run22/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run23/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run24/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run25/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run32/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run33/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run38/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        #
        # "./scripts/results/mec/mappo/check/run55/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run56/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run57/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run58/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run63/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run64/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run65/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run66/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run59/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run60/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run61/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run62/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run45/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run39/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run40/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run41/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # 修改无人机数目
        # "./scripts/results/mec/mappo/check/run50/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run51/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run52/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run53/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run54/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run55/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run46/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run47/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run49/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run73/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run74/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run84/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run71/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run85/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run78/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run79/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run86/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run80/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run81/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run82/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run72/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run83/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run75/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run76/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run55/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run77/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # # MAPPO 带宽
        # "./scripts/results/mec/mappo/check/run87/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run88/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run89/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run95/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run25/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run90/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run96/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # F-PPO 带宽
        "./scripts/results/mec/mappo/check/run91/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run93/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run92/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run38/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        "./scripts/results/mec/mappo/check/run94/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # DNC-PPO 计算资源
        # "./scripts/results/mec/mappo/check/run98/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run99/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run100/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run55/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run101/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run19/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run22/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run23/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run24/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run25/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run45/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run39/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run40/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run41/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run50/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run51/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run52/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run53/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run54/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run55/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run46/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run47/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run49/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run67/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run68/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run69/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run73/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run109/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run74/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run84/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run71/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run85/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run78/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # # "./scripts/results/mec/mappo/check/run108/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run79/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run86/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run80/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run81/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run82/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run72/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run83/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run75/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run76/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run55/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",
        # "./scripts/results/mec/mappo/check/run77/logs/agent0/delay_true_all_GUs/agent0/delay_true_all_GUs",

        # "./scripts/results/mec/mappo/check/run69/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run73/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # # "./scripts/results/mec/mappo/check/run109/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run74/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run84/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run71/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run85/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run78/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # # "./scripts/results/mec/mappo/check/run108/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run79/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run86/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run97/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run80/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run81/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run82/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",
        # "./scripts/results/mec/mappo/check/run72/logs/agent0/energy_all_GUs_UAVs/agent0/energy_all_GUs_UAVs",

        # "./scripts/results/mec/mappo/check/run19/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run20/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run21/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run22/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run23/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run24/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run25/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run73/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run74/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run71/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",
        # "./scripts/results/mec/mappo/check/run72/logs/agent0/n_GUs_by_coverd/agent0/n_GUs_by_coverd",

        # "./scripts/results/mec/mappo/check/run19/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run20/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run21/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run22/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run23/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run24/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run25/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run32/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run33/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run34/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run35/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run36/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run37/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run38/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run55/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run56/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run57/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run58/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run59/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run60/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run61/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run62/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run70/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run74/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run71/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run72/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run75/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run76/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run77/logs/agent0/cumulative_reward/agent0/cumulative_reward",

        # "./scripts/results/mec/mappo/check/run26/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run27/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run28/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run29/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run30/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run31/logs/agent0/cumulative_reward/agent0/cumulative_reward",

        # "./scripts/results/mec/mappo/check/run45/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run39/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run40/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run41/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run50/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run51/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run52/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run53/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run55/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run56/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run57/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # # "./scripts/results/mec/mappo/check/run58/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run54/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run46/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run47/logs/agent0/cumulative_reward/agent0/cumulative_reward",
        # "./scripts/results/mec/mappo/check/run49/logs/agent0/cumulative_reward/agent0/cumulative_reward",
    ]
    experiment_names = [
        # "19proposed",
        # # "20nearest_asso_with_sp_avail",
        # # "21fix_uav_pos",
        # # "22not_avg",
        # # "23federated_ppo",
        # # "24nature_ppo",
        # "25mappo",
        # "32nearest_asso",
        # "33ave_resource",
        # # "34federated_interval_10",
        # # "35federated_interval_8",
        # # "36federated_interval_4",
        # # "37federated_interval_2",
        # # "38federated_interval_6",
        # "38federated_ppo",
        # # 随机种子为2
        # # "55proposed",
        # "56ave_resource",
        # "57mappo",
        # # "58federated_ppo",
        # # 随机种子为3
        # # "63proposed",
        # "64ave_resource",
        # "65mappo",
        # # "66federated_ppo",
        # # 奖励平均分给了覆盖的无人机
        # "59proposed",
        # "60ave_resource",
        # "61mappo",
        # "62federated_ppo",

        # # 修改能耗权重
        # # "67 0.00001",
        # # "68 0.0001",
        # "69 0.001",
        # "73 0.005",
        # "70 0.01",
        # # "109 0.01",   # 06.18重新跑一个0.01
        # "74 0.05",
        # "84 0.05",
        # "71 0.1",
        # "85 0.1",
        # "78 0.15",
        # # "108 0.15", # 06.18重新跑一个0.15
        # "79 0.2",
        # "86 0.2", # 忘记改种子了。和79一样。不过比79训练的完整
        # "97 0.2", # 06.14还没下载
        # "80 0.25",
        # "81 0.3",
        # "82 0.5",
        # "72 1",

        # 修改带宽
        "83 5MHz",
        "75 10MHz",
        "76 15MHz",
        "55 20MHz",
        "77 25MHz",
        # # MAPPO 修改带宽
        # "MAPPO87 5MHz",
        # "88 10MHz",
        # "89 15MHz",
        # "95 15MHz", # 种子2，06.14还没下载
        # "25 20MHz",
        # "90 25MHz",
        # "96 25MHz", # 种子2，06.14还没下载
        # # F-PPO 修改带宽
        "F-PPO91 5MHz",
        "93 10MHz",
        "92 15MHz",
        "38 20MHz",
        "94 25MHz", # 06.14还没下载

        # # 修改计算资源
        # "98 10G",
        # "99 12G",
        # "100 14G",
        # "55 15G",
        # "101 16G",
        # "107 18G",    # 06.14还没下载
        # 这里纯看时延，所以计算资源越大，最后的奖励也越高。 但是如果增加能耗权重呢？\lambda_r 0.1
        # " 0.1lambda_r_102 10G",
        # "103 12G",
        # "104 14G",
        # "105 16G",
        # "106 18G",

        # 四个无人机了。[120, 120], [480, 120], [120, 480],[480, 480]
        # "四个无人机26proposed",
        # "27nearest_asso",
        # "28fix_uav_pos",
        # "29federated_ppo",
        # "30mappo",
        # "31ave_resource",

        # 改变无人机数目。
        # #DNC-PPO
        # "45DNC-PPO_10UAVs",
        # "39DNC-PPO_15UAVs",
        # "40DNC-PPO_20UAVs",
        # "41DNC-PPO_25UAVs",
        # # 减小了n_n_rollout_threads的DNC-PPO
        # "50DNC-PPO_10UAVs",
        # "51DNC-PPO_15UAVs",
        # "52DNC-PPO_20UAVs",
        # "53DNC-PPO_25UAVs",
        # # MAPPO
        # "54MAPPO_10UAVs",
        # "46MAPPO_15UAVs",
        # "47MAPPO_20UAVs",
        # "49MAPPO_25UAVs",
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
