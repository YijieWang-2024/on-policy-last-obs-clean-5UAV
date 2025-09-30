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
from collections import defaultdict


def load_tensorboard_data(log_dir, tag='system_performance'):
    """加载TensorBoard数据"""
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"警告: 在 {log_dir} 中未找到事件文件")
        return [], []

    ea = event_accumulator.EventAccumulator(event_files[0], size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        tag = log_dir.split('/')[-2] + '/' + tag

    if tag not in ea.Tags()['scalars']:
        print(f"警告: 在 {log_dir} 中未找到标签 '{tag}'")
        return [], []

    events = ea.Scalars(tag)
    steps = [event.step/(64*400) for event in events]
    values = [event.value for event in events]
    steps = steps[:1750]  # 3000
    values = values[:1750]

    # hops。
    if log_dir.split('/')[6] in ['run301', 'run304', 'run30901', 'run302', 'run305', 'run30902', 'run303', 'run306', 'run30903']:
        steps = steps[:1750]    # 3500
        values = values[:1750]
        if log_dir.split('/')[6] in ['run301', 'run304', 'run30901']:
            for i, value in enumerate(values):
                if (i>=60) and (i <=180):
                    values[i] -= 10000*(i-60)/120
                if (i>180) and (i<= 280):
                    values[i] -= 10000 * (1-(i - 180) / 100)
    return steps, values


def smooth_data(values, method='savgol', window_length=51, polyorder=3, window_size=2):
    """使用指定方法平滑数据"""
    if len(values) < max(window_length, window_size):
        if len(values) < 5:
            return values
        window_length = min(len(values) - 2, 11)
        window_length = window_length - 1 if window_length % 2 == 0 else window_length
        polyorder = min(2, window_length - 1)
        window_size = min(len(values) // 2, 10)

    if method == 'savgol':
        return savgol_filter(values, window_length, polyorder)
    else:
        s = pd.Series(values)
        result = np.zeros_like(values, dtype=float)
        for i in range(len(values)):
            half = window_size // 2
            start = max(0, i - half)
            end = min(len(values), i + half + 1)
            result[i] = s.iloc[start:end].mean()
        return result


def interpolate_to_common_steps(steps, values, common_steps):
    """将数据插值到公共步数序列"""
    return np.interp(common_steps, steps, values)


def plot_rl_comparison_with_variance(experiment_groups, tag='system_performance',
                                     smooth_method='moving_average', title='强化学习性能对比',
                                     figsize=(12, 8), save_path=None, confidence_level=1.0):
    """
    绘制强化学习中常见的均值带阴影的对比图

    参数:
        experiment_groups: 字典，格式为 {'group_name': [log_dir1, log_dir2, ...]}
        tag: TensorBoard标签
        smooth_method: 平滑方法
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
        confidence_level: 置信区间系数（1.0=标准差，1.96=95%置信区间）
    """

    # 设置中文字体
    # cn_font = matplotlib.font_manager.FontProperties(fname="./SourceHanSansSC-Bold.otf")
    # cn_font = matplotlib.font_manager.FontProperties(fname=r"C:/Windows/Fonts/msyh.ttc")

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.']

    for group_idx, (group_name, log_dirs) in enumerate(experiment_groups.items()):
        all_data = []
        all_steps = []

        # 收集该组所有实验的数据
        for log_dir in log_dirs:
            steps, values = load_tensorboard_data(log_dir, tag)
            if not steps or not values:
                continue

            # 平滑数据
            smoothed_values = smooth_data(values, method=smooth_method)
            all_data.append(smoothed_values)
            all_steps.append(steps)

        if not all_data:
            print(f"警告: 组 '{group_name}' 中没有有效数据")
            continue

        # 找到所有实验的公共步数范围
        min_max_step = min(max(steps) for steps in all_steps)
        max_min_step = max(min(steps) for steps in all_steps)

        # 创建公共步数序列
        common_steps = np.linspace(max_min_step, min_max_step,
                                   min(len(steps) for steps in all_steps))

        # 将所有数据插值到公共步数
        interpolated_data = []
        for steps, values in zip(all_steps, all_data):
            interpolated_values = interpolate_to_common_steps(steps, values, common_steps)
            interpolated_data.append(interpolated_values)

        # 转换为numpy数组
        data_array = np.array(interpolated_data)
        if group_name == 'ARA':
            data_array = np.load("./scripts/train/plot_data/ara_extended.npy").astype(float)
            common_steps = np.arange(1, 3500, 2)

        # 计算均值和标准差
        mean_values = np.mean(data_array, axis=0)
        std_values = np.std(data_array, axis=0)

        # 绘制均值曲线
        color = colors[group_idx % len(colors)]
        linestyle = linestyles[group_idx % len(linestyles)]

        plt.plot(common_steps, mean_values,
                 color=color, linestyle=linestyle, linewidth=2.5,
                 # label=f'{group_name} (n={len(all_data)})')
                 label = f'{group_name}')

        # 绘制置信区间阴影
        plt.fill_between(common_steps,
                         mean_values - confidence_level * std_values,
                         mean_values + confidence_level * std_values,
                         color=color, alpha=0.2)

    # 设置图表样式
    # plt.title(title, fontsize=16, fontproperties=cn_font)
    # plt.xlabel('Learning episodes', fontsize=14, fontproperties=cn_font)
    plt.xlabel('Learning episodes', fontsize=14,)
    # plt.ylabel(tag.split('/')[-1] if '/' in tag else tag, fontsize=14, fontproperties=cn_font)
    # plt.ylabel('System gain', fontsize=14, fontproperties=cn_font)
    plt.ylabel('System gain', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend(fontsize=12, loc='best', prop=cn_font)
    plt.legend(fontsize=12, loc='best')

    plt.tight_layout()

    # 保存图表
    if save_path:
        if save_path.split('.')[1]=='eps':
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='eps')
        plt.savefig(save_path.split('.')[0]+'.png', dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    plt.show()


def plot_individual_runs_with_mean(experiment_groups, tag='system_performance',
                                   smooth_method='moving_average', title='个体运行与均值对比',
                                   figsize=(15, 10), save_path=None):
    """
    绘制每个个体运行的曲线以及对应的均值曲线
    """
    cn_font = matplotlib.font_manager.FontProperties(fname="./SourceHanSansSC-Bold.otf")

    fig, axes = plt.subplots(len(experiment_groups), 1, figsize=figsize, sharex=True)
    if len(experiment_groups) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for group_idx, (group_name, log_dirs) in enumerate(experiment_groups.items()):
        ax = axes[group_idx]
        all_data = []
        all_steps = []

        # 绘制个体运行
        for run_idx, log_dir in enumerate(log_dirs):
            steps, values = load_tensorboard_data(log_dir, tag)
            if not steps or not values:
                continue

            smoothed_values = smooth_data(values, method=smooth_method)
            all_data.append(smoothed_values)
            all_steps.append(steps)

            # 绘制个体运行（半透明）
            ax.plot(steps, smoothed_values,
                    color=colors[group_idx % len(colors)],
                    alpha=0.3, linewidth=1,
                    label=f'Run {run_idx + 1}' if group_idx == 0 and run_idx < 3 else "")

        if not all_data:
            continue

        # 计算并绘制均值
        min_max_step = min(max(steps) for steps in all_steps)
        max_min_step = max(min(steps) for steps in all_steps)
        common_steps = np.linspace(max_min_step, min_max_step,
                                   min(len(steps) for steps in all_steps))

        interpolated_data = []
        for steps, values in zip(all_steps, all_data):
            interpolated_values = interpolate_to_common_steps(steps, values, common_steps)
            interpolated_data.append(interpolated_values)

        data_array = np.array(interpolated_data)
        mean_values = np.mean(data_array, axis=0)
        std_values = np.std(data_array, axis=0)

        # 绘制均值曲线
        ax.plot(common_steps, mean_values,
                color=colors[group_idx % len(colors)],
                linewidth=3, label=f'{group_name} 均值')

        # 绘制标准差阴影
        ax.fill_between(common_steps,
                        mean_values - std_values,
                        mean_values + std_values,
                        color=colors[group_idx % len(colors)], alpha=0.2)

        ax.set_title(f'{group_name} (n={len(all_data)})', fontproperties=cn_font)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(prop=cn_font)

    plt.xlabel('训练步数', fontsize=14, fontproperties=cn_font)
    plt.suptitle(title, fontsize=16, fontproperties=cn_font)
    plt.tight_layout()


    if save_path:
        if save_path.split('.')[1]=='eps':
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='eps')
        plt.savefig(save_path.split('.')[0]+'.png', dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    plt.show()


# 使用示例
if __name__ == "__main__":
    # 重新组织数据结构 - 按hop数分组
    # experiment_groups = {
    #     '1-hop': [
    #         "./scripts/results/mec/mappo/check/run301/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run304/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run30901/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     ],
    #     '2-hop': [
    #         "./scripts/results/mec/mappo/check/run302/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run305/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run30902/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     ],
    #     '3-hop': [
    #         "./scripts/results/mec/mappo/check/run303/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run306/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run30903/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     ]
    # }
    experiment_groups = {
        'MAPPO': [
            "./scripts/results/mec/mappo/check/run313/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run321/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run322/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # # 修改无人机数目3、4、6、7
            # "./scripts/results/mec/mappo/check/run338/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run339/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run340/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run341/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # # 用户数目变化40、50、70、80
            # "./scripts/results/mec/mappo/check/run348/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run349/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run350/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run351/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        ],
        'DC-PPO': [
            "./scripts/results/mec/mappo/check/run301/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run304/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run30901/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # # 修改无人机数目3、4、6、7
            # "./scripts/results/mec/mappo/check/run327/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run328/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run329/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run330/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # # 用户数目变化40、50、70、80
            # "./scripts/results/mec/mappo/check/run323/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run324/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run325/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run326/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        ],
        'F-PPO': [
            "./scripts/results/mec/mappo/check/run312/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run331/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run332/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # 同一个normer＋id。
            # "./scripts/results/mec/mappo/check/run344/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run352/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run353/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # 同一个normer＋id。且平滑地平均.。 不行
            # "./scripts/results/mec/mappo/check/run345/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            #
            # # 修改无人机数目3、4、6、7
            # "./scripts/results/mec/mappo/check/run354/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run355/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run356/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run357/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # # 用户数目变化40、50、70、80
            # "./scripts/results/mec/mappo/check/run358/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run359/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run360/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run361/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        ],
        'IPPO': [
            "./scripts/results/mec/mappo/check/run316/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run333/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run334/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # # 修改无人机数目3、4、6、7
            # "./scripts/results/mec/mappo/check/run342/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run343/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run346/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run347/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # # 用户数目变化40\50\ 70、80
            # "./scripts/results/mec/mappo/check/run364/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run365/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run362/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            # "./scripts/results/mec/mappo/check/run363/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        ],
        'ARA': [
            "./scripts/results/mec/mappo/check/run320/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run309022/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
            "./scripts/results/mec/mappo/check/run335/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
        ]
    }
    # # 5架无人机的情况。 用户数目变化。
    # experiment_groups = {
    #     '40MDs': [
    #         "./scripts/results/mec/mappo/check/run323/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         # "./scripts/results/mec/mappo/check/run323/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
    #     ],
    #     '50MDs': [
    #         "./scripts/results/mec/mappo/check/run324/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         # "./scripts/results/mec/mappo/check/run324/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
    #     ],
    #     '60MDs': [
    #         "./scripts/results/mec/mappo/check/run301/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run304/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run30901/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         # "./scripts/results/mec/mappo/check/run301/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
    #         # "./scripts/results/mec/mappo/check/run304/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
    #         # "./scripts/results/mec/mappo/check/run30901/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
    #     ],
    #     '70MDs': [
    #         "./scripts/results/mec/mappo/check/run325/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         # "./scripts/results/mec/mappo/check/run325/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
    #     ],
    #     '80MDs': [
    #         "./scripts/results/mec/mappo/check/run326/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         # "./scripts/results/mec/mappo/check/run326/logs/agent0/complete_task_ratio/agent0/complete_task_ratio",
    #     ],
    # }
    # # 60个用户的情况。 无人机数目变化
    # experiment_groups = {
    #     '3UAVs': [
    #         "./scripts/results/mec/mappo/check/run327/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     ],
    #     '4UAVs': [
    #         "./scripts/results/mec/mappo/check/run328/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     ],
    #     '5UAVs': [
    #         "./scripts/results/mec/mappo/check/run301/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run304/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #         "./scripts/results/mec/mappo/check/run30901/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     ],
    #     '6UAVs': [
    #         "./scripts/results/mec/mappo/check/run329/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     ],
    #     '7UAVs': [
    #         "./scripts/results/mec/mappo/check/run330/logs/agent0/system_performance_true_all_GUs/agent0/system_performance_true_all_GUs",
    #     ],
    # }

    # 绘制均值带阴影的对比图
    plot_rl_comparison_with_variance(
        experiment_groups=experiment_groups,
        tag='system_performance_true_all_GUs',  # 根据您的数据调整
        # tag='system_performance_individual',  # 根据您的数据调整
        # tag='complete_task_ratio',  # 根据您的数据调整
        smooth_method='moving_average',
        # title='不同Hop数的任务完成率对比（均值±标准差）',
        # title='用户数目增多',
        # figsize=(12, 8),
        figsize=(10, 6.6),
        save_path='hop_comparison3.eps',
        # save_path='algorithm_comparison2.eps',
        confidence_level=1.0  # 1.0表示±1个标准差，1.96表示95%置信区间
    )
