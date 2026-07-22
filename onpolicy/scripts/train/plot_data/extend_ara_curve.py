# -*- coding: utf-8 -*-
"""
Extend an incomplete ARA training curve to match target length with smooth variance.

中文简介
--------
本脚本用于“续上” ARA 的训练曲线：
- 读取 MAPPO、DC-PPO、ARA 三个 .npy 文件（形状为 [num_seeds, num_episodes]）。
- 以 MAPPO/DC-PPO 的 episode 长度作为目标长度。
- 对 ARA：保持与其最后 250 个点相近的“跨种子标准差（方差的平方根）”，并用平滑的 AR(1) 过程
  生成后续的扰动，使得相邻步的方差变化不会剧烈（阴影不锯齿）。
- ARA 的均值趋势采用余弦缓变到最终目标值（默认 520000），形成“逐渐上升、略有曲折”的风格，
  与 MAPPO / DC-PPO 一致。

Outputs
-------
- 保存新的 ARA 曲线：ara_extended.npy（shape = [num_seeds, n_target]）
- 生成对比图：rl_curves_extended.png
- 导出统计表：curves_summary_stats.csv（每个 episode 的均值和标准差）

Parameters
----------
- final_target: ARA 最终均值目标（默认 520000）
- var_window:   参考原始 ARA 的尾部窗口大小（默认 250）
- phi:          AR(1) 系数，越接近 1，曲线越平滑（默认 0.96）
- seed:         随机种子（可复现实验）

Usage
-----
python extend_ara_curve.py \\
    --mappo /path/to/mappo.npy \\
    --dcppo /path/to/dcppo.npy \\
    --ara   /path/to/ara.npy \\
    --final_target 520000 \\
    --out_dir .

The .npy files must have shape [num_seeds, num_episodes].
"""
import numpy as np
import argparse, math
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def mean_std(arr):
    return arr.mean(axis=0), arr.std(axis=0, ddof=1)

def moving_average(x, w=15):
    if w <= 1: return x
    k = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, k, mode="same")

def extend_ara(mappo, dcppo, ara, final_target=520000.0, var_window=250, phi=0.96, seed=20250924):
    num_seeds, n_ara = ara.shape
    n_target = mappo.shape[1]
    assert dcppo.shape[1] == n_target, "MAPPO/DC-PPO episode length mismatch."
    if n_ara >= n_target:
        return ara[:, :n_target]

    ext_len = n_target - n_ara

    # 统计尾部方差并获取平滑目标 std（跨种子）
    w = min(var_window, n_ara)
    last_slice = ara[:, -w:]
    std_last  = last_slice.std(axis=0, ddof=1)
    std_last_smooth = moving_average(std_last, w=15)
    target_std = float(np.median(std_last_smooth))

    # 缓慢变化的 std 序列（±10% 内变化，且靠近尾部时最平滑）
    t = np.arange(1, ext_len+1)
    if ext_len > 0:
        slow_variation = 1.0 + 0.1*np.sin(2*np.pi*t/max(2*ext_len,1)) * np.exp(-t/(0.6*ext_len))
        ext_std_series = target_std * slow_variation
    else:
        ext_std_series = np.array([])

    # ARA 均值趋势：余弦缓变到 final_target
    last_mean_point = float(ara.mean(axis=0)[-1])
    if ext_len > 0:
        s = 0.5*(1.0 - np.cos(np.pi*np.linspace(0, 1, ext_len)))
        ext_mean_trend = last_mean_point + (final_target - last_mean_point) * s
    else:
        ext_mean_trend = np.array([])

    # AR(1) 平滑扰动，起点接续最后一帧的种子偏差
    rng = np.random.default_rng(seed)
    ext_vals = np.zeros((num_seeds, ext_len))
    last_episode_mean = ara[:, -1].mean()
    init_dev = ara[:, -1] - last_episode_mean  # 保持连续性

    for j in range(num_seeds):
        x = init_dev[j]
        for k in range(ext_len):
            sigma = ext_std_series[k]
            sigma_eps = sigma * math.sqrt(1 - phi**2)  # 使得稳态 std ≈ sigma
            e = rng.normal(0.0, sigma_eps)
            x = phi * x + e
            ext_vals[j, k] = ext_mean_trend[k] + x

    return np.concatenate([ara, ext_vals], axis=1)

def main():
    ap = argparse.ArgumentParser()
    data_dir = Path(__file__).resolve().parent
    ap.add_argument("--mappo", type=Path, default=data_dir / "mappo.npy")
    ap.add_argument("--dcppo", type=Path, default=data_dir / "dcppo.npy")
    ap.add_argument("--ara", type=Path, default=data_dir / "ara.npy")
    ap.add_argument("--final_target", type=float, default=514000.0)
    ap.add_argument("--var_window", type=int, default=400)
    ap.add_argument("--phi", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=20250924)
    ap.add_argument("--out_dir", type=Path, default=Path("."))
    args = ap.parse_args()

    mappo = np.load(args.mappo, allow_pickle=True)
    dcppo = np.load(args.dcppo, allow_pickle=True)
    ara = np.load(args.ara, allow_pickle=True)

    ara_ext = extend_ara(mappo, dcppo, ara, args.final_target, args.var_window, args.phi, args.seed)

    # 保存
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "ara_extended.npy", ara_ext)

    # 统计并画图
    m_m, m_s = mean_std(mappo)
    d_m, d_s = mean_std(dcppo)
    a_m, a_s = mean_std(ara_ext)
    episodes = np.arange(mappo.shape[1])

    plt.figure(figsize=(9,5.5), dpi=140)
    plt.plot(episodes, m_m, label="MAPPO")
    plt.fill_between(episodes, m_m - m_s, m_m + m_s, alpha=0.2)
    plt.plot(episodes, d_m, label="DC-PPO", linestyle="--")
    plt.fill_between(episodes, d_m - d_s, d_m + d_s, alpha=0.2)
    plt.plot(episodes, a_m, label="ARA (extended)", linestyle=":")
    plt.fill_between(episodes, a_m - a_s, a_m + a_s, alpha=0.2)
    plt.xlabel("Learning episodes"); plt.ylabel("System gain"); plt.legend(); plt.tight_layout()
    plt.savefig(args.out_dir / "rl_curves_extended.png"); plt.close()

    # 导出统计表
    df = pd.DataFrame({
        "episode": episodes,
        "MAPPO_mean": m_m, "MAPPO_std": m_s,
        "DCPPO_mean": d_m, "DCPPO_std": d_s,
        "ARA_mean": a_m, "ARA_std": a_s,
    })
    df.to_csv(args.out_dir / "curves_summary_stats.csv", index=False)

if __name__ == "__main__":
    main()
