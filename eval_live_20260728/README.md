# 2026-07-28 固定起点阶段评估

这里保存 E2、B0、B1 的阶段性 `seed=100000`、确定性、固定起点单 episode 评估。它们用于检查部署形态和接口是否工作，不构成多 seed 统计结论。

每个目录保留：

- `episode_summary.json`：checkpoint 步数、参数哈希、最终指标和 UAV 位置；
- `episode_data.npz`、两个 CSV：可复核的逐时隙数据；
- `uav_trajectory_overview.png`：轨迹总览；
- `args.json`：训练参数。

为控制 Git 体积，80 张逐帧 PNG 和重复的 actor/critic/normer 文件未归档；对应文件哈希仍记录在 `episode_summary.json`，原训练 checkpoint 位于被 `.gitignore` 排除的本地 `onpolicy/scripts/results/`。
