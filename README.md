# on-policy-last-obs-clean-5UAV

> 最新入口（2026-07-31）：[当前交接状态](HANDOFF.md)、[最近两周工作日志](WORK_NOTES.md)、[Q0/Q1/Q2 最终补充总结](RECENT_EXPERIMENT_SUMMARY_20260729_20260731.md)。

5-UAV 动态移动边缘计算实验工作区。当前分支集中记录 strict 1+5 动态用户场景、MAPPO/DC-PPO、Cartesian 飞行动作、Spatial Flight Actor、Reset Curriculum 及其消融实验。

## 从这里开始

- [近期实验进展](RECENT_EXPERIMENT_SUMMARY_20260725_20260728.md)：论文相关结论、数值证据、负结果、失败运行、进行中实验与下一步门控。
- [项目来源与论文图复现](PROJECT_NOTES.md)：仓库基线、旧论文图数据来源及完整性限制。
- [5-UAV 代码走读](CODE_WALKTHROUGH_5UAV.md)：环境、动作、critic、advantage 聚合及论文/代码一致性审计。
- `tests/`：动态用户、Cartesian 飞行、Spatial Flight Actor 的最小回归测试。
- `onpolicy/scripts/train/run_dynamic_5uav.ps1`：当前 Windows 训练入口。
- [Type-S 不可靠数据面与 MD 状态重建契约](UNRELIABLE_DATAPLANE.md)：packet、receiver memory、zero/last_obs/MD-GRU 闭环及启动命令。

## Git 中包含与不包含的内容

Git 保存代码、测试、实验命令、精简评估结果和论文诊断图。大型 checkpoint、TensorBoard 日志和 `onpolicy/scripts/results/` 不进入 Git；另一台电脑 clone 后可以查看当前实验结论和复现实验配置，但不能直接取得本机训练权重。

`eval_live_20260728/` 只保留三个阶段性固定起点评估的 JSON、CSV、NPZ、总览图和参数快照。逐帧 PNG 与重复模型副本已删除，以避免用 Git 传输约 133 MB 的可再生产物。
