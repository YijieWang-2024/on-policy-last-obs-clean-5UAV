# Actor-message 与速度实验主项目归档（2026-08-13）

本文是 `agent/dcppo-runtime-optimization` 当前主线的最短交接入口。完整环境、算法和输入维度见 `FINAL_ENV_ALGORITHM_PARAMETER_AUDIT_20260812.md`；逐 run 速度状态见 `速度实验.md`；8 月 1–9 日历史尝试见 `RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md`。

## 1. 当前冻结结论

- 主速度选择环境是 Fixed600-200 index0：600m 地图、固定 200m/400m 双热点、5 UAV 固定起点、严格每 slot `1+4` 候选动态 MD、lifetime=12、MD 平均速度 3m/s、无 UAV curriculum。
- 当前算法是 separated runner 下五套独立 actor/critic 的局部奖励 PPO 后端，不是共享策略 MAPPO。联合合同为 R520、Spatial Cartesian flight actor、`task_summary + receiver_gated_sum + absolute_raw_v2` Actor message、R520 ego-query attention critic，以及 `per_agent_noise/noise_scale=3/K=50`。
- 正式 message-capable 速度合同的 Actor 原始观测为 251 维；资源头读删除消息后的 211 维；`actor_message_mode=disabled` 时 Actor 本体也为 211 维。Critic 每 UAV token 为 211 维，R520 内最多五个 token，原始 state 为 1055 维。
- v10/v20/v30 的完整 seed32/42 tail100 均值为 485,026 / 496,693 / 562,608，当前选择 `v_max=30m/s`。v40 尚未完成 60M，不能进入最终排名。
- no-message R520 与 actor-message R520 在共同 40.8832M 的 tail25/tail100 差异仅为 +0.017%/+0.452%。这支持“单 seed 中训练表现近似”，但不支持“Actor message 已被证明无用”。

## 2. Actor-message 对照到底关闭了什么

`actor_message_mode=disabled` 会结构性删除 40 维消息块，使 Actor observation 从 251 维变为 211 维，Spatial flight head 不再创建 message encoder/gate。资源、关联、带宽和计算头在有消息版本中本来就只读 211 维本地输入，因此直接结构变化集中在飞行头。

该消融没有关闭整个系统通信：R520 critic mask、5×211 ego-query attention state、Metropolis 图和优势处理保持不变。它回答的是“飞行 Actor 是否需要显式邻机任务/几何摘要”，而不是“整个算法是否需要 UAV 通信”。

## 3. 提交时训练快照

2026-08-13 01:30 使用同一脚本只读刷新：

| 速度 | seed2 | seed32 | seed42 | 当前处理 |
|---:|---:|---:|---:|---|
| 10 | 21.7856M，运行中 | 59.9808M，完成 | 59.9808M，完成 | v30 的完整比较基线 |
| 20 | 21.7344M，运行中 | 59.9808M，完成 | 59.9808M，完成 | v30 的完整比较基线 |
| 30 | 17.7408M，运行中 | 59.9808M，完成 | 59.9808M，完成 | 当前冻结速度 |
| 40 | 23.1168M，冻结快照 | 22.5536M，冻结快照 | 21.7856M，冻结快照 | 等三 seed 完整 60M |

本次 Git 操作没有停止、重启或修改本地 v10/v20/v30 训练进程。

## 4. 主线证据与可复现入口

- 参数与代码路径审计：`FINAL_ENV_ALGORITHM_PARAMETER_AUDIT_20260812.md`
- 速度矩阵和启动方式：`速度实验.md`
- 速度轻量汇总：`analysis/fixed600_speed_3seed_20260812/`
- no-message 轻量汇总：`analysis/fixed600_no_actor_vs_actor_20260811/`
- Random12 参数与远端源码快照：`RANDOM12_RADIUS_BASELINE_RECORD_20260810.md`、`analysis/random12_radius_baseline_20260810/`
- 本地公共启动器：`onpolicy/scripts/train/run_fixed2hotspot_per_agent_noise.ps1`
- 远端速度启动器：`onpolicy/scripts/train/run_fixed600_200_dcppo_R520_noise3p0_md12_seed32_vmax.sh`
- no-message 启动器：`onpolicy/scripts/train/run_fixed600_200_dcppo_R520_noise3p0_md12_no_actor_message.sh`

## 5. Git 归档边界

纳入 Git：源码、测试、正式启动器、参数 JSON、汇总 JSON/CSV、绘图脚本、最终图和总结文档。

不纳入 Git：`onpolicy/scripts/results/` 下模型和完整运行目录、TensorBoard 原始事件、`training_logs/`、逐帧轨迹以及未被最终总结引用的大批历史分析中间产物。这些仍保留在本地设备，不在本次提交中删除。

不可靠通信/MD-GRU 方案继续位于独立 worktree `D:\wyj\Projects\on-policy-unreliable-dataplane` 的 `feature/unreliable-dataplane` 分支，不与本次 actor-message/速度主线混合提交。

## 6. 下一步最小 gate

1. 等 v40 三 seed 完整 60M，再用相同 tail100 与 matched-step 协议和 v30 比较。
2. 等本地 v10/v20/v30 seed2 完成，把当前双 seed结论升级为三 seed。
3. 对最终 v30/R520 message-capable checkpoint 做同 checkpoint 正常消息/消息屏蔽评估；不要用“分别从头训练”的 no-message 曲线代替反事实。
4. Random12 单独作为泛化协议汇报，不与 Fixed600-200 速度曲线按绝对 True60 混排。
