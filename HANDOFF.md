# 5-UAV 动态 MEC 实验交接

> 状态快照：2026-07-31 22:15（Asia/Shanghai）
>
> 当前分支：`agent/dcppo-runtime-optimization`

## 1. 当前结论

主算法训练方式可以按 **Q2** 敲定，不再继续搜索资源动作训练方式。Q2 的主干是：

- strict 1+5 动态用户，每时隙左下 1 个、右上 5 个候选；
- 固定会话寿命 10，最多 60 个活跃槽位；
- Cartesian 飞行动作；
- Spatial Flight Actor；
- Reset Curriculum；
- completion-priority user sort；
- centralized attention critic；
- local reward；
- continuous association/resource actions；
- seed 2、64 个 rollout threads、100M 环境步。

现有证据支持 Q2 已经学会有效的覆盖部署、连续巡航和最终卸载筛选。细粒度带宽/CPU 比例没有证明优于“在相同已选用户集合内等分”，但这不妨碍固定 Q2 为算法主干；后续应转入算法对比和多训练种子确认。

## 2. 最近一轮正式训练结果

训练曲线指标均为 `system_performance_true_all_GUs`。`last20` 是最后 20 个记录点均值。

| 实验 | 关键变化 | 结束步数 | last20 | 末段接纳率 | 末段平均 active | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E1 | Cartesian + Spatial，无 curriculum | 47.13M | 398,118 | 0.7581 | 42.79 | 中途停止，不作最终排名 |
| E2 | E1 + Reset Curriculum | 99.97M | 469,346 | 0.8911 | 50.85 | curriculum 有明显帮助 |
| B0 | Cartesian 普通 actor + curriculum | 98.53M | 456,287 | 0.8772 | 49.16 | 不及 Q0/Q2 |
| B1 | B0 + actor neighbor observation | 97.46M | 451,979 | 0.8745 | 48.73 | 邻居观测没有带来收益 |
| Q0 | Polar 普通 actor + curriculum + completion priority | 99.97M | 527,045 | 0.8259 | 48.17 | 有竞争力的稳定基线 |
| Q1 | Polar Spatial + curriculum + completion priority | 99.97M | 458,473 | 0.7062 | 41.18 | 94M 后退化 |
| **Q2** | **Cartesian Spatial + curriculum + completion priority** | **99.97M** | **570,817** | **0.9099** | **52.76** | **当前最佳主干** |

Q1 的最后 50 个点为 469,809，最佳 20 点窗口为 491,458；它的较好轨迹评测来自约 94.2M checkpoint，不应与最终 100M 模型混用。

## 3. 统一十集评测

以下是共同的 10 个确定性环境种子；“后半段 active”只统计 slots 201–400，避免初始位置偏置。

| 模型/checkpoint | True60 | Eq60 | 实际覆盖目标 | 接纳率 | 任务完成率 | 后半段 active | 总航程 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q0 final | 552,312 ± 9,129 | 565,707 ± 8,091 | 362,540 ± 5,623 | 0.8160 | 0.98162 | 49.55 | 1,424 m |
| Q1 94.2M | 516,332 ± 4,113 | 532,890 ± 3,575 | 351,806 ± 3,680 | 0.7680 | 0.97479 | 46.87 | 1,046 m |
| **Q2 90.2M** | **612,606 ± 3,039** | **619,205 ± 2,917** | **372,675 ± 1,048** | **0.9193** | **0.97920** | **56.08** | **4,967 m** |

Q2 的 fresh final-100M `seed=100000` 评测进一步得到：True60 624,690、Eq60 630,370、接纳率 0.93083、后半段 active 56.5、完成率 98.13%。五架 UAV 没有碰撞或撞边界，最小机间距 35.2 m；部署结构为左下 1 架、右上 4 架。

这些评测种子不是独立训练种子。论文最终表格仍需至少补齐多个训练 seed，不能把十个环境随机流写成十个训练重复。

## 4. 学到的动作是否合理

### 飞行

Q2 学到的是持续、受控的热点巡航，不是到点悬停。final-100M 单集平均命令速度为 4.55 m/s，95 分位约 9.53 m/s，总航程 4,548 m；没有早期模型常见的左上角吸收、持续撞墙或机间碰撞。由于飞行能耗权重仅 `1e-6`，当前目标对悬停没有强偏好，因此论文中应描述为动态巡航策略，不能暗示 UAV 已学习最小航程。

### 关联与资源

final Q2 单集共有 20,021 个原始候选卸载，最终选择 11,803 个，后处理取消率 41.05%。把所有候选都等分资源会使 True60 从 624,690 降到 550,337，说明卸载筛选确实有效。

在保持相同最终选择集合时，把带宽和 CPU 改为等分，单集反而从 624,690 升至 631,619。10 个环境种子上的结果为：

- learned allocation：615,090.9 ± 4,925.2；
- same selected set + equal allocation：621,893.8 ± 4,825.9；
- 平均提升 6,802.9（1.106%），10/10 seeds 获胜。

因此最准确的结论是：Q2 的覆盖和最终选择动作已学得较好；连续资源比例仍有可见次优，但继续围绕资源头调参的收益上限约 1%，当前阶段不再投入训练预算。

## 5. 当前进行中的算法对比

以下三条训练均使用 Q2 的环境、飞行、curriculum 和用户排序主干，仅改变算法或 DC-PPO 通信距离。它们在本快照时都仍在写入事件文件，中途点只用于确认健康运行。

| 实验目录 | 快照步数 | 最新单点 True-all | 状态 |
| --- | ---: | ---: | --- |
| `q2backbone_dcppo_comm260_seed2_100m_20260731` | 28.544M | 507,555 | 运行中 |
| `q2backbone_dcppo_comm520_seed2_100m_20260731` | 28.186M | 465,279 | 运行中 |
| `q2backbone_ippo_independent_seed2_100m_r64_20260731` | 30.336M | 546,977 | 运行中 |

早先没有 `_r64_` 的 IPPO 正式目录只产生了空事件头，不能作为实验；`smoke_q2_ippo_*` 也只验证接口和显存配置。

## 6. 代码状态

本轮代码已补齐：

- `ippo` 训练入口；
- DC-PPO 可配置通信距离；
- completion-priority 排序开关，并与 distance-only 排序互斥；
- Spatial Flight Actor 支持 Cartesian 和 Polar 两种飞行动作；
- 对应回归测试。

大型 checkpoint、TensorBoard 日志、逐帧评测和正在写入的启动日志不进入 Git。轻量 CSV、JSON、绘图脚本和论文诊断图进入版本库。

## 7. 接下来只做什么

1. 等待 DC-PPO-260、DC-PPO-520 和 IPPO 跑满共同预算，不依据当前单点提前排名。
2. 对三条最终模型和 Q2 使用同一组 10 个环境种子，统一报告 True60、Eq60、实际覆盖目标、接纳率、任务完成率、slots 201–400 平均 active、轨迹长度和碰撞/边界事件。
3. 为 Q2 和有竞争力的对比算法补多个训练 seed，再形成论文均值与方差。
4. 后续开展论文所需算法对比与消融；不再重新打开资源训练方式搜索，除非审稿证据明确要求。
5. 旧 Fig.4 叠图只用于量级观察，因为旧图是固定 60 用户且包含历史后处理，不能作为公平对比结论。

## 8. 关键证据位置

- 完整阶段总结：`RECENT_EXPERIMENT_SUMMARY_20260725_20260728.md`
- Q0/Q1/Q2 实际覆盖曲线：`paper_artifacts/diagnostics/true_all_competitive_q012_20260730_220752/`
- 覆盖随时间变化：`paper_artifacts/diagnostics/q0_q1_q2_coverage_time_20260730/`
- Q2 与旧 Fig.4 叠图：`paper_artifacts/diagnostics/fig4_q2_true_all_overlay_20260731/`
- 原始动作审计：`paper_artifacts/diagnostics/final_policy_audit_20260731/raw_action_audit.json`
- 十集资源反事实：`paper_artifacts/diagnostics/final_policy_audit_20260731/q2_multiseed_resource_check.json`
