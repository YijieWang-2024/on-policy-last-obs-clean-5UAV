# 最近两周实验工作日志

> 覆盖时间：2026-07-22 至 2026-07-31。本文只记录可复用的工程事实、正式实验和决策；启动失败、smoke 与重复导出不会伪装成独立实验。

## 2026-07-22 至 2026-07-23：工作区归档与训练提速

- 整理 5-UAV 论文复现工作区，明确旧 Fig.3–Fig.8 的脚本、处理数据和完整性风险。
- 建立动态 strict 1+5 用户实验的文档与测试基线。
- 优化 DC-PPO 训练热路径，减少重复张量构造和不必要的数据搬运，为后续 50M/100M 实验提供可接受吞吐。
- 固定当前工作分支为 `agent/dcppo-runtime-optimization`。

## 2026-07-24 至 2026-07-26：环境时序与早期失败诊断

- 逐段核对动态 MD 的出生、准入、持续运动、寿命到期、覆盖离开和槽位复用。
- 确认 strict 1+5、lifetime=10 下每个 episode 有 400 批、2,400 个候选；60 是活跃会话槽位上限，不是始终存在 60 个真实 MD。
- 确认任务/奖励在时隙开始位置计算，UAV 和 MD 在时隙末移动；出生未覆盖与移动后离开联合覆盖的用户不会在后台继续运动后返回。
- 诊断 DC-PPO 左上角吸收：无信息动作先验接近向西中速、边界 clip、覆盖外需求不可见、无离开/边界即时惩罚和局部/均值优势信用混合共同造成。
- 完成 GAE、初始位置、50M 接续 100M 等早期比较；确认 50M 容易过早判断，且 Eq60 与实际覆盖目标必须分开报告。

## 2026-07-27 至 2026-07-28：E1/E2/B0/B1 结构消融

- 实现 Cartesian 飞行动作、Spatial Flight Actor、Reset Curriculum 和 actor 邻居观测消融。
- 启动 E1/E2/B0/B1 正式运行；处理并发显存、页面文件、BrokenPipe 和 32/64 rollout threads 的启动经验。
- E2 相比 E1 证明 Reset Curriculum 是有效改动；B1 没有证明额外 actor 邻居观测有效。
- 发现 Spatial Flight Actor 曾隐式改变资源用户排序，导致归因混杂；随后把 completion-priority 排序独立成显式配置。

## 2026-07-29：Q0/Q1/Q2 原子对照

- Q0：Polar 普通 actor + curriculum + completion-priority。
- Q1：Polar Spatial Actor + curriculum + completion-priority。
- Q2：Cartesian Spatial Actor + curriculum + completion-priority。
- 三组使用相同 seed、线程数、学习率、critic、奖励与资源动作合同，消除早期排序混杂。
- 训练后期 Q2 明显领先；Q1 在约 94M 后退化，因此保留最佳中期 checkpoint 与 final checkpoint 的区别。

## 2026-07-30：统一曲线、轨迹与覆盖评测

- 绘制 E2/B0/B1 和 Q0/Q1/Q2 的 `system_performance_true_all_GUs` 曲线。
- 用共同的 10 个确定性环境种子评估 Q0/Q1/Q2，并按 episode 指标排名轨迹。
- 按用户要求将 active 统计改为 slots 201–400 的后半段均值。
- Q2 在十集评测中达到 True60 612,606 ± 3,039、Eq60 619,205 ± 2,917、后半段 active 56.08、接纳率 0.9193，形成左下 1 架、右上 4 架的稳定部署。
- 轨迹审计显示 Q0 更接近定点驻留，Q2 是覆盖更广的持续巡航；Q2 不再出现早期模型的左上角卡死。

## 2026-07-31：最终动作审计、主干冻结与对比实验

- 对 final-100M Q0/Q1/Q2 做 fresh deterministic evaluation；Q2 单集 True60 624,690、Eq60 630,370、后半段 active 56.5。
- 审计 Q2 原始动作：20,021 个候选卸载中最终选中 11,803 个，说明后处理/选择门控承担了重要作用。
- 完成资源反事实：保持相同选择集合并等分资源，10 个种子平均比 learned allocation 高 1.106%；结论是筛选已学好、细粒度资源比例仍略次优。
- 根据收益上限和研究优先级，决定冻结 Q2 算法主干，不再继续尝试资源训练方式，后续转入对比实验。
- 把 Q2 的 True-all 曲线与旧论文 Fig.4 叠图；Q2 在相近横坐标上的量级与历史 MAPPO 接近，但由于动态/固定用户和统计协议不同，仅作诊断。
- 新增 IPPO 入口、DC-PPO 通信距离参数、Polar Spatial Actor 合同及排序互斥测试。
- 启动 DC-PPO 260 m、DC-PPO 520 m 和独立 IPPO 三条 100M 对比训练；截至 22:15 分别运行到 28.544M、28.186M 和 30.336M。

## 当前决策记录

- 主干：Q2。
- 主要训练指标：`system_performance_true_all_GUs`；Eq60 作为补充数量对齐指标。
- active 统计：episode 后半段 slots 201–400。
- 评测：共同 deterministic seeds；环境 seed 与训练 seed 必须分开表述。
- 资源策略：记录当前约 1% 的等分反事实差距，但不再进行资源头搜索。
- 下一阶段：完成 DC-PPO 通信距离和 IPPO 对比，再补多训练种子与论文消融。
