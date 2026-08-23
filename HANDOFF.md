# 5-UAV 动态 MEC 实验交接

> 主线归档更新：2026-08-13（Asia/Shanghai）
>
> 当前分支：`agent/dcppo-runtime-optimization`
>
> **当前阅读顺序：** `ACTOR_MESSAGE_SPEED_ARCHIVE_20260813.md` → `FINAL_ENV_ALGORITHM_PARAMETER_AUDIT_20260812.md` → `速度实验.md` → 本文件的时间线。

> **历史边界：** 第 1--12 节以及 2026-08-06 authoritative state 都是当时快照；若与 2026-08-12/13 的归档段落冲突，以当前阅读顺序中的三份新文档和本文件的 2026-08-12/13 段落为准。

## 1. 历史结论（2026-07-31 至 2026-08-01 Q2 主干）

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
| **Q2 90.2M（历史快照）** | **612,606 ± 3,039** | **619,205 ± 2,917** | **372,675 ± 1,048** | **0.9193** | **0.97920** | **56.08** | **4,967 m** |

截至 2026-08-06，10 个固定评估 episode 的当前汇总以周总结中的原始 evaluation/episode_metrics.csv 为准：True60 均值 617,738，样本标准差 4,096，接纳率 0.9256，任务完成率 0.9791。上面的 90.2M 行和 fresh final-100M 单集行保留作历史/单集记录，不再作为当前十集均值。

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

---

## 9. 2026-08-01 至 2026-08-03 续接交接：DC-PPO 通信半径主线

### 9.1 当前研究问题与统一协议

Q2 环境与 Actor 主干仍然保留。本阶段不再搜索资源动作，而是检验在严格匹配协议下能否得到：

```text
DC-PPO-R0 < DC-PPO-R260 < DC-PPO-R1000
```

最近三天的正式半径实验统一采用 strict 1+5 动态用户、lifetime=10、Cartesian Spatial Flight Actor、completion-priority 用户排序、Reset Curriculum、5 套独立 Actor/Critic、seed=2、64 rollout threads。除明确标注的消融外，学习率、PPO、奖励和环境参数保持一致。

当前 curriculum 为：

- 0--20M：`p_random=0.5`；
- 20--50M：从 0.5 线性降到 0；
- 50M 后：全部固定初始位置；
- random reset 先生成满足最小间距的无标签五点集合，再随机分配给 5 个 UAV ID。

训练期间不额外启动 eval worker，避免降低 FPS；记录 overall 与 fixed/random 条件指标。正式判断以共同 TensorBoard 步数、固定起点评估和独立训练 seed 为准。

### 9.2 三天实验矩阵与已得到的结果

所有百分比均按组内共同 TensorBoard step 的最后 25 个共同事件点计算，主指标为 `agent0/system_performance_true_all_GUs`。

| 实验族 | 目的 | 共同步数结果 | 当前判定 |
|---|---|---|---|
| 旧统一归一化残差 R0/R260/R1000 | 检查“半径越大、共识残差越小”是否自动带来收益 | 54.656M：R260−R0 `−3.10%`，R1000−R260 `−5.07%` | 稳定反序，否定该机制 |
| Mixed-consensus R0/R260/R1000 | 检查 `A_i+0.5*consensus` | 53.018M：`−5.71%`、`−2.68%` | 超过50M仍反序；fixed-reset同方向 |
| Local-`A_i` R260/R1000 | 只改变 critic 可见范围 | 53.376M：R1000−R260 `+0.004%` | 几乎无效；更多critic信息不自动改善策略 |
| Pure-consensus R260/R1000 | 检查纯团队优势 | 38.733M：R1000低 `16.34%` | 信用分配失败，已停止 |
| Ego-query vs mean-pool critic | 检查critic读出结构 | 38.579M：mean-pool低 `0.22%` | 不是主矛盾 |
| R520 K=1/5/50 | 检查共识轮次 | 34.074M早期 K1→K5 `+1.21%`、K5→K50 `+0.87%`；62.438M K50反低K1 `0.66%` | 早期弱单调不持久，不能声称K越大越好 |
| Externality beta=.025/.05/.10 | 检查外部性优势权重 | 8.32M：`.025 > .05 > .10` | beta越大越差，三路已停止 |
| Mean-pool Actor task message A0/A1/A2 | 同时扩大Actor消息和critic半径 | 29.005M：R260−R0 `−0.77%`，R1000−R260 `+1.44%` | 只完成一半目标 |
| Zero Actor message A0/B1/A3 | 在同一消息架构下只改变critic半径 | 29.670M：`−0.32%`、`−11.76%` | critic-only仍不成立 |
| R1000消息内容 zero/geometry/task | 隔离可行动消息 | 29.005M：geometry−zero `+13.99%`，task−geometry `+0.64%` | 邻居几何贡献绝大部分收益 |
| 高速MD v=10 的 R0/R260/R1000 | 检查更强移动耦合 | 29.773M：`+0.40%`、`−5.39%` | 提速不能解决R1000问题 |
| Receiver-gated Actor message G0/G260/G1000 | 避免R1000无关邻居被均值池化 | 21.9904M overall：`+0.225%`、`+1.040%`；fixed：`+0.262%`、`+1.013%` | **首次得到完整目标顺序，保留到50M** |

### 9.3 当前最重要的机制结论

1. “critic 看见更多 UAV 状态”不足以推出最终策略更好。Local-`A_i` 的 R260/R1000 在 53M 后只差 `0.004%`，说明当前局部 baseline 已经足够，更多 critic 信息没有转化为可测收益。
2. 纯共识优势和较强 externality 项会加重跨 UAV 信用分配噪声；继续调 `A_mean`、beta 或人工残差不是当前主线。
3. 执行时 Actor 信息才是通信半径产生价值的关键。R1000 下 task message 相对 zero 提升 `14.72%`；其中几何消息已经贡献 `13.99%`，复杂任务摘要只再增加 `0.64%`。
4. 简单 mean pooling 会把新增远端邻居全部放入平均分母，R1000 容易发生消息稀释。receiver-gated fixed-sum 主要改善 R1000：同半径比较中 gated−mean 为 R0 `−0.54%`、R260 `−0.07%`、R1000 `+1.77%`。
5. receiver-gated 是目前唯一同时在 overall 与 fixed-reset 训练曲线上得到 `R0<R260<R1000` 的候选，但仅为 seed=2、约22M的筛选结果，不能写成论文最终结论。

### 9.4 Receiver-gated 实现合同

门控消息只进入 Spatial Flight head，不改变关联、带宽、计算资源 Actor、Critic目标、奖励、优势或环境公式：

\[
h_i^{msg}=\frac{1}{4}\sum_{j\ne i}m_{ij}g_{ij}\phi(x_{ij}),
\qquad
g_{ij}=\sigma(f_g(d_i,\phi(x_{ij}),\Delta p_{ij})).
\]

- 分母固定为 `M-1=4`，不随当前邻居数量变化；
- 每个 sender 使用独立 sigmoid gate；
- gate 由接收UAV自身描述、sender编码与相对位置共同决定；
- gate bias 初始化为 `logit(0.1)`；
- 新模块使用隔离 RNG 初始化，避免改变后续 5 套网络初始化随机流；
- 实现提交：`e2ea522`，RNG隔离修正：`55d4fdb`；
- 远程独立 clone：`/data/home/tanglanProf_user02/wyj/Projects/on-policy-gated-55d4fdb`。

### 9.5 当前设备与实验位置

- 本地 Windows：`D:\wyj\Projects\on-policy-last-obs-clean-5UAV`，运行 Mixed R0/R260/R1000；2026-08-03 09:34 冻结曲线时均已超过 53M。
- 远程 `114.212.200.233:9001`：Actor-message 九路、receiver-gated 三路；共享结果位于 `/data/home/tanglanProf_user02/wyj/Projects/` 下对应 clone。
- 远程 `114.212.200.233:9012`：R520 K1/K50 继续运行，K5 已在约34M停止并保留日志/模型。
- 本地仓库 HEAD：`36909ee`，分支 `agent/dcppo-runtime-optimization`；相对 origin ahead 1。运行中的远程实验来自固定独立 clone，不应直接假定与本地未推送提交同步。

### 9.6 下一步 gate（不要临时改规则）

1. 保留 G0/G260/G1000 到 50M，不再新增 advantage、beta、MD速度或消息字段网格。
2. 在 50M 冻结三路 checkpoint、normalizer 和 args；使用预先固定的开发评测种子做固定起点确定性评估。
3. 同时报告 True60、任务完成率、接纳率、平均active、轨迹/部署和碰撞/边界事件；要求完成率和接纳率不能通过明显下降换取True60。
4. 对 G1000 做同checkpoint反事实：正常消息、全部屏蔽消息、仅屏蔽距离大于260m的消息，确认策略是否实际利用新增远端包。
5. 如果50M fixed-start仍保持两个相邻正差，再启动至少3个独立训练seed；只有均值满足顺序且至少2/3 seed同序，才形成论文级结果。
6. 若50M顺序消失，优先检查gate分布、远端消息实际使用率和任务包方向信息；不要回到人为按半径加噪声制造排序。

### 9.7 最新证据入口

- **最近三天29条实验总审计与11张配对曲线**：`paper_artifacts/diagnostics/dcppo_radius_review_20260803/README.md`
- 精确实验汇总：`paper_artifacts/diagnostics/dcppo_radius_review_20260803/experiment_summary.csv`
- 共同step比较：`paper_artifacts/diagnostics/dcppo_radius_review_20260803/comparison_last25_common_step.csv`
- 冻结事件与 args：`paper_artifacts/diagnostics/dcppo_radius_review_20260803/sources/`
- 远程路径/资源快照：`paper_artifacts/diagnostics/dcppo_radius_review_20260803/probe_9001*.json`、`probe_9012.json`
- 通信半径专项完整实验账本：`DCPPO_RADIUS_TUNING_20260802.md`

### 9.8 结论边界

- 当前只有一个训练 seed；训练曲线中每个episode的64环境均值不能替代多训练seed。
- receiver-gated结果刚过20M gate，尚未到50M固定checkpoint评估。
- `R0<R260<R1000` 是当前候选机制结果，不是已经完成的论文结论。
- 不同实验族只在内部做匹配比较；不能跨优势模式、速度或消息架构直接按绝对True60排名。
- 旧论文Fig.4的固定60用户和历史后处理与当前动态MD协议不同，仍只能用于量级参考。

## 10. 2026-08-03 跨设备最终刷新与实验收缩

### 10.1 当前准确结论

- 截至共同36.1728M，receiver-gated三路为`R260 < R0 < R1000`：末25为543.565k / 544.253k / 548.457k，末100为542.423k / 545.670k / 549.083k。
- 21.9904M时出现的`R0 < R260 < R1000`是暂态现象，不能再表述为“已经得到目标排序”。
- 本地mixed-consensus到共同67.2512M呈`R0 > R260 > R1000`；mean-pool Actor消息在低速、高速下也未形成目标排序。
- R1000 geometry-message相对zero-message提升约12%–13%，但完整task-message没有稳定超过geometry；这是目前最可靠的组件结论。
- MD速度提高到10m/s没有放大通信收益；R520 K=50也没有优于K=1。停止继续扩展这两类网格。

### 10.2 当前运行状态

- 已停止并保留全部输出：本地mixed-consensus三路、9001 mean-pool九路、9012 K1/K50；K5此前已停止。
- 唯一继续运行：9001 receiver-gated R0/R260/R1000，各65进程，运行到50M checkpoint。
- 第二台远程SSH的实际端口为`9012`，不是`9002`。

### 10.3 下一 gate

到50M后统一做固定起点确定性评估。若仍无完整排序，先对同一checkpoint做正常消息、全屏蔽消息、屏蔽距离大于260m消息的反事实；只有固定评估同序后才追加多seed。

完整数值、逐组目的、曲线和停机记录：`paper_artifacts/diagnostics/all_devices_refresh_20260803/README.md`。

## 11. 2026-08-03 RandomLayout-700 对称起点复核

- 正式MEC环境已接通经过验证的600/700m `episode_template4`；`fixed_legacy`仍严格保留600m旧行为。
- 700m使用中心对称十字固定起点；四种热点只在episode reset时采样一次，400时隙内不跳变；R1000保证完全图。
- 对称起点下重新运行600/700各10,000次几何reset和2,000个共享轨迹配对episode。
- 600m：R260-R0 `+6.036%`（99.95%正），R1000-R260 `+0.035%`（34.70%正）。
- 700m：R260-R0 `+7.579%`（99.90%正），R1000-R260仅`+0.158%`（57.05%正）。
- 700m未通过预注册的R1000-R260至少0.5%且70%配对episode为正的训练门槛；旧起点下`+0.639%/79.3%`包含明显起点几何贡献。
- 决策：不启动三路30M/100M PPO。完整58项测试通过；单路800-step gated-task R1000训练smoke成功。
- 证据：`paper_artifacts/diagnostics/random_layout_symmetric_preflight_20260803/README.md`。

## 2026-08-03: MovingHotspot700 30M radius experiment is active

The new `episode_moving_template4` mode moves only the hidden MD birth-intensity rectangles; it does not expose hotspot coordinates to the policy and does not drag or delete active MDs. Local/remote tests passed (9 tests plus an 800-step PPO smoke). Four matched 30M, 64-worker, seed-2 runs are active on remote 9001 in `/data/home/tanglanProf_user02/wyj/Projects/on-policy-movinghotspot700-20260803`: R0 PGID 100806/GPU5, R200 PGID 22722/GPU5, R400 PGID 61580/GPU6, and R600 PGID 100439/GPU7. All use local advantage, task-summary receiver-gated Actor messages, ego-query attention critic, Spatial Flight Actor, completion-priority ordering, curriculum v2, and shared return normalization. Only `neighbor_distance` and `neighbor_R` change together. The abandoned 10M pilot logs were retained; do not mix them with the clean 30M curves.

## 12. 2026-08-06：8 月 1–5 日周总结重排后的 authoritative state（历史状态）

- 本周三条任务线、已保存训练事件、评估 CSV/JSON 和轨迹图已重新梳理；当时的权威入口为 RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md。
- 当前稳定主基线是 Q2；10 个固定评估 episode 的 True60 为 617,738 ± 4,096。旧的 612,606 ± 3,039 是过期快照，不再使用。
- 当前唯一值得继续做正式验证的通信候选是 RandomLayout700-v2 Full template12、无 UAV curriculum、receiver-gated task/geometry message 的两 seed 阶梯；它仍受 R1000 小区域覆盖捷径约束。
- 优势混合、Critic-only 半径扩展、pure consensus、externality beta、K、MD 高速、mean-pool 大网格、FixedDiag 和未完成 MovingHotspot 均已保留证据但不再扩展。
- 12-layout UAV curriculum 和 Subset29 已完成或保存到可复查状态，但不能与无 curriculum 主线直接合并排名；episode_layout_context 截止日无最终结果。
- 8 月 6 日之后启动的 v_max=10/20/30/40 运行不属于本周完成结果。后续先做固定起点评估、small/large 分区指标和消息反事实，只有通过才补训练 seed。

本节及其之前的主动运行描述均为历史时间点记录；若与 2026-08-12/13 的 Fixed600-200 主线归档冲突，以文首当前阅读顺序所列新文档、后续归档段落及对应原始 JSON/CSV 为准。

## 2026-08-09: 600m Random12 input-v2 experiment preparation

- Baseline work was preserved first in commit `44b7575`.
- Added `episode_template12_600_200`: 12 ordered 200m/400m corner layouts on the 600m map.
- Added versioned raw-message and meter-layout contracts. New message content enters `ob_norm`; only each packet mask is preserved.
- Added a matched noise-3.0 R0/R260/R520/R780 launcher and backward-compatible evaluation masking.
- Design and commands: `RANDOMLAYOUT600_NOISE3_INPUT_V2_20260809.md`.
- No new long training run was launched during this implementation.

## 2026-08-12：最终环境、算法、Actor/Critic 输入与速度参数审计

- 新增权威审计入口：`FINAL_ENV_ALGORITHM_PARAMETER_AUDIT_20260812.md`。
- 当前速度/参数选择环境是 Fixed600-200 index0、严格 1+4 动态 MD、lifetime=12、无 UAV curriculum；Random12-600 作为单独泛化协议，不与速度曲线合并排名。
- 完整 60M 的 seed32/42 表明 v30 tail100 两 seed 均值 562.6k，明显高于 v20 的 496.7k 和 v10 的 485.0k；当前冻结 v30。2026-08-13 01:30 的 v40 冻结事件三 seed 共同只到 21.7856M，尚未进入最终选择。
- 实际算法不是共享策略 MAPPO：`--share_policy` 是 store-false，使用 separated runner 和五套独立 actor/critic；`--use_valuenorm` 同样是 store-false，实际为 shared return RMS、ValueNorm off。
- Actor 原始输入 251 维；资源头删除 40 维消息后读 211 维；Spatial flight head 使用空间/移动/布局/receiver-gated task-summary。Critic 输入为 R520 内最多 5 个 211 维 token，共 1055 维，ego token 查询 attention。
- `neighbor_distance=520` 同时控制 Actor 消息、Critic mask 和 Metropolis 图；`neighbor_R=520` 因 `actor_neighbor_obs=false` 不进有效路径。noise=3 在 R520 下的有效 `m_i` 约 1e-6，训练优势实际上接近 local A_i。
- 远端速度 run 的 12 份 args 和事件快照已冻结，但远端无 Git 元数据且本轮没有速度专用源码 SHA manifest；参数合同已确认，逐字节源码一致性仍是溯源边界。

## 2026-08-12：no Actor message 对照复核

- 8 月 10 日远端确实启动了 Fixed600-200/MD12/v30/noise3 的 R0/R260/R520 `no_actor_message` 三条训练；本地最后冻结快照为 38.07M/38.37M/40.88M，未保存到 60M 最终状态。
- R520 no-message 与 actor-message seed2 的 `args.json` 除实验名和 `actor_message_mode=disabled/task_summary` 外完全一致。共同 40.8832M：tail25 为 547,509/547,601，message 仅 +0.017%；tail100 为 548,930/551,410，message +0.452%。这支持“当时看起来没明显差别”，但仍只是单 seed、未满预算的训练曲线诊断。
- `disabled` 是结构删除消息 block：Actor obs 251→211，Spatial flight head 不创建 message encoder/gate；资源头本来就只读相同的 211 维本地输入。Critic 仍保留 R520 的 5×211=1055 维 attention state，共识/优势也仍使用 R520。
- 详细说明已追加到 `FINAL_ENV_ALGORITHM_PARAMETER_AUDIT_20260812.md` 第 11 节。

## 2026-08-13：Actor-message/速度主项目 Git 归档

- 新增主线归档索引：`ACTOR_MESSAGE_SPEED_ARCHIVE_20260813.md`；后续接手优先读取该文件，再进入完整参数审计和速度记录。
- 本次归档只纳入主线源码、正式启动器、参数/曲线轻量汇总、生成脚本和文档直接引用的最终图；明确排除模型、TensorBoard 原始事件、控制台训练日志和历史分析中间目录。
- 本地 v10/v20/v30 seed2 在提交时仍在训练；本次操作只读刷新曲线，没有停止或修改训练进程。
- `feature/unreliable-dataplane` 的 GRU/不可靠通信 worktree 不属于本次主项目提交，继续独立保留。
# 2026-08-13：不可靠数据面同步到最终速度基线

- `feature/unreliable-dataplane` 的 2026-08-01 早期分叉已重建到速度分支归档提交 `fa53f3a` 之上；环境参数、Spatial Actor、Actor-message、ego-query critic、shared return normalization、advantage modes 和 checkpoint manifest 均以速度分支为准。
- 保留并适配 Type-S 不可靠 critic 数据面、Type-A running-sum advantage 通信，以及共享 MD-GRU/last-observation/zero 三种重建模式。
- 新正式入口：`onpolicy/scripts/train/run_fixed600_200_unreliable_dataplane.ps1`。默认关闭 Actor message，避免当前仅距离门控的 task-summary 成为可靠旁路；可显式开启做速度合同兼容对照。
- 公平重建对比固定 `critic_md_metadata`，critic 为 `5×271=1355`；no-message Actor 为 211，task-summary Actor 为 251。
- `externality_consensus` 不允许与不可靠 running-sum 混用；其可靠图系数语义未被伪造。
- 验收：全套 `95 passed`，以及 reliable-zero / unreliable-zero / unreliable-last_obs / unreliable-md_gru 四组端到端 2-slot smoke 全部完成 PPO 更新和 checkpoint。
- 详细合同见 `UNRELIABLE_DATAPLANE.md` 第 7–10 节。

## 2026-08-19：可靠主线 8 月 14—19 日增量已适配

- 再次确认 `on-policy-last-obs-clean-5UAV` 的 `agent/dcppo-runtime-optimization` 是可靠通信/Actor-message/速度实验主线；逐提交审计见 `UNRELIABLE_SPEED_COMPATIBILITY_AUDIT_20260813.md` 第 13 节。
- 已接入 association threshold、deadline prefilter 开关、任意 UAV 数、显式起点优先、异构 per-UAV 带宽/算力，以及动态 agent 数的 checkpoint/render。
- 原有 Type-S 丢包、receiver-local GRU、Type-A running-sum、优势模式和 PPO 后处理保持不变；Actor-message 仍可消融且不伪装成不可靠链路。
- 新增六 UAV 正式入口 `onpolicy/scripts/train/run_fixed600_200_6uav_unreliable_dataplane.ps1`；五 UAV 入口也已暴露新参数。
- CPU-only 全套 `107 passed, 2 warnings`；4 个启动器语法解析和 5/6 UAV argv probe 通过。没有启动 GPU 或长训练。
- `hybrid_actor.py` 仅服务可靠主线的离线 checkpoint 混合评估，没有进入训练 import graph，故未纳入不可靠算法。
## 2026-08-13：MD-GRU 论文/实现训练闭环复核

- 论文 IV-D 要求每 UAV 一套本地 GRU 参数、每个 receiver/MD session 一份独立 hidden；旧实现只有一套跨 UAV 共享模型，已改为五套相同初值但本地独立更新的 predictor/optimizer。
- 在线 hidden 在 episode reset 清空；监督 sequence replay 跨 episode 保留、有界 FIFO，保存原始观测短序列并在训练时用当前参数重算 hidden，避免 recurrent-state staleness。
- GRU 采用独立 Smooth-L1 监督更新；PPO actor/critic 梯度不进入 GRU，重构只供 critic。`md_prediction_loss_coef` 仅保留兼容性，不再缩放 optimizer loss。
- runner 更新顺序已改为 `compute -> PPO -> predictor -> next rollout`，防止一个 on-policy batch 内 estimator 版本漂移。
- 启动脚本新增 `-CPUOnly`。本机 `marl` 的 PyTorch 2.12 CUDA wheel 在禁用 GPU 后可计算有限 GRU 梯度，但最小独立脚本也会在 CPU 原生后端出现 `0xC0000005`，故不要把该环境当作 CPU 训练验收环境。
- 论文工作树本轮只读，原有未提交 `paper.tex/ref.bib/pdf/bbl` 等修改未触碰。

## 2026-08-13：不可靠/速度合同二次审计

- 证据与兼容矩阵见 `UNRELIABLE_SPEED_COMPATIBILITY_AUDIT_20260813.md`。
- 当前环境/Actor/PPO 主合同继承速度提交 `fa53f3a`；Type-S、Type-A 和 critic-only reconstruction 是正交叠加。
- 速度兼容优势是 `per_agent_noise=3`；论文 IV-D 直接使用 running-sum 是 `pure_consensus`。固定启动器现已显式开放该选择。
- 优势通信使用 reset 前的终局 UAV 位置；自 2026-08-19 起 `neighbor_distance/d_com` 同时硬门控 Type-A 名义范围图，旧全连接语义已废止。
- Actor-message 在不可靠项目里仍只有几何门控；若开启，应标记为速度兼容 Actor 路径，而不是全链路不可靠。
- 同时修复 residual 零分母端点、GRU 5 m/s 速度编码和 separated eval 当前环境/训练器接口。
- CPU-only 隔离断言通过 17+11+8+2 项，两个 launcher argv probe 通过；本机长 Python 进程仍可复现已知 `0xC0000005`，所以完整 PPO CPU 验收仍未完成。
- GRU 当前 replay capacity/train samples/min-ready samples 为 `32768/512/512`；不要再把 capacity 当成每个 PPO update 的训练样本数。
- Type-S timely-reception KPI 现在只计 R 内几何邻居的有向尝试，并在每个 episode reset 清零；此前 R 外抽样和跨 episode 累计只影响日志口径，不改变 critic 的几何 mask。

## 2026-08-19：Type-A 范围图、P_c/d_com 与 natural-only GRU 定稿

- 修复 Type-A 的旧完全图假设：按 rollout 终局位置和可调 `d_com` 建图，running-sum 使用每个 sender 的实际 `out_degree+1`；Type-S/Type-A 共用同一范围门控。
- 默认 `d_com=520 m`、`P_c=1.1809658836 W`、`K_c=10 dB`、`H=50`。距离/功率可只给一项并反算另一项；两项同时给出时误差必须不超过 5 m。
- 保留 `per_agent_noise`，不加入 `|C_i|/M` 缩放；修复 local 恰为全局均值但 consensus 偏离时 residual 被错误置零的问题。
- GRU 默认只训练真实信道自然重观测样本，不制造人工缺失；新增 replay-label/rollout-query age 分桶及 GRU-vs-last_obs prequential RMSE。
- 预测器仍为 receiver-local 独立 Smooth-L1/Adam；重构 NumPy state 只供 critic，PPO 梯度不进入 GRU。checkpoint 是模型/optimizer warm-start，不是含 replay 的 bitwise exact resume。
- CPU-only 定向回归 `48 passed`，全套 `117 passed, 2 warnings`；5/6-UAV launcher 语法通过，5-UAV power-only argv probe 通过。未启动长训练或 GPU 工作负载。
- 追加 2-slot/1-worker CPU-only 训练闭环 `codex_cpu_smoke_rangegraph_gru_20260819/run1`：PPO 更新完成，args 为 520 m / 1.1809658836 W / 10 dB / H50 / per-agent-noise / md_gru，checkpoint manifest 和 `md_gru_shared.pt` 正常生成；仅作执行验证，不作性能证据。

## 2026-08-20：3090 unreliable-zero 结果回收与四路曲线

- 通过历史 3090 连接流程只读登录 `test@114.212.117.24`，将完整远端目录下载并按相对路径/文件大小清单校验到 `onpolicy/scripts/results/mec/mappo/F600_B0_unreliable_control_zero_seed2_60m_cuda`。远端 `run2` 是 59.9808M 完整结果；`run1` 是 0.0768M 不完整重复，不纳入曲线。
- 四路同图产物：`analysis/fixed600_four_way_20260820/plot.png`；精确数据 `curves.csv`；重绘脚本 `plot.py`；审核 `audit.md`/`final-status.md`。指标统一为 `agent0/system_performance_true_all_GUs`，无平滑、插值或外推。
- 完成曲线末点：本地 reliable zero `556634.75`，远端 3090 unreliable zero `558398.94`，后者高 `1764.19`（约 `0.317%`）。最新刷新时本地 MD-GRU 到 `2.0736M`，last-observation 到 `1.2032M`；两者仍在训练，不能据此宣称最终优劣。
- 比较边界：reliable reference 的 `critic_md_metadata=false`，三个 unreliable/current run 为 `true`；因此当前图是运行状态对比，不是严格单变量消融。四路实验均为 seed 2，但最终比较应等待两个本地 run 到 60M，并统一 metadata 配置。
- 配置审计见 `analysis/fixed600_four_way_20260820/config-audit.md`：所有主要环境/PPO参数一致，但 Reliable zero critic 输入为 211 维，三个不可靠/current critic 为 271 维（多出 20 个 MD 槽各 3 个 metadata 特征）。`per_agent_noise` 下两种通信模式还使用不同 consensus 路径决定噪声幅度；因此 5--11M 的 unreliable 领先可由 critic metadata、优势噪声分布和单 seed 随机性解释，不能解释成丢包更优。

## 2026-08-21：本地重启后的 MD-GRU/last-observation 断点

- 本地重启后两个 trainer 均已退出，但 checkpoint 完整且 manifest 校验通过：MD-GRU run1 到 `12,569,600` steps，last-observation run1 到 `13,721,600` steps。
- 按 runner 的 rollout 对齐，名义 60M 实际目标为 `59,980,800`；恢复所需剩余步数分别为 `47,411,200` 和 `46,259,200`。
- `run_fixed600_200_unreliable_dataplane.ps1` 已增加 `-ModelDir`。恢复时沿用原 experiment name 会新建 `run2`，不会覆盖 run1。恢复加载 actor/critic、normer 和 MD-GRU predictor，但不是 bitwise exact resume，因为不保存 PPO optimizer state、环境/RNG、rollout buffer 和在线 hidden。
- 诊断结论见 `analysis/fixed600_four_way_20260820/diagnosis.md`：5--11M 的 unreliable-zero 暂时领先主要不是“丢包更好”，而是其 critic 额外接收 60 维 MD metadata，输入 token 为 271 维；reliable token 为 211 维。该区间平均领先 6.683%，20--60M 平均只领先约 0.388%。后续需在同一 commit/设备上做 metadata=false/true 的 2×2 匹配实验。

## 2026-08-20：Reliable zero 与历史 Fixed600-200 基准

- 历史算法消融基准已经定位为 `dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed2_60m_20260814`，而不是另一条旧的 PPO-epoch 消融曲线。
- 当前 Reliable zero 与该基准的参数文件在可比字段上完全一致；比较图和逐字段审计见 `analysis/fixed600_reliable_vs_historical_baseline_20260820/`。
- 当前曲线在 5--11M 平均低约 6.28%，公共末点低 1.63%，但历史事件文件没有 source SHA，且两次运行不是同一时刻/同一进程状态；先按单 seed 运行差异记录，不应直接归因于参数或判定代码错误。

## 2026-08-23：8 月 21 日后不可靠通信实验跨设备权威快照

> 快照时间：2026-08-23 22:13（Asia/Shanghai）。本节按真实 `args.json`、TensorBoard 事件和本地/远端进程核对，覆盖 8 月 21 日以后完成、停止或仍在运行的相关实验；它取代上文“两个 seed-2 run1 仅等待恢复”的历史状态。

### 统一的不可靠通信正式协议

四条无 curriculum 的正式 MD-GRU/last-observation 实验在所有序列化参数上严格匹配：同 seed 的两种重构只差 `experiment_name` 和 `state_reconstruction`，同一种重构的 seed 2/32 只差 `experiment_name` 和 `seed`。

- Fixed600-200 index 0，600×600 m，5 UAV 固定起点，严格每 slot `1+4` MD 到达，lifetime=12，MD 平均速度 3 m/s，UAV `v_max=30 m/s`；
- `episode_layout_context=meters_v2`，Actor-message disabled，deadline prefilter disabled，`association_threshold=0.5`；
- Type-S/Type-A 共用 R520，`P_c=1.1809658836 W`，`K_c=10 dB`，running-sum `H=50`；
- `per_agent_noise`、noise scale 3、local reward、Spatial Cartesian Flight Actor、completion-priority user sort、R520 k-hop ego-query attention critic；
- separated 5-policy MAPPO/PPO、shared return normalization、64 rollout workers、名义 60M（实际 rollout 对齐终点 59.9808M）、clip=0.15、gamma=0.99、PPO epoch=4；
- MD-GRU 使用 receiver-local 独立模型、自然重观测监督、rollout-local session buffer，以及每 receiver `2048 targets × 10 batches × 1 epoch`；PPO 梯度不进入预测器。

### 本地正式训练结果

主指标是原始 TensorBoard `agent0/system_performance_true_all_GUs`；`tail20` 为最后 20 个记录点的均值。

| 重构/训练 seed | curriculum | 事件终点 | tail20 | 状态 |
|---|---|---:|---:|---|
| MD-GRU, seed 2 (`run2`) | off | 59.9808M | 552,225.89 | 从头重跑并完成 |
| last-observation, seed 2 (`run2`) | off | 59.9808M | 489,771.21 | 从头重跑并完成 |
| MD-GRU, seed 32 (`run1`) | off | 59.9808M | 492,798.01 | 完成 |
| last-observation, seed 32 (`run1`) | off | 59.9808M | 493,493.62 | 完成 |
| MD-GRU, seed 2 | p=0.7, 10M--25M | 32.0256M | 545,211.80 | 主动提前停止 |
| last-observation, seed 2 | p=0.7, 10M--25M | 54.0416M | 551,480.01 | 主动提前停止 |

- 8 月 20 日被重启打断的 seed-2 `run1` 仍保留：MD-GRU 事件到 12.5696M，last-observation 事件到 13.6960M（后者 checkpoint 到 13.7216M）。后来的 `run2` 的 `model_dir` 为空，实际是从头重跑，不是 warm resume，不能把 run1/run2 步数相加。
- seed 2 的 MD-GRU tail20 比 last-observation 高 12.75%，但 seed 32 低 0.14%。两 seed tail20 的简单均值差为 +6.28%，样本数只有 2 且方差很大，不能据此宣称稳定 RL 收益。
- 预测器本身在两个 seed 上都学到了稳定的短期状态估计：seed 2 的验证 RMSE tail20 为 0.799 m，对应 last-observation 1.910 m；seed 32 为 0.817 m 对 1.951 m，均约降低 58%。这证明“预测误差更低”，不自动证明“最终策略一定更好”。
- 在 32.0256M 的 curriculum 公共终点，历史 Fixed600-200、MD-GRU curriculum、last-observation curriculum 的 tail20 分别为 547,808.46、545,211.80、546,614.98，三者差距小于 0.5%。curriculum 主要消除了部署探索瓶颈，也消除了 MD-GRU/last-observation 的可辨识差异；因此 curriculum 不进入证明 GRU 必要性的主消融。
- 26.7776M 的 seed-2 MD-GRU 冻结 checkpoint 已做 3 个固定 layout-0 确定性 episode：True-all 为 544,403 / 542,913 / 557,552，均值约 548,289；轨迹显示 1 架覆盖左下、其余 UAV 在右上大区域形成分工。这只是中期 checkpoint 行为检查，不是最终多 seed 评测。

### 8 月 21 日后可靠侧诊断及设备状态

这些训练位于可靠主线 `on-policy-last-obs-clean-5UAV`，服务于解释不可靠曲线、优势噪声和通信半径，不是 MD-GRU 重构实验。它们均为 `state_reconstruction=zero`，不能与上表合并成单变量 GRU 排名。

- 远端 3090 的 PPO 敏感性组使用旧的 deadline-filter ON 协议并已完成：clip 0.05/0.30 的 tail20 为 533.110k/548.438k；gamma 0.90/0.95 为 443.755k/559.350k；PPO epoch 1/10 为 546.419k/501.054k。原 clip=0.15、gamma=0.99、PPO epoch=4 仍是冻结主设置；这组不能直接与 deadline-filter OFF 的不可靠正式实验比较。
- R0 `local_mean_per_agent_noise` 的 noise=0/0.4/0.8/1/2/4 六路均在约 19--21M 停止。noise=0 的 tail20 最高（460.116k），增大噪声没有形成可靠收益；这是筛选，不是满 60M 结论。
- deadline-filter OFF、`per_agent_noise=0` 的远端 R0/R260/R520 分别停止于 57.3184M/41.1392M/40.6784M，tail20 为 482.939k/543.579k/546.270k；半径增大明显改善该单 seed 训练，但终点不齐，仍只能作机制诊断。
- `local_mean_per_agent_noise=0` 的 R260/R520 本地运行停止于 43.7504M/43.6992M，tail20 为 497.139k/499.172k；配套 R0 只有远端 19.2256M 快照，不能作最终半径排序。
- 截至快照时，本地仍运行 `per_agent_noise=2` 的 R0/R260/R520：32.0768M/32.0256M/29.0048M，tail20 为 473.292k/540.821k/528.863k。
- 远端 3090 仍运行 `per_agent_noise=0.8` 的 R0/R260/R520，GPU PID 为 2183268/2184235/2184654：34.9952M/33.9200M/34.1760M，tail20 为 476.917k/541.782k/539.544k。三进程各使用约 6.2 GiB 显存。
- 远端 `/home/test/wyj/Projects/on-policy-unreliable-dataplane` 在 8 月 21 日以后没有新 TensorBoard 事件；真正的不可靠 MD-GRU/last-observation 正式训练均在本地 Windows 完成。远端 3090 当前跑的是可靠侧优势/半径诊断。

### 当前结论与下一步

1. 保持无 curriculum 作为 MD-GRU 必要性主消融；curriculum 只可作为独立的训练技巧结果，不能混入主比较。
2. 当前最可靠的算法事实是“GRU 的自然重观测预测误差在 seed 2/32 上均显著低于 last-observation”；策略收益仍高度 seed-sensitive，至少补一个预先固定的第三训练 seed，并统一做最终 checkpoint 多 episode 评测后再判断。
3. 完成当前本地 noise=2 与远端 noise=0.8 三半径组后，只在共同 step 比较；不要按异步最新点排名，也不要把可靠侧 `zero` 诊断当作 MD-GRU 消融。
4. 严格可靠/不可靠通信对比仍需统一 `critic_md_metadata`。历史 reliable zero 是 211 维 critic token，而当前不可靠重构是 271 维，旧四路图不能承担单变量通信可靠性结论。

证据入口：

- 无 curriculum 主图：`analysis/fixed600_four_way_20260820/plot.py`、`curves.csv`、`plot.png`；
- curriculum 三路：`analysis/fixed600_curriculum_three_way_20260822/`；
- 三种可靠基准：`analysis/fixed600_three_reliable_baselines_20260823/`；
- 可靠侧优势/半径：`on-policy-last-obs-clean-5UAV/analysis/remote_r0_noise_mode_comparison_20260822/`、`radius_noise0_comparison_20260822/`、`peragent_noise_radius_compare_20260823/`；
- MD-GRU 三 episode 评测：`onpolicy/scripts/results/mec/mappo/F600_B0_proposed_unreliable_mdgru_tb2048x10_seed2_60m_cuda/run2/eval_3episodes_latest_step26777600/`。
