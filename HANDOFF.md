# 5-UAV 动态 MEC 实验交接

> 状态快照：2026-08-06（Asia/Shanghai；周总结依据最新已保存证据）
>
> 当前分支：`agent/dcppo-runtime-optimization`
>
> **阅读提示：第 1--11 节含历史快照；当前状态、最新结论和待办以文末 2026-08-06 authoritative state 以及周总结为准。**

> **周级交接更新（2026-08-05，历史记录）：** 8 月 4–5 日的环境实验已统一整理；章节编号随后在 2026-08-06 周总结重排中改变。当前应阅读文末 2026-08-06 authoritative state 和 `RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md`。

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

## 12. 2026-08-06：8 月 1–5 日周总结重排后的 authoritative state

- 本周三条任务线、已保存训练事件、评估 CSV/JSON 和轨迹图已重新梳理，权威入口为 RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md。
- 当前稳定主基线是 Q2；10 个固定评估 episode 的 True60 为 617,738 ± 4,096。旧的 612,606 ± 3,039 是过期快照，不再使用。
- 当前唯一值得继续做正式验证的通信候选是 RandomLayout700-v2 Full template12、无 UAV curriculum、receiver-gated task/geometry message 的两 seed 阶梯；它仍受 R1000 小区域覆盖捷径约束。
- 优势混合、Critic-only 半径扩展、pure consensus、externality beta、K、MD 高速、mean-pool 大网格、FixedDiag 和未完成 MovingHotspot 均已保留证据但不再扩展。
- 12-layout UAV curriculum 和 Subset29 已完成或保存到可复查状态，但不能与无 curriculum 主线直接合并排名；episode_layout_context 截止日无最终结果。
- 8 月 6 日之后启动的 v_max=10/20/30/40 运行不属于本周完成结果。后续先做固定起点评估、small/large 分区指标和消息反事实，只有通过才补训练 seed。

旧的主动运行描述是历史时间点记录；如果与上述状态或周总结冲突，以本节、周总结和对应原始 CSV/JSON 为准。

## 2026-08-09: 600m Random12 input-v2 experiment preparation

- Baseline work was preserved first in commit `44b7575`.
- Added `episode_template12_600_200`: 12 ordered 200m/400m corner layouts on the 600m map.
- Added versioned raw-message and meter-layout contracts. New message content enters `ob_norm`; only each packet mask is preserved.
- Added a matched noise-3.0 R0/R260/R520/R780 launcher and backward-compatible evaluation masking.
- Design and commands: `RANDOMLAYOUT600_NOISE3_INPUT_V2_20260809.md`.
- No new long training run was launched during this implementation.
