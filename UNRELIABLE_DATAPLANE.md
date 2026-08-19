# Type-S 不可靠数据面与 MD 状态重建契约

本文档说明当前代码闭环。优势函数由 `advantage_mode` 显式决定：速度基线使用的 `per_agent_noise` 以 local advantage 为确定性主体，由通信共识残差调节随机扰动；论文 IV-D 所写“直接使用 running-sum 估计”则对应 `pure_consensus`。二者不能混称为同一算法。

## 1. Type-S packet 的语义

Type-S packet 是发送 UAV 当前 critic 局部状态块的结构化、带身份编码。发送端只使用一套 MD roster builder：

1. 筛选发送 UAV 覆盖范围内的活动 MD；
2. 按当前实验配置执行 completion-priority 或 distance-only 排序；
3. 使用 session ID 作为距离相同时的确定性 tie-breaker；
4. 截取前 `max_GUs_in_range=20` 条记录并 padding；
5. 同一有序记录同时用于本地状态和 Type-S packet。

packet 包含发送 UAV header、位置、覆盖计数，以及每个槽位的 session ID、MD 数值字段和有效位。session ID 只用于身份匹配，不作为 GRU 或 critic 的连续输入。

成功接收必须满足：解码后的数值块与发送端 canonical critic block 逐元素一致。

## 2. 丢包后的 receiver memory

在线 memory 使用固定容量表：

```text
(environment, receiver UAV, MD session ID) -> hidden/state memory
```

同一 MD 即使同时出现在多个成功 packet 中，也只在接收 UAV 的 memory 中保留一条。新 session 分配空槽，生命周期结束后释放，episode reset 时清空；容量由当前最大活动 MD 数 `n_GUs=60` 限制，不随历史累计 session ID 增长。

发送 UAV 的位置/header 也只在 packet 成功接收时更新。当前 packet 丢失时使用上次成功接收的发送方信息；从未成功接收过该发送方时不凭环境真值构造状态块。

## 3. 三种重建模式

- `zero`：原有零填充基准；不维护重建 memory。
- `last_obs`：保存最后一次成功观测到的持续特征，任务字段在当前不可观测时标为未知。
- `md_gru`：每个 receiver UAV 各有一套本地 GRU 参数；该 UAV 的所有 MD ID 共享这套参数，而每个 receiver/session 保留独立 hidden state。五架 UAV 的模型以相同随机初值启动，之后只用本 UAV 可获得的重观测样本更新，不交换预测器参数或训练样本。

MD-GRU 输入/目标为位置、速度、移动方向及参考方向的连续编码，并显式使用丢包 age。剩余生命周期按时隙确定性递减，不作为网络预测目标。当前任务 `D/C/deadline` 不从历史外推：若该 MD 没有通过任一当前成功 packet 被接收，则使用占位值并令 `task_valid=0`。

重建后重新按发送 UAV 的覆盖关系和正式排序协议构造最多 20 条记录。未知任务记录在 completion-priority 模式中排在已知任务记录之后；距离使用发送 UAV 的最后已知位置。

## 4. MD-GRU 训练约束

训练样本只在一个已知 session 后续再次成功接收时产生：历史接收上下文作为输入，重新接收到的真实持续特征作为监督目标。丢包时不会读取模拟器真值来更新 hidden state 或产生标签。

在线 hidden/memory 与环境 episode 对齐：每个 episode reset 时清空，以免把不同 episode 的 MD session 串接为一条轨迹。监督训练 replay 与在线 hidden 不同：它保存每次重观测前的原始短观测序列、预测 age 和重观测真值，在 episode 之间保留，并用有界 FIFO 控制内存。

rollout/episode 边界对齐后，每个有新增重观测标签的 UAV 从自己的 replay 训练自己的 GRU；没有新样本时跳过更新。训练时从零 hidden 用当前 GRU 参数重新展开原始序列，避免旧模型生成的 stale hidden 被当作固定上下文。replay 容量由 `md_gru_max_samples=32768` 控制，但每次只抽 `md_gru_train_samples=512` 条，避免把容量误当成每轮训练预算；达到 `md_gru_min_ready_samples=512` 前继续使用 last-observation fallback，不让一两个样本训练出的预测器直接污染 critic。预测采用独立 optimizer 和 Smooth-L1 监督损失；PPO 的 actor/critic loss 不反传进 GRU，GRU 输出在 rollout 构造 critic 输入时已经 detached。checkpoint 文件仍叫 `md_gru_shared.pt` 以兼容现有 manifest，但文件内部包含五套 receiver-local 模型和 optimizer；旧版单模型 checkpoint 加载时会复制为五套相同的本地初始化。正式训练目前使用 separated MEC runner；Windows 入口中的旧参数 `--share_policy` 正是选择该 runner，脚本会自动传入。

## 5. Windows 启动入口

零填充不可靠数据面：

```powershell
.\onpolicy\scripts\train\run_dynamic_5uav.ps1 `
  -Method dcppo -CommunicationMode unreliable `
  -StateReconstruction zero -ExperimentName unreliable_zero
```

带相同 critic metadata 的公平零填充消融：

```powershell
.\onpolicy\scripts\train\run_dynamic_5uav.ps1 `
  -Method dcppo -CommunicationMode unreliable `
  -StateReconstruction zero -CriticMDMetadata `
  -ExperimentName unreliable_zero_metadata
```

Last-observation 与 MD-GRU：

```powershell
.\onpolicy\scripts\train\run_dynamic_5uav.ps1 `
  -Method dcppo -CommunicationMode unreliable `
  -StateReconstruction last_obs -ExperimentName unreliable_last_obs

.\onpolicy\scripts\train\run_dynamic_5uav.ps1 `
  -Method dcppo -CommunicationMode unreliable `
  -StateReconstruction md_gru -ExperimentName unreliable_md_gru
```

未显式指定时，脚本对可靠通信使用 260 m，对不可靠通信使用 520 m。Type-S 的结构化最小 bit 数由环境计算；若 `StatePayloadBits` 小于该值，初始化会直接报错，避免以不可能容纳 packet 的配置启动实验。

## 6. 当前自动验收

`tests/test_md_state_reconstruction.py` 覆盖：成功 packet 逐元素等价、排序后截断、丢包无当前真值泄漏、ID 去重、生命周期释放、仅重新观测产生 GRU 标签、receiver-local GRU 可训练以及速度编码覆盖正式 `[0,5] m/s` 范围。完整测试命令为：

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
C:\Users\wyj2\.conda\envs\marl\python.exe -m pytest tests -q
```

## 7. 2026-08-13：同步速度分支后的基线

本项目不再以 2026-08-01 的早期 MEC 合同作为主干，而是以速度/Actor-message 分支归档提交 `fa53f3a` 为功能基线，再正交叠加 Type-S 不可靠 critic 数据面、Type-A 不可靠 advantage 数据面和 receiver-local MD-GRU。同步后的正式 Fixed600 配置为：

| 类别 | 值 |
|---|---|
| 地图/热点 | `600×600 m`，`episode_template4_600_200` index 0 |
| 动态 MD | 每 slot 严格 `1+4`，lifetime `12`，容量 `60` |
| MD 移动 | 均速 `3 m/s`，初始化 std `0.6`、倍率 `[0.4,1.6]`，更新裁剪 `[0,5]` |
| UAV | 5 架，固定 line 起点，`v_max=30 m/s`，`Delta_t=0.5 s` |
| Actor | Spatial Cartesian；completion-priority roster；layout context `meters_v2` |
| Critic | ego-query attention；R520；共享 return normalization |
| Advantage | `per_agent_noise`，`noise_scale=3`；不可靠模式用 running-sum ratio consensus |
| 通信 | R520；Type-S `8000 bit / 13.54 ms`；Type-A `16000 bit / 21.54 ms`；50 rounds |

通用入口 `run_dynamic_5uav.ps1` 现在显式接受 `EpisodeLength` 和 `MDLifetime`，不再把 speed 合同需要的 MD12 隐藏在另一套脚本中。正式入口为：

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  .\onpolicy\scripts\train\run_fixed600_200_unreliable_dataplane.ps1 `
  -StateReconstruction md_gru -Seed 2
```

该入口默认 `ActorMessageMode=disabled`。原因不是认定 Actor message 永远无效，而是当前 `task_summary` 只有距离门控，尚未经过不可靠信道；默认关闭可避免给“不可靠通信 + GRU”实验留下可靠 Actor 旁路。若要做与速度分支逐参数一致的兼容对照，可显式使用：

```powershell
...run_fixed600_200_unreliable_dataplane.ps1 `
  -StateReconstruction md_gru -ActorMessageMode task_summary
```

此时 Actor message 保持速度分支的 `receiver_gated_sum + absolute_raw_v2`，但必须标注为“Type-S/Type-A 不可靠，Actor message 仅距离门控”，不能写成全链路不可靠。

## 8. 同步后的 Actor / Critic 输入合同

- no-message Actor：`211` 维，即 `self xy(2) + layout(8) + count(1) + 20×MD(10)`。
- message-capable Actor：`251` 维，在上述输入中加入 `4×10=40` 维 task-summary packet。
- Critic 不读取 Actor-only 的 40 维消息。每个 UAV 的速度基线 token 为 `211` 维。
- 公平的不可靠通信消融统一开启 `critic_md_metadata`，每个 token 追加 `20×(record_valid, task_valid, age)=60` 维，得到 `271` 维；5 个 ID 对齐 token 拼接后为 `1355` 维。
- Type-S 成功接收时，其 decoded token 与 canonical critic token 逐元素一致；失败时只替换对应 sender token，不修改 Actor observation。
- 8 维 hotspot layout 是 episode 开始时的共同先验，不计入 Type-S payload；sender UAV 位置由低速可靠控制面提供，GRU 只重建/预测 MD 记录。

环境与重建器现在共用 `md_roster.py` 的稳定排序：completion-priority 或 distance-only、距离、session ID。这样真实 token 与重建 token 不会因并列距离或动态槽位复用产生不同顺序。

## 9. Advantage 与兼容边界

速度分支的 `local`、`mixed_consensus`、`pure_consensus`、`legacy_noise`、`per_agent_noise` 均保留。在 `communication_mode=unreliable` 下，Type-A 用独立随机流采样及时接收并执行 running-sum ratio consensus；`per_agent_noise` 仍是已跑速度合同对应模式。其训练优势为

```text
A_train_i = A_local_i + noise_scale * residual_i * std_agents(A_local) * epsilon_i
residual_i = |A_running_sum_i - mean(A_local)| / |A_local_i - mean(A_local)|
epsilon_i ~ N(0, 1)
```

因此 running-sum 决定的是拓扑/丢包相关残差，不单独决定最终噪声样本；`noise_scale`、当前样本的跨 UAV 标准差以及独立 Gaussian draw 同样参与。若某个 `A_local_i` 恰等于精确均值，则只有 `A_running_sum_i` 也仍等于均值时残差为 0；如果不完整通信把它推离均值，零分母端点定义为 1，避免漏掉真实共识误差。

`per_agent_noise` 还使用五架 UAV 的 local advantage 直接计算精确 `mean(A_local)`，所以它是共识误差驱动的仿真 surrogate，而不是仅依赖每架 UAV 本地 running-sum 状态的完全分散算法。若论文要求不访问这个全局 oracle，应使用 `pure_consensus` 或另行定义本地 residual estimator。mode 写入 buffer 后，PPO 仍会对每架 UAV 自己的时间样本做 advantage 标准化与 `[-10,10]` 裁剪；reliable/unreliable 两条路径的后处理相同。

论文 IV-D 当前文字和伪代码描述的是 `A_train_i=A_running_sum_i`，即 `pure_consensus`。要复现实验归档中的速度算法使用默认 `-AdvantageMode per_agent_noise -NoiseScale 3`；要严格复现论文伪代码使用 `-AdvantageMode pure_consensus`，此时 `NoiseScale` 不参与计算。固定入口已经开放 `AdvantageMode`、`NoiseScale`、`CommunicationDistance` 和 `RunningSumRounds`，避免这些方法选择继续被脚本写死。

`externality_consensus` 在不可靠模式下会明确报错。它依赖可靠有限轮 Metropolis 图的 component size 和 self-contribution coefficient，不能把 running-sum 输出直接代入而声称算法等价。

MD-GRU 只在 episode/rollout 边界对齐时更新。`md_gru_shared.pt` 连同五套 receiver-local optimizer 和 `predictor_ready` 一起进入 checkpoint manifest；`zero` 和 `last_obs` 不创建伪 GRU 文件。评估快照只接受这一种已知可选文件，其他未知 checkpoint 文件仍会被拒绝。

## 10. 2026-08-13：论文 IV-D 与训练闭环复核

论文 IV-D 的核心状态结构是正确的：每个 UAV 使用一套参数、其维护的每个 MD ID 具有独立 hidden/age/lifetime。GRU 只用于丢失 Type-S 时构造 surrogate neighbor state，该状态仅进入 critic；actor 和 MD 侧关联保持不变。

本轮复核修正了两处旧实现偏差：旧代码跨 UAV 共用单一预测器，且每轮清空 reservoir 并用旧参数生成的 hidden 只做一步反传；现在改为 receiver-local predictor + 跨 episode 的原始短序列 replay。论文的总损失 `L_V + lambda_p L_pred` 在参数集合互不共享且重构状态已 detached 时只是两个独立优化问题；代码因此明确使用两个 optimizer，不再用 `lambda_p=0.1` 对 Adam 梯度作没有清晰目标权衡意义的缩放。若论文保留该公式，应说明优化是交替进行且两项之间没有交叉梯度。

速度归一化也已与正式环境合同对齐：上界取 `max(1.3×mean_velocity, init_max_factor×mean_velocity, update_clip_max)`。正式参数 `mean_velocity=3`、`init_max_factor=1.6`、`update_clip_max=5` 因而使用 `5 m/s`，不会再把合法的 `3.9--5 m/s` 速度错误裁成 `3.9 m/s`。

每轮更新顺序为 `compute returns/advantages -> PPO actor/critic update -> predictor supervised update -> next rollout`。这样同一 on-policy batch 的 estimator 版本保持冻结，更新后的 GRU 只服务下一轮 critic state 构造。

CPU-only 启动可显式传 `-CPUOnly`；该开关会把 Python 入口的 legacy `--cuda` store-false 标志传下去。当前 Windows `marl` 环境的 `torch 2.12.0+cu126` 在 `CUDA_VISIBLE_DEVICES=-1` 下对最小 GRU backward 已能产生有限梯度，但进程退出/optimizer 初始化附近会出现原生 `0xC0000005`；这是独立于项目代码可复现的 CPU wheel/runtime 问题。因此本机只把 CPU 测试作为逻辑 smoke，正式 GRU 训练前需要在隔离环境安装稳定 CPU wheel，或等现有 GPU 训练结束后再做完整训练验证。

## 11. 验收矩阵

最初速度分支同步完成时通过（早于本节之后的 receiver-local raw-sequence replay 重构）：

- 全部单元/回归测试：`95 passed`；
- reliable + zero；
- unreliable + zero；
- unreliable + last_obs；
- unreliable + md_gru。

四个端到端 smoke 均使用 Fixed600-200、MD12、v30、R520、Spatial Actor、completion-priority、layout meters、per-agent-noise=3，并完成一次环境 rollout、PPO 更新和 checkpoint 发布。MD-GRU 清单额外绑定 `md_gru_shared.pt`；另外三组不绑定该文件。

当前 receiver-local GRU/统计/eval 修改后的本轮验收采用 CPU-only 隔离测试：MD reconstruction/GRU 17 项、reliable consensus 11 项、unreliable communication 8 项、eval/normalization 2 项断言通过。由于本机 PyTorch CPU runtime 仍可在较长进程中触发 `0xC0000005`，本轮没有把历史四个 smoke 当作当前代码的完整 PPO 复验；正式训练前仍需在稳定 CPU wheel 或空闲 GPU 上补 one-update smoke。

## 12. 旧分支思路如何迁移

| 2026-08-01 早期实现/假设 | 当前处理 |
|---|---|
| 直接在早期 MEC 环境上叠加 Type-S/GRU | 放弃早期环境主干；以 `fa53f3a` 的速度环境和输入合同为基线 |
| MD10、旧热点/出生过程、较低移动速度 | 换为 Fixed600-200、严格 1+4、MD12、MD 3m/s 与 `[0,5]` 更新裁剪 |
| Actor/Critic 共用旧局部观测结构 | Actor 完全沿用速度分支；Type-S 只操作去掉 Actor-only message 后的 critic token |
| 环境和重建器分别实现 MD 排序 | 合并为共享 `md_roster.py`，用 session ID 做稳定 tie-breaker |
| 不可靠 advantage 仍围绕旧 exact-mean/legacy-noise 逻辑 | 接入当前统一 `advantage_mode`；正式合同使用 `per_agent_noise=3` 与 running-sum estimate |
| 状态通信、advantage noise 与环境可能共享随机源 | 分离为独立 RNG，通信开关不改变 Actor/MD 环境随机流 |
| GRU 文件独立保存，不受速度分支 manifest 约束 | `md_gru_shared.pt` 进入 manifest 和评估快照一致性校验 |
| `last_obs` 和 `md_gru` 都被笼统当作“有重建器” | 只有 `md_gru` 发布模型文件；`last_obs` 只维护在线 memory |
| Actor task-summary 默认随不可靠实验一起开启 | 正式不可靠入口默认关闭，避免当前未建模丢包的可靠旁路；显式开启仅作兼容对照 |

没有迁移旧分支中“无条件把 Gaussian-Markov MD 速度裁剪到均速的 `[0.7,1.3]` 倍”这一修改，因为速度分支已经把移动初始化和每步更新裁剪拆成可配置参数；当前正式速度合同使用初始化倍率 `[0.4,1.6]` 和绝对更新范围 `[0,5] m/s`。这比硬编码旧范围更符合实际已跑速度实验。

## 13. 2026-08-19：可靠主线增量兼容层

本项目现已继续吸收 `agent/dcppo-runtime-optimization` 在 8 月 14—19 日新增的训练语义，同时保留本章第 7—10 节定义的 Type-S、Type-A 和 GRU 合同：

- `--association_threshold`：默认 `0.5`，统一控制 actor available-action mask 和环境 association 执行；
- `--disable_offload_deadline_filter`：可关闭执行前 deadline 可行性过滤，执行后的真实 deadline/reward 判断仍保留；
- `--uav_resource_mode homogeneous|heterogeneous` 与 `--uav_resource_scale_factors`：支持每 UAV 带宽/算力异构，scale 必须为正且总和等于 UAV 数；
- `--n_UAVs`、显式 `--uav_start_positions` 和 checkpoint/render agent count 已通用化；Type-S/GRU 继续按 receiver 数创建本地 predictor/bank；
- `run_fixed600_200_unreliable_dataplane.ps1` 暴露上述参数；新增 `run_fixed600_200_6uav_unreliable_dataplane.ps1` 作为六 UAV 不可靠 + GRU 入口。

默认参数保持五 UAV 旧行为，因此已有五 UAV homogeneous checkpoint 可以继续恢复；六 UAV 会改变策略数量和 critic 输入，不允许从五 UAV checkpoint 静默 warm-start。可靠主线的 hybrid checkpoint actor 仅为离线评估工具，没有进入训练调用链，本轮没有接入算法。

本轮在不使用 GPU 的情况下完成 `107 passed, 2 warnings`；四个 launcher 通过 PowerShell 语法检查，五/六 UAV 最终 argv probe 通过。详细提交映射、排除项和实验边界见 `UNRELIABLE_SPEED_COMPATIBILITY_AUDIT_20260813.md` 第 13 节。

## 14. 2026-08-19：可调物理范围图与 natural-only GRU 合同

本轮修复了 Type-A 仍隐含完全图的根本错误。Type-S 和 Type-A 现在都使用同一 `neighbor_distance`（别名 `--d_com`）构造名义范围图；范围外链路不采样为可用包，也不计入收包率。Type-A 在 rollout 结束、reset 之前保存的终局 UAV 位置上固定该图，running-sum 每个 sender 按其实际 `out_degree+1` 分配质量。连通图收敛到全局均值，断连图分别收敛到各 component 均值，孤立 UAV 保持 local advantage；不再添加 `|C_i|/M` 缩放。收包率分母严格为 `H × 名义有向边数`，无边时定义为 0。

物理参数只有一个真源：

- 两项都不传：`d_com=520 m`，由链路预算得到 `P_c=1.1809658836 W`；
- 只传 `--d_com/--neighbor_distance`：反算 `P_c`；
- 只传 `--a2a_transmit_power_w`：反算 `d_com`；
- 两项都传：若功率推导距离和输入距离相差超过 `5 m`，启动立即失败；
- 默认 `K_c=10 dB`、running-sum `H=50`；`d_com=0,P_c=0` 只保留为严格无边消融。

Rician `K_c` 控制范围内衰落分布，不进入 nominal radius 的确定性链路预算；因此调通信半径时以 `P_c <-> d_com` 为主，`K_c` 作为单独可靠性参数。`task_summary` Actor-message 的实际几何门控也读取解析后的 `neighbor_distance/d_com`，所以 power-only 时会自动使用反算出的半径；启动器中的 `ActorNeighborDistance` 对应旧 `neighbor_R/actor_neighbor_obs` 路径，不会覆盖 task-summary 的物理范围。

MD-GRU 默认继续只使用真实信道产生的自然重观测标签，不制造人工缺失。连续成功接收产生 age=1，一次缺失后重观测产生 age=2，以此类推；age=1 多本身符合实际首次缺包风险，不能直接判为训练偏差。代码新增 replay-label age 与 rollout-query age 的 1/2/3/4+ 统计，并在同一真实重观测事件上比较 GRU 与 last-observation 的 prequential 位置 RMSE。只有日志证明某个实际常用 query-age 桶缺少自然标签，才把信道一致的数据增强作为独立消融，不改变默认论文口径。

预测器保持独立 Smooth-L1、独立 Adam 和梯度裁剪：预测损失只更新 receiver-local GRU/head；重构后的 NumPy critic state 不携带计算图，PPO value/actor 梯度不能进入预测器。online hidden 按 episode/session 清空，跨 episode replay 保留；checkpoint 恢复模型、optimizer 和 ready 状态，但 replay 不保存，因此是 warm-start 而非 bitwise exact resume。

本轮在 `CUDA_VISIBLE_DEVICES=''` 的 CPU-only 环境通过通信、running-sum、per-agent residual、MD reconstruction/GRU 的 `48 passed` 定向回归，并以工作区临时目录完成全套 `117 passed, 2 warnings`；两条 warning 是已有的空统计除法。PowerShell 三个入口语法通过，5-UAV power-only argv probe 确认只下传功率、由 Python 统一反算距离。未启动长训练或任何 GPU 工作负载。

在确认约 `39.56 GB` 空闲内存、无 Python 训练进程后，又执行了正式入口的 2-slot/1-worker CPU-only 闭环：`codex_cpu_smoke_rangegraph_gru_20260819/run1` 完成 rollout、Type-A、PPO、predictor 边界处理和 checkpoint 发布。`args.json` 记录 `cuda=false/device=cpu`、`d_com=520`、`P_c=1.1809658836179866`、`K_c=10`、`H=50`、`per_agent_noise`、`md_gru`；manifest 同时绑定五套 actor/critic/normer 和 `md_gru_shared.pt`。该 smoke 只验证执行闭环，不用于性能结论。
