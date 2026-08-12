# Type-S 不可靠数据面与 MD 状态重建契约

本文档说明当前代码闭环。优势函数闭环保持原设计，不由本模块改变：通信估计只控制随机扰动幅度，确定性主体仍为 local advantage 与全 UAV 精确均值之和。

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
- `md_gru`：使用一个由所有 UAV 和所有 MD 共享参数的 GRU 预测持续移动特征；每个 receiver/session 保留独立 hidden state。

MD-GRU 输入/目标为位置、速度、移动方向及参考方向的连续编码，并显式使用丢包 age。剩余生命周期按时隙确定性递减，不作为网络预测目标。当前任务 `D/C/deadline` 不从历史外推：若该 MD 没有通过任一当前成功 packet 被接收，则使用占位值并令 `task_valid=0`。

重建后重新按发送 UAV 的覆盖关系和正式排序协议构造最多 20 条记录。未知任务记录在 completion-priority 模式中排在已知任务记录之后；距离使用发送 UAV 的最后已知位置。

## 4. MD-GRU 训练约束

训练样本只在一个已知 session 后续再次成功接收时产生：历史接收上下文作为输入，重新接收到的真实持续特征作为监督目标。丢包时不会读取模拟器真值来更新 hidden state 或产生标签。

样本使用有容量上限的 reservoir，rollout/episode 边界对齐后训练共享 GRU。GRU checkpoint 为 `md_gru_shared.pt`。正式训练目前使用 separated MEC runner；Windows 入口中的旧参数 `--share_policy` 正是选择该 runner，脚本会自动传入。

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

`tests/test_md_state_reconstruction.py` 覆盖：成功 packet 逐元素等价、排序后截断、丢包无当前真值泄漏、ID 去重、生命周期释放、仅重新观测产生 GRU 标签以及共享 GRU 可训练。完整测试命令为：

```powershell
C:\Users\wyj2\.conda\envs\marl\python.exe -m pytest tests -q
```

## 7. 2026-08-13：同步速度分支后的基线

本项目不再以 2026-08-01 的早期 MEC 合同作为主干，而是以速度/Actor-message 分支归档提交 `fa53f3a` 为功能基线，再正交叠加 Type-S 不可靠 critic 数据面、Type-A 不可靠 advantage 数据面和共享 MD-GRU。同步后的正式 Fixed600 配置为：

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

速度分支的 `local`、`mixed_consensus`、`pure_consensus`、`legacy_noise`、`per_agent_noise` 均保留。在 `communication_mode=unreliable` 下，Type-A 用独立随机流采样及时接收并执行 running-sum ratio consensus；`per_agent_noise` 仍是最终速度合同对应模式。

`externality_consensus` 在不可靠模式下会明确报错。它依赖可靠有限轮 Metropolis 图的 component size 和 self-contribution coefficient，不能把 running-sum 输出直接代入而声称算法等价。

MD-GRU 只在 episode/rollout 边界对齐时更新。`md_gru_shared.pt` 连同 optimizer 和 `predictor_ready` 一起进入 checkpoint manifest；`zero` 和 `last_obs` 不创建伪 GRU 文件。评估快照只接受这一种已知可选文件，其他未知 checkpoint 文件仍会被拒绝。

## 10. 验收矩阵

同步后通过：

- 全部单元/回归测试：`95 passed`；
- reliable + zero；
- unreliable + zero；
- unreliable + last_obs；
- unreliable + md_gru。

四个端到端 smoke 均使用 Fixed600-200、MD12、v30、R520、Spatial Actor、completion-priority、layout meters、per-agent-noise=3，并完成一次环境 rollout、PPO 更新和 checkpoint 发布。MD-GRU 清单额外绑定 `md_gru_shared.pt`；另外三组不绑定该文件。

## 11. 旧分支思路如何迁移

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
