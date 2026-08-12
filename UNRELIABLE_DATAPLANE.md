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
