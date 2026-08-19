# 不可靠数据面与速度/Actor-message 基线兼容性审计

日期：2026-08-13  
审计对象：

- `D:\wyj\Projects\on-policy-unreliable-dataplane`，分支 `feature/unreliable-dataplane`；
- `D:\wyj\Projects\on-policy-last-obs-clean-5UAV`，分支 `agent/dcppo-runtime-optimization`；
- 速度/Actor-message 归档基线提交 `fa53f3a`；
- 不可靠项目同步提交 `9ceaffd` 及其当前未提交的 GRU 训练完善。

## 1. 总结论

当前不可靠项目已经以速度分支 `fa53f3a` 为环境和 DC-PPO 主合同，而不是继续运行 2026-08-01 的早期环境。Fixed600-200、动态 MD、v30、Spatial Cartesian Actor、completion-priority roster、layout context、ego-query critic、PPO 超参数和归一化逻辑兼容。

不可靠项目正交增加了三部分：

1. Type-S：每时隙不可靠 critic 状态 packet；
2. Type-A：episode 末基于 running-sum ratio consensus 的优势交换；
3. `zero/last_obs/md_gru`：只服务 critic 的状态缺失处理。

需要避免三个过度结论：

- 默认 `per_agent_noise` 不是论文 IV-D 中“直接用 running-sum 输出”的算法；后者是 `pure_consensus`。
- `neighbor_distance=520` 不是 Type-A 物理信道的硬截断；它是 reliable Metropolis、Actor-message 几何门控和 critic 邻域的半径。
- Actor-message 在不可靠项目中仍只有几何门控，没有经过当前 Type-S/Type-A 丢包模型；开启它时只能称为兼容对照，不能称为“所有通信均不可靠”。

## 2. 两个项目的继承关系

早期不可靠功能提交为：

| 日期 | 提交 | 内容 |
|---|---|---|
| 2026-07-31 | `b265913` | finite-round reliable consensus |
| 2026-07-31 | `8062245` | unreliable running-sum advantage data plane |
| 2026-08-01 | `4507615` | per-slot unreliable Type-S packets |
| 2026-08-01 | `483db29` | causal MD state reconstruction/GRU |

此后速度主线继续加入 advantage modes、Actor-message、receiver-gated pooling、输入合同、Random12 和速度实验。2026-08-13 的 `91e8623`/`9ceaffd` 已把 `fa53f3a` 合并为不可靠项目的主基线，再叠加 Type-S/Type-A/GRU。由此，当前审计不能把 8 月 1 日的旧脚本默认值当作现行合同。

## 3. 真实速度实验合同

以下来自已实际运行的 `vmax30_seed32.args.json`，不是仅根据 parser 默认值推测：

| 类别 | 正式值 |
|---|---|
| runner | `algorithm_name=mappo`，`share_policy=false`，实际进入 separated MEC runner |
| UAV/MD | 5 UAV，60 个动态 MD 槽；每时隙严格 5 个候选，区域 `1+4` |
| 地图/布局 | 600×600 m，`episode_template4_600_200`，固定 layout index 0 |
| UAV 起点 | `(110,180),(220,180),(330,180),(440,180),(400,400)` |
| MD lifetime | 12 slots |
| UAV 速度 | 比较 10/20/30/40；当前选定 `v_max=30 m/s` |
| MD 速度 | mean 3，init std 0.6，init factor `[0.4,1.6]`，每步 clip `[0,5] m/s` |
| Actor | Cartesian + Spatial flight；completion-priority；layout `meters_v2` |
| Actor message | `task_summary + receiver_gated_sum + absolute_raw_v2` |
| Critic | attention + ego-query；R520；不使用 actor-only message block |
| advantage | `per_agent_noise`，`noise_scale=3`，reliable `n_iterations=50` |
| PPO | hidden 256，2 layers，lr `1e-4`，critic lr `5e-4`，epoch 4，clip 0.15，mini-batch 1，entropy 0 |
| normalization | observation norm；return norm；shared return scale；不使用 ValueNorm |
| budget | episode 400，64 rollout envs，60M steps，training thread 1 |

“命令名写 mappo”不等于算法只是原始 MAPPO：当前项目通过 `advantage_mode`、通信和 separated runner 实现 DC-PPO 变体。`--share_policy` 是旧式 `store_false` 参数；脚本传入它后 `share_policy=false`，选择五套 separated policy/trainer。

## 4. 环境和输入兼容矩阵

| 合同 | 速度项目 | 不可靠项目 | 结论 |
|---|---|---|---|
| Fixed600-200 / strict 1+4 | 是 | 是 | 兼容 |
| MD12 / MD 3m/s / clip `[0,5]` | 是 | 是 | 兼容 |
| 固定五 UAV 起点 | args 显式传入 | 固定入口也显式传入同一组起点 | 参数与数值均兼容 |
| v30 / Cartesian / Spatial | 是 | 是 | 兼容 |
| completion-priority roster | 是 | 环境与重建器共用 `md_roster.py` | 兼容且消除 tie-order 漂移 |
| Actor no-message | 211 维 | 211 维 | 兼容 |
| Actor task-summary | 251 维 | 251 维 | 输入兼容；不可靠项目尚未给消息加丢包 |
| reliable critic | `5×211=1055` | reliable 模式同值 | 兼容 |
| unreliable critic | 无 | metadata 后 `5×271=1355` | 有意扩展；旧 critic checkpoint 不能直接加载 |
| PPO/归一化 | 速度 args | 固定入口同值 | 兼容 |

Actor-message 的 40 维是 `4 sender × 10 values`，只进入 Actor。critic 通过 `_get_critic_local_obs` 排除 Actor-only block，所以开关 Actor-message 不改变可靠 critic token，也不会直接改变 Type-S schema。

## 5. `neighbor_distance`、`neighbor_R` 和物理信道

| 参数 | 真正用途 |
|---|---|
| `neighbor_distance` | Actor-message 几何 gate；critic 的 sender/receiver 几何邻域；reliable Metropolis 终局图 |
| `neighbor_R` | 仅旧 `actor_neighbor_obs` 输入块；正式 Actor-message 不依赖它 |
| Type-S physical channel args | 每时隙状态 packet 是否在 deadline 内成功 |
| Type-A physical channel args | 每轮 running-sum 累积量是否及时到达 |

Type-A 的每个其他 UAV 都是潜在 receiver，距离通过 path loss/SNR 改变及时接收概率；代码没有先用 R520 将 520 m 外链路硬删除。这与论文 Type-A “每个其他 UAV 都可接收，但信道随机失败”的描述一致。

固定入口现在开放：

```powershell
-CommunicationDistance 520
-RunningSumRounds 50
-ActorMessageMode disabled   # 或 task_summary
```

若研究物理丢包强度，应优先调 payload、deadline、发射功率、带宽和噪声等 channel 参数；把 R 当作 Type-A packet-loss 半径会改变算法语义。

## 6. 优势函数的真实数据流

### 6.1 终局位置是否正确

正确。数据顺序是：

1. 最后一个 slot 的 `MEC.step()` 在 `info['uav_positions']` 中保存 reset 前的 UAV 位置；
2. vector wrapper 发现 done 后可以自动 reset obs/state，但原来的 `info` 不被替换；
3. runner 在 rollout 完成后从 `infos` 复制终局位置到 `self.uav_positions`；
4. reliable consensus 和 unreliable running-sum 都读取这份 `self.uav_positions`。

因此优势通信没有误用下一 episode reset 后的位置。buffer 最后槽的 Metropolis 图可能属于 reset 后环境，所以 reliable 路径也没有依赖它，而是由终局位置和 `neighbor_distance` 重新建图。

### 6.2 reliable 和 unreliable 是否“同一种优势处理”

外层 advantage mode 相同，通信估计器不同：

```text
local advantages A_i
  reliable   -> K-round Metropolis estimate C_i
  unreliable -> H-round running-sum ratio estimate C_i
advantage_mode(A_i, C_i) -> actor PPO training advantages
```

对速度合同 `per_agent_noise`：

```text
r_i = clip(|C_i - mean_j A_j| / |A_i - mean_j A_j|, 0, 1)
A_train_i = A_i + noise_scale * r_i * std_j(A_j) * epsilon_i
epsilon_i ~ Normal(0,1)
```

所以：

- `noise_scale=3` 是最大倍率，不是每个样本的实际标准差；
- running-sum 决定 `r_i`；
- local advantage 的跨 UAV 标准差和 Gaussian draw 也决定实际噪声；
- 无法说“最终噪声完全由 running-sum 决定”。

还有一个方法边界必须明确：`r_i` 的分子和分母都使用了 `mean_j A_j`，当前 runner 是直接从五架 UAV 的 local advantage 计算这个**全局精确均值**。因此 `per_agent_noise` 是用真实共识残差控制噪声的仿真 surrogate/诊断模型，但它不是仅凭每架 UAV 本地 running-sum 状态即可实现的完全分散算法。若论文主张不依赖全局 oracle，应使用 `pure_consensus`，或重新设计一种只依赖本地可观测量的 residual estimator。

上述 mode 选出 `A_train_i` 后，`R_MAPPO` 还会对每架 UAV 各自 buffer 中的 advantage 做均值/标准差归一化，并裁剪到 `[-10,10]` 再进入 clipped PPO objective。这一步和 reliable/unreliable 共用，且不会跨 UAV 重新求精确均值；它意味着 consensus/noise 首先改变的是每架 UAV advantage 的时间形状与相对排序，绝对幅值随后会被 agent-wise normalization 消去一部分。

本次修复了一个零分母边界：若某 UAV 的 local advantage 已经等于精确均值，它的初始共识误差为零，残差应为 0。旧代码返回 1，会给已处于正确均值的 UAV 错误注入满幅噪声。

### 6.3 与论文 IV-D 的差异

论文当前伪代码直接使用 running-sum 输出作为 advantage：

```text
A_train_i = C_i
```

它对应：

```powershell
-AdvantageMode pure_consensus
```

速度实验归档对应：

```powershell
-AdvantageMode per_agent_noise -NoiseScale 3
```

固定入口已经允许二者选择。正式论文实验必须二选一并在方法、图例和 args 中保持一致；不能用 `per_agent_noise` 结果支撑论文的 pure running-sum 公式而不说明 surrogate 设计变化。

## 7. Actor-message 对照是否存在、是否同合同

存在。2026-08-10 至 08-11 的 R520 no-message 与 actor-message/v30/seed2 逐字段比较，除实验名和 `actor_message_mode` 外一致：Fixed600-200、MD12、v30、R520、noise3、PPO、critic、归一化和预算相同。

共同 40.8832M 时：

| 指标 | no message | Actor message | 差值 |
|---|---:|---:|---:|
| tail25 True60 | 547,509 | 547,601 | +0.017% |
| tail100 True60 | 548,930 | 551,410 | +0.452% |

这支持“当时单 seed 的训练表现近似”，不支持“Actor-message 已被证明无效”：no-message 冻结快照没到 60M，且不是同 checkpoint 的测试时屏蔽反事实。

`disabled` 真正做的是移除 Actor 的 40 维消息块及 message encoder/gate 参数；不是把 40 维保留后置零。critic 通信、R520 和 advantage consensus 仍正常运行。

## 8. GRU/last-observation 消融

| 模式 | 丢失 Type-S 时的处理 | 是否有训练网络 |
|---|---|---|
| `zero` | 对丢失 sender token 零填充 | 否 |
| `last_obs` | 使用该 receiver 最后成功接收的 persistent MD feature | 否 |
| `md_gru` | 训练前回退 last-observation；ready 后由 hidden + age 预测 | 是 |

公平消融统一启用 `critic_md_metadata`，保证三种模式都是 1355 维，避免“方法差异”混入 critic 参数量/输入维度差异。

每架 receiver UAV 有一套本地 GRU 参数和 optimizer。该 UAV 所见的所有 MD session 共享参数，但每个 `(env, receiver, session_id)` 有独立 hidden、age、lifetime、last feature 和 task-valid 状态。这是“每个 MD 一个 hidden state”，不是“每个 MD 一张网络”。

在线 belief memory 在 episode 结束清空；监督 sequence replay 不清空，使用每 UAV 各自的有界 FIFO。只有某 session 之后重新成功观测时才产生标签，连续丢包期间不读取模拟器真值。

## 9. GRU 损失和梯度流

当前梯度合同是：

```text
receiver-local raw sequence replay
  -> GRU + prediction head
  -> Smooth-L1 L_pred
  -> that receiver's GRU optimizer only

detached reconstructed NumPy state
  -> critic value network
  -> PPO value loss
  -> critic optimizer only

local/consensus training advantage
  -> PPO clipped policy objective
  -> actor optimizer only
```

这条流是正确、可解释的 modular estimator 设计：predictor 在一个 rollout 内冻结，PPO 先用采样时固定的 critic states 更新，再在 episode 边界训练 predictor；更新后的 predictor 只影响下一 rollout。不会出现 PPO epoch 中途 state representation 漂移，也不会让 critic 为短期 value fitting 任意扭曲 mobility predictor。

论文若保留 `L_train=L_V+lambda L_pred`，应说明两项作用于不相交参数集、交替使用两个 optimizer；当前并不是真正端到端联合反传。端到端联合也可设计，但属于另一种算法，需要 differentiable state construction、共享 encoder 和专门的梯度权衡消融，不能由现代码的 `lambda` 参数假装实现。

监督目标是归一化后的 `x,y,speed,sin/cos(direction),sin/cos(reference)`，使用 Smooth-L1 和 gradient clip 1.0。任务 `D/C/deadline` 不从轨迹预测：不可观测时置 unknown 并给 critic `task_valid=0`。

本次修复了 GRU speed codec：旧上界固定为 `1.3×mean_velocity=3.9 m/s`，但正式环境允许 5 m/s，会在 decode 时把合法速度错误截断。新上界按完整环境参数取 5 m/s。

checkpoint 保存五套 predictor、optimizer 和 ready 状态，但**不保存监督 replay**。恢复训练后可以立即沿用已经 ready 的 predictor，后续重新观测样本再逐步填充 replay；这对“模型恢复”成立，但不是数据采样级的 bitwise exact resume。若实验要求故障后完全复现，需另外序列化 replay 和采样 RNG。

论文与现代码还有三处不能静默混用的超参数/目标差异：论文当前写 MSE 和 `lambda_p=0.1`，代码使用独立 optimizer 的 Smooth-L1，且 `md_prediction_loss_coef` 已是兼容保留参数、不会制造联合梯度；论文表中 running-sum `H=30`，当前速度兼容固定入口默认 `H=50`；论文伪代码直接使用 `C_i`，当前入口默认仍是 `per_agent_noise`。正式出图前必须把代码参数、论文表格和图例统一到同一合同。

## 10. 与成熟实现/论文的对照

- R2D2：replay 中旧 recurrent state 会随模型参数更新而 stale；当前保存原始短序列、训练时用当前参数从零 hidden 重放，符合其 sequence replay/burn-in 的核心原则。
- GRU-D：显式 mask 与 time interval；当前 `record_valid/task_valid/age` 同样把缺失模式显式交给 critic/predictor。
- DreamerV3：world model 从跨 episode replay 更新，actor/critic 与 model 有独立优化配置；说明模型 replay 不必跟 PPO on-policy buffer 一起清空。
- MAPPO 官方实现：rollout threads、episode length、PPO epoch、mini-batches、clip 等是复现实验的关键合同；本审计因此以真实 args 文件而非 parser 默认值作兼容判断。

这些工作只能证明当前设计是合理的成熟范式之一，不能证明它在本 MEC 任务上必然优于 last-observation。必须用 prediction metric 和 RL return 两层消融验证。

## 11. 本次代码修复

1. 固定启动器开放 `AdvantageMode/NoiseScale/CommunicationDistance/RunningSumRounds`；
2. Actor-message 即使 disabled，也把 pool/contract 显式写入 args，保证实验合同可审计；
3. 修正 per-agent consensus residual 的零分母端点；
4. 修正 GRU speed normalization 覆盖 `[0,5] m/s`；
5. 修复 separated runner 内置 `eval()` 对五返回值/八返回值、trainer list 和 per-agent action 的旧接口错误；
6. eval normalization 使用冻结统计，不污染训练 running mean/variance。
7. 固定入口显式传入五 UAV 起点和 `n_training_threads=1`，使生成的 args 与速度实验合同可逐字段审计，而不再依赖数值相同的环境内置 fallback。
8. 将 GRU replay 容量、每轮训练抽样量和 predictor ready warm-up 拆为 `32768/512/512`；每个 mini-batch 的可变长短序列改为 padding+mask 的 batched GRUCell 展开，避免 Python 逐样本反传成为正式 64-env 训练瓶颈。
9. 修正 Type-S reception-rate 口径：只统计 `neighbor_distance` 内实际进入 critic 数据面的有向链路，并在每个 episode reset 时重置计数；R 外物理采样不再被误报为 Type-S 尝试。

## 12. 当前限制与正式验证顺序

本地 GPU 已满且有现有训练运行，本次不启动 GPU 测试，也不停止任何进程。CPU 环境此前存在独立可复现的 PyTorch Windows native `0xC0000005`，因此轻量断言通过不能冒充完整 PPO one-update 已通过。

正式实验建议按以下顺序：

1. 在稳定 CPU wheel 或空闲 GPU 上跑全部 pytest 和 one-update smoke；
2. 固定相同 channel seeds，比较 `zero / last_obs / md_gru`；
3. prediction 先报告按 age 分桶的位置 RMSE、速度 MAE、方向误差，并加 constant-velocity baseline；
4. 再比较 RL return/value error；若 GRU 预测更准但 return 不提升，应报告 critic 对该误差不敏感；
5. advantage 方法单独比较 `per_agent_noise` 与 `pure_consensus`，不要同时改变重建器或 Actor-message；
6. Actor-message 的结论需至少多 seed 满预算，最好补同 checkpoint normal-vs-mask 的配对评估。

最终资源复核：约 33.91 GB 可用内存；GPU 已用 23205/24576 MiB，且本地三条训练主进程及其 64-worker 子进程仍存在。没有停止或修改任何现有进程，也没有启动 GPU/PPO one-update 测试。

CPU-only 验证结果：

- Python `py_compile`：通过；两个 PowerShell 启动器 AST parse：通过；`git diff --check`：通过。
- launcher contract probe 使用 `Write-Output` 代替 Python，可在不创建环境/训练进程的条件下捕获最终 argv：speed-compatible 和 paper-style pure-consensus 两种合同均通过；MD-GRU 正式入口共捕获 184 项参数，包含 replay/train/warm-up=`32768/512/512`，pure-consensus 不传无效的 `noise_scale`。
- MD reconstruction/GRU：17 项断言均通过，包含 Type-S canonical codec、无真值泄漏、receiver-local optimizer 隔离、跨 episode replay、minimum-ready warm-up、Actor 输入不变和 5 m/s speed round-trip。
- reliable consensus/terminal position：11 项断言通过。
- unreliable running-sum/physical channel/Type-S RNG isolation：8 项断言通过，包括 Type-S 统计只计本 episode 的几何邻居链路。
- separated eval + frozen normalization：2 项断言通过。

第一次把上述测试放进同一个较长 Python 进程时，先有 14 项明确 PASS，随后复现 Windows native `0xC0000005`。将剩余集合拆为短隔离进程并在断言后 `os._exit(0)`，所有目标断言均明确 `EXIT=0`。因此结论是“Python 数据流和梯度更新断言通过”；由于本机 PyTorch CPU runtime 的原生崩溃仍存在，不能升级为“完整 CPU PPO 训练已验收”。

## 13. 2026-08-19：同步可靠主线 8 月 14—19 日运行语义

### 13.1 主线身份与提交审计

`on-policy-last-obs-clean-5UAV` 的 `agent/dcppo-runtime-optimization` 仍可视为可靠通信、Actor-message 和速度实验的当前主线。该分支在 8 月 14—19 日的提交并不都改变训练语义：

| 提交 | 分类 | 对不可靠项目的处理 |
|---|---|---|
| `73845de` | 关联阈值与 offload deadline 预过滤开关 | 已移植到 parser、ACTLayer 和环境执行/奖励路径 |
| `080738c` | 阈值实验启动器 | 不逐个复制实验脚本；把可调参数接入通用和正式不可靠启动器 |
| `898bdb6` / `0bcb887` | 任意 UAV 数、显式起点优先、异构 UAV 资源、评估归档工具 | 训练语义和通用 checkpoint/render 部分已移植；历史分析产物未复制 |
| `81abc75` | pushed blob/换行规范化 | 无运行语义，不移植 |
| `4acded9` / `b810e5f` | 异构资源绘图输入参数化 | 仅画图，不影响环境或训练，不移植 |
| `6c14ac8` | 本地/GitHub 历史对齐 | 无运行语义，不移植 |

`hybrid_actor.py` 也没有进入可靠主线的训练 import graph，只被 `analysis/psi_hybrid_checkpoint_20260817/` 的 checkpoint 混合评估使用。因此本轮没有把它伪装成不可靠算法组成部分；若以后要复现 hybrid checkpoint 诊断，应作为独立 evaluation-only 工具移植和标注。

### 13.2 已接入的不可靠训练合同

1. `association_threshold` 默认仍为 `0.5`，但现在 ACTLayer 的可用动作 mask、环境的 association 执行以及相关判断共用同一阈值，避免策略能选而环境不执行或反之。
2. `offload_deadline_filter` 默认开启，保持旧行为；`--disable_offload_deadline_filter` 只关闭执行前的可行性筛选，不删除执行后的真实 deadline/reward 计算。
3. 动态 MD 不再硬编码五架 UAV。`n_UAVs`、`max_UAVs_in_neighbor`、`max_UAVs_obs_concat`、显式起点、critic token 数、Type-S receiver/sender、GRU predictor/bank 数和 checkpoint agent 数按同一 agent count 工作。
4. 显式 `uav_start_positions` 优先于 legacy layout fallback。可靠和不可靠模式因此能在完全相同的终局/初始几何合同下比较；Type-A 共识仍使用 reset 前由 episode `info` 保存的终局 UAV 位置。
5. `uav_resource_mode=heterogeneous` 要求恰好 `n_UAVs` 个有限正 scale factor，且总和为 `n_UAVs`。带宽和算力分别按 UAV 缩放，但系统总预算与 homogeneous 基线相同；所有 action execution 和 reward delay 路径读取 per-UAV capacity。
6. 这些变化没有改写现有 Type-S 几何 mask/丢包、receiver-local GRU、Type-A running-sum、advantage mode 或 PPO 后处理。可靠通信默认参数仍走原路径；不可靠通信只在其上正交替换 critic 数据面和 advantage 通信估计器。

### 13.3 新入口与维度边界

- 五 UAV 正式入口仍是 `onpolicy/scripts/train/run_fixed600_200_unreliable_dataplane.ps1`，新增阈值、deadline filter、异构资源和 PPO sensitivity 参数。
- 六 UAV 正式入口是 `onpolicy/scripts/train/run_fixed600_200_6uav_unreliable_dataplane.ps1`：6 UAV、72 GU、1+5 arrivals、MD12、R520、v30、显式六起点，可选择 `zero/last_obs/md_gru`、Actor-message、curriculum 和异构资源。
- homogeneous 五 UAV 且默认阈值/过滤时，旧 checkpoint 输入维度和行为保持兼容。改变 UAV 数或 `critic_md_metadata` 会改变 actor/critic checkpoint 数或 critic state 维度；五 UAV checkpoint 不能直接当六 UAV checkpoint 恢复。GRU checkpoint 同样绑定 receiver 数，不跨 5/6 UAV 静默复用。
- Actor-message 仍是独立可调的 actor 输入支路，并未纳入 Type-S 丢包模型。正式不可靠实验默认 `disabled`；若打开 `task_summary`，必须标为“可靠 Actor-message + 不可靠 critic 数据面”的组合实验。

### 13.4 本轮 CPU-only 验收

测试前资源为约 5% CPU、34.2 GB 可用内存，未发现 Python 训练进程。本轮设置 `CUDA_VISIBLE_DEVICES=-1` 和单线程 BLAS/OpenMP，没有启动 GPU 或 PPO 长训练。

- Python `py_compile`：通过；
- 4 个 PowerShell 启动器 AST parse：通过；
- 5-UAV/6-UAV launcher argv probe：通过，后者包含异构资源、阈值、no-deadline-filter 和 GRU 参数；
- 定向兼容测试：`74 passed`；
- 完整 `tests/`：`107 passed, 2 warnings`，两个 warning 均为既有动态 MD 统计在空计数时的除零提示；
- `git diff --check`：通过。

这证明当前单元级环境、动作、重建、共识、checkpoint 和启动参数合同兼容；它不等价于六 UAV MD-GRU 已完成正式收益验证。下一实验仍应采用单变量矩阵：先固定 homogeneous/threshold=0.5/filter-on，比 `zero/last_obs/md_gru`，再分别测试 6 UAV、异构资源或阈值，不能一次同时改变。
