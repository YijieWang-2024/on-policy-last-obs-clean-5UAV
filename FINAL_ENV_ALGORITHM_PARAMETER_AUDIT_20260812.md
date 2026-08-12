# 近两周 MEC 环境、算法与代码参数最终审计（截至 2026-08-12）

## 0. 结论先行

截至 2026-08-12 晚间，昨天和今天用于速度选择的正式合同已经收敛为：

- **环境**：`600 m × 600 m`、5 UAV、固定 `episode_template4_600_200` 第 0 个双热点布局、小区 `[0,200]×[0,200]`、大区 `[200,600]×[200,600]`、每 slot 候选 MD 为严格 `1+4`、MD lifetime=12、平均移动速度 3 m/s、无 UAV reset curriculum。
- **UAV 速度**：当前锁定 `v_max=30 m/s`；`Delta_t=0.5 s`，因此单 slot 最大位移 15 m。`v_max=40` 仍未跑满；2026-08-13 01:30 的本地冻结事件中，三 seed 共同只到 `21.7856M`，仍不能与 v30 的完整 60M 结果直接排名。
- **算法合同**：命令行名虽然是 `mappo`，实际是 separated runner 下的五套独立策略，使用局部奖励、R520 联合通信合同、Spatial Cartesian Flight Actor、receiver-gated task-summary 消息、ego-query attention critic，以及 `per_agent_noise/noise_scale=3/n_iterations=50` 的 DC-PPO 风格优势处理。
- **Actor 原始观测**：正式 message-capable 速度合同为每机 251 维；资源决策头实际读 211 维，飞行头读 251 维中的空间/移动/布局/邻机摘要子集。`actor_message_mode=disabled` 时 Actor 本体也缩为 211 维。
- **Critic 原始输入**：先从 Actor 观测删掉 40 维 Actor-only 消息，得到每 UAV 211 维 token；再把 R520 内最多 5 个 UAV token 拼成 1055 维并做 attention，token 0 为 ego query。
- **半径选择**：R520 是当前端到端联合合同的饱和点/保守折中，不是“Actor 通信 520 m 单因素最优”的证明。`neighbor_distance` 同时控制 Actor 消息、Critic mask 和 Metropolis 共识图。
- **噪声选择的真实含义**：参数写的是 noise=3，但 R520 完整速度运行中的 `noise_magnitude_mean` 约为 `4e-6` 量级，有效注入噪声已被共识残差几乎关闭；训练优势实际上非常接近纯局部 `A_i`。
- **环境边界**：Random12-600 是布局泛化/稳健性协议，不是本轮 `v_max` 选择环境；不能把 Random12、RandomLayout700 或旧 175m fixed 环境与本轮 fixed600-200 曲线合并排名。

这份审计同时区分三类判断：

1. `args.json` 和代码直接确定的事实；
2. 已有完整预算结果支持的参数选择；
3. 仍只是工程默认值、尚未由单因素消融证明的配置。

---

## 1. 执行代码分支与证据范围

### 1.1 本地执行分支

- 仓库：`D:\wyj\Projects\on-policy-last-obs-clean-5UAV`
- 当前分支：`agent/dcppo-runtime-optimization`
- 速度训练时可识别的 Git 基点：`716e080ac9cfaeed95a336ae0e14c8e696ed4c89`
- 基点提交说明：`feat: add 600m Random12 input-v2 experiments`
- 本地速度 seed2 当前解释器：`C:\Users\wyj2\.conda\envs\marl\python.exe`

速度实验执行时，本地工作区并非干净 HEAD。特别是 `onpolicy/envs/mec/mec.py` 当时还有未提交的固定布局候选出生 RNG 修改；本次审计因此以**实验时实际工作树代码 + 每个 run 的 `args.json` + 实际进程命令行**为准，而不是只按训练时 Git 基点推断。2026-08-13 的主项目归档把这些主线代码、启动器和轻量证据统一固化到当前分支，但不会改变“远端速度源码缺少独立 SHA manifest”的历史溯源边界。

### 1.2 远端 3090 的可复现边界

- 远端速度 seed32 和 v40 的事件文件、`args.json` 已在 2026-08-12 20:07 刷新到 `analysis/fixed600_speed_3seed_20260812/data/remote/`。
- 12 份速度实验 `args.json` 已逐字段比较；除 `experiment_name`、`seed`、`v_max` 外，其余字段完全一致。
- 远端项目没有 Git 元数据，因此不能诚实地给远端速度运行标一个“Git 分支/commit”。
- 8 月 10 日 Random12 实验有单独的源码 SHA-256 快照，见 `RANDOM12_RADIUS_BASELINE_RECORD_20260810.md`；但本轮 fixed600 速度实验没有再冻结一份完整源码哈希快照。
- 当前直接 SSH 认证不可用，因此本轮可以确认**远端实际参数与事件数据**，但无法追加确认速度实验全部源码与本地工作树逐字节相同。这是当前唯一的源码溯源缺口。

因此，最严谨的表述是：**本地逻辑代码族确定；12 路参数合同确定；远端速度源码的独立 commit/SHA 证据缺失。**

### 1.3 主要证据入口

- 12 路速度协议与曲线：`analysis/fixed600_speed_3seed_20260812/audit.md`
- 进度快照：`analysis/fixed600_speed_3seed_20260812/progress_summary.json`
- 远端事件和参数：`analysis/fixed600_speed_3seed_20260812/data/remote/`
- 8 月 7–9 日噪声/半径审计：`RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md` 第 10 节
- Random12 远端源码冻结：`RANDOM12_RADIUS_BASELINE_RECORD_20260810.md`
- 当前速度启动器：
  - 本地公共入口：`onpolicy/scripts/train/run_fixed2hotspot_per_agent_noise.ps1`
  - 远端速度入口：`onpolicy/scripts/train/run_fixed600_200_dcppo_R520_noise3p0_md12_seed32_vmax.sh`

---

## 2. 今天速度实验最终决定了什么

### 2.1 12 路正式矩阵

12 路实验只有 `seed`、`v_max`、`experiment_name` 不同：

| `v_max` | seed 2 | seed 32 | seed 42 |
|---:|---|---|---|
| 10 | 本地，运行中 | 远端，60M 完成 | 本地，60M 完成 |
| 20 | 本地，运行中 | 远端，60M 完成 | 本地，60M 完成 |
| 30 | 本地，运行中 | 远端，60M 完成 | 本地，60M 完成 |
| 40 | 远端，约 8M | 远端，约 8M | 远端，约 8M |

被排除的重复/旧运行：旧的 20260811 local seed2、remote v10 seed32 `run2`、remote v30 seed32 早期失败 retry；正式 v30 seed32 使用 `retry2/run1`。

### 2.2 完整 60M 结果

指标为 `agent0/system_performance_true_all_GUs`。为避免单个末点抖动，以下使用每条完整曲线末 100 个记录点的均值，再对两个训练 seed 汇总：

| `v_max` | seed32，远端 tail100 | seed42，本地 tail100 | 两 seed 均值 | 两 seed离散度（population std） |
|---:|---:|---:|---:|---:|
| 10 | 474,290 | 495,761 | **485,026** | 10,736 |
| 20 | 489,926 | 503,460 | **496,693** | 6,767 |
| 30 | 562,101 | 563,115 | **562,608** | 507 |

`v_max=30` 相比 20 的两 seed tail100 均值高约 **13.27%**，相比 10 高约 **15.99%**，而且本地 seed42 与远端 seed32 高度一致。因此，**今天能够正式决定的是 10/20/30 中选择 30。**

### 2.3 为什么暂时不能选择 40

v40 三个远端 seed 在最初权威快照时只到约 7.76M–8.63M；提交归档前再次读取本地冻结事件，三条分别到 `23.1168M/22.5536M/21.7856M`，共同预算为 `21.7856M`。它们仍未达到 60M，因此既不能把当前末点当作最终性能，也不能和已完成 60M 的 v30 末端直接比较。

因此当前参数冻结为：

```text
v_max = 30 m/s
Delta_t = 0.5 s
max displacement per slot = 15 m
```

但这应标为“**30 已胜过完整的 10/20；40 待完整预算复核**”，不能写成“10/20/30/40 四者已全部完成且 30 全局最优”。

### 2.4 计算设备速度不是算法变量

相同协议下，快照中的近期环境步吞吐约为：

- v10/v20：本地约 752–773 steps/s，远端约 848–890 steps/s；
- v30：本地/远端约 1,116/1,151 steps/s；
- v40 早期远端约 805–835 steps/s。

这些是运行吞吐，不是 UAV 物理速度的收益解释。v30 的性能结论来自相同 60M 环境步预算，而不是因为它在墙钟时间上跑得更快。

---

## 3. 最终速度选择环境：精确代码合同

### 3.1 地图、热点和 UAV 初始部署

| 参数 | 最终值 | 实际含义 |
|---|---:|---|
| `x/y_min/max_uav` | 0/600 | UAV 在 600m 方形地图内运动 |
| `x/y_min/max_gu` | 0/600 | MD 候选出生和移动的地图边界 |
| `hotspot_layout_mode` | `episode_template4_600_200` | 600m 专用的 200m/400m 双矩形模板 |
| `fix_hotspot` | true | 固定热点合同 |
| 实际 `layout_index` | 0 | 代码对该模式强制 index 0，不随机抽四布局 |
| 小区 | `[0,200]×[0,200]` | 每 slot 1 个候选 |
| 大区 | `[200,600]×[200,600]` | 每 slot 4 个候选 |
| `n_UAVs` | 5 | 五架 UAV |
| `uav_start_positions` | `(110,180),(220,180),(330,180),(440,180),(400,400)` | 固定显式起点 |
| `H_UAV/H_GU` | 120m / 1m | 高度 |
| `Cover_R` | 120m | UAV 对 MD 的覆盖/接纳半径 |
| `uav_reset_curriculum` | false | 不使用 reset curriculum |

代码依据：`mec.py:980-990` 定义四个 600/200 布局，`mec.py:1022-1059` 对 `episode_template4_600_200` 强制 `layout_index=0`。

这里容易混淆的三套环境必须分开：

| 环境 | 小区 | 布局 | 角色 |
|---|---:|---|---|
| 旧 `fixed_legacy` | 175×175 | 固定 | 8 月 7–9 日噪声阈值和早期半径审计 |
| **当前 speed fixed600-200** | **200×200** | **固定 index 0** | **今天 v_max 选择的正式环境** |
| Random12-600 input-v2 | 200×200 | 每 episode 从 12 个有序角落组合采样 | 泛化/稳健性协议，不参加本轮速度排名 |

### 3.2 动态 MD 出生、容量和移动

| 参数 | 最终值 | 说明 |
|---|---:|---|
| `dynamic_md` | true | 动态 MD，不是固定 60 用户 |
| `md_arrivals_min/max` | 5/5 | 每 slot 恰好 5 个候选 |
| `md_arrivals_per_region` | `[1,4]` | 小区 1、大区 4 |
| `n_GUs` | 60 | 活跃 MD 槽位容量，不代表始终有 60 个活跃 MD |
| `md_lifetime_min/max` | 12/12 | 每个被接纳 MD 存活 12 slots |
| `mean_velocity` | 3 m/s | MD 平均速度 |
| 初始速度 std | 0.6 | Gaussian-Markov 初始化扰动 |
| 初始倍率 | `[0.4,1.6]` | 初始速度尺度范围 |
| 更新裁剪 | `[0,5] m/s` | MD 后续速度范围 |

满覆盖且没有槽位/边界损失时，`5 arrivals/slot × 12 slots = 60`，正好达到 60 个槽位容量。但实际候选只有出生时处在至少一个 UAV 120m 覆盖内才被接纳，离开联合覆盖的活动 MD 也会退出，所以“固定 60 活跃用户”的说法不正确。

候选位置在各自矩形内均匀生成。当前本地工作树为 static template 模式使用独立 `candidate_birth_rng`，见 `mec.py:840-867`；这用于避免准入行为改变后续候选随机流。

### 3.3 每个 episode 和任务分布

- `episode_length=400`
- `Delta_t=0.5 s`
- 每 episode 物理时间 200s
- 训练预算 `num_env_steps=60,000,000`
- `n_rollout_threads=64`

每个 MD 当前任务：

| 参数 | 最终值 |
|---|---:|
| 数据量 `D` | Uniform[200,000, 400,000] bits |
| CPU cycles `C` | Uniform[675,000,000, 800,000,000] |
| deadline | Uniform[0.499, 0.5] s |
| UAV 总计算 `F_m` | 20 GHz |
| 本地 MD 计算 `F_n` | 1.5 GHz |
| 总带宽 `B` | 30 MHz |
| `Dis_min` | 3m |

### 3.4 奖励合同

当前启动器固定：

```text
alpha_r=32, beta_r=0.5, gamma_r=26, delta_r=32, epsilon_r=0
lambda_r=1e-6, mu_r=64
q1=q2=q3=q4=q5=1
w1=20, w2=1, p3=500
local_reward=true
not_served_rew_to_nearest=true
```

注意 `gamma_r=26` 是环境奖励权重；PPO 折扣因子 `gamma=0.99` 是另一个参数，不能混写。

---

## 4. 最终算法到底是什么

### 4.1 为什么不能只写“MAPPO”

运行路径和目录中显示 `algorithm_name=mappo`，这只决定底层 PPO 实现和关闭循环网络：

```text
algorithm_name=mappo
use_recurrent_policy=false
use_naive_recurrent_policy=false
use_centralized_V=true
```

真正决定当前论文/实验算法的是：

```text
share_policy=false                       # separated runner
local_reward=true
advantage_mode=per_agent_noise
noise_scale=3.0
n_iterations=50
neighbor_distance=520
actor_message_mode=task_summary
actor_message_pool=receiver_gated_sum
actor_message_contract=absolute_raw_v2
spatial_flight_actor=true
state_is_k_hops=true
all_uav_k_hops=true
use_atten_critic=true
ego_query_critic=true
shared_ret_norm=true
```

因此更准确的名称是：

> **五策略 separated MAPPO/PPO 后端上的 DC-PPO-R520 联合合同：局部奖励与局部优势、残差自适应噪声、Actor 可执行任务/几何消息、R520 k-hop ego-query critic。**

### 4.2 五套独立网络，不共享策略

`config.py` 中 `--share_policy` 使用 `action='store_false'`。启动器传了 `--share_policy`，结果反而是：

```text
share_policy = false
share_actor = false
whether_average_network_parameters = false
```

`train_mec.py:517-520` 因此选择 `onpolicy.runner.separated.mec_runner`。五架 UAV 各自有一套 actor、critic、optimizer 和 feature normalizer；不共享 actor，也不做网络参数平均。

这和“共享 return normalization”并不冲突：后者只让五个 reward/discounted-return scaler 使用同一 pooled return 分布，不共享网络参数。

### 4.3 最终优势计算

每架 UAV 首先用自己的 return 和自己的 critic 值形成局部优势：

```text
A_i_local = returns_i - V_i
```

然后在 rollout 末端保留的 UAV 位置上，用 `neighbor_distance=520` 构图，做 50 轮 Metropolis 混合，得到 `A_i_cons`。每个 UAV/时刻/环境的噪声系数为：

```text
m_i = clip(
  |A_i_cons - mean_j(A_j_local)| / |A_i_local - mean_j(A_j_local)|,
  0, 1
)
```

分母为零时采用 1 作为 fallback，使 R0/self-only 情况保持单位噪声系数。最终：

```text
A_i_train = A_i_local
          + m_i * 3.0 * std_j(A_j_local) * epsilon_i,
epsilon_i ~ N(0,1)
```

代码在 `onpolicy/runner/separated/mec_runner.py:371-398`。这里**不加入全局精确均值，也不把 consensus advantage 直接加回去**。

进入 PPO 前，每架策略内部再对自己的 active samples 标准化，并裁剪到 `[-10,10]`，见 `onpolicy/algorithms/r_mappo/r_mappo.py:249-277`。

### 4.4 noise=3 为什么现在近似局部优势

早期固定 175m 环境中，R0 的 `m_i=1`，noise=3 会显著伤害优化；R260/R520/R780 做 50 轮共识后，tail100 的 `m_i` 分别约 `1.45e-5/2.17e-6/1.90e-6`。

当前完整 v30 速度 runs 中，`noise_magnitude_mean` 约 `4.25e-6`（seed32）和 `4.80e-6`（seed42）。因此配置名仍写 noise3，但有效噪声项只有 `3 × m_i` 的极小尺度。当前的工程选择本质上是：

- 保留 DC-PPO 共识残差定义；
- 在 R520 连接图上让残差几乎归零；
- Actor 更新实际接近纯局部 `A_i`。

这也是为什么不能把 R0→R520 的收益全解释成“通信消息收益”：R0 同时承受了单位系数的强噪声。

---

## 5. Actor 输入、网络分支和动作

### 5.1 Actor 原始观测：251 维

实际训练日志确认每架 UAV：

```text
observation_space = (251,)
```

精确组成：

| 切片 | 维度 | 内容 |
|---|---:|---|
| self | 2 | 自身 UAV 绝对 `x,y` |
| Actor message block | 40 | 4 个其他 UAV × 每包 10 维 |
| layout context | 8 | 小区和大区两个矩形的 `[xmin,xmax,ymin,ymax]`，单位为 meter |
| local active count | 1 | 当前该 UAV 覆盖内活动 MD 数量 |
| local user slots | 200 | 最多 20 个用户 × 每用户 10 维 |
| **总计** | **251** | `2+40+8+1+20×10` |

没有 timestep，也没有 one-hot agent ID。

### 5.2 每个用户 slot 的 10 维

动态 MD 时每个 slot 是：

```text
[x, y,
 speed, current_direction, base_direction, remaining_lifetime,
 channel_gain,
 data_bits, cpu_cycles, deadline]
```

只保留覆盖范围内排序后的前 20 个用户，不足部分补零。排序不是纯距离：

1. 本地计算无法在 deadline 内完成的任务优先；
2. 可本地完成的任务在后；
3. 每组内部再按 UAV-MD 距离升序。

对应 `completion_priority_user_sort=true`，代码在 `mec.py:3692-3730`。

### 5.3 每个邻机消息包的 10 维

对每个其他 UAV 固定一个 slot：

```text
[sender_x, sender_y,
 sender_active_count,
 sender_local_GU_centroid_x, sender_local_GU_centroid_y,
 sender_mean_GU_velocity_x, sender_mean_GU_velocity_y,
 sender_GU_RMS_spread,
 sender_hard_task_fraction,
 message_mask]
```

- 只有 UAV 间距不超过 `neighbor_distance=520` 时 mask=1；否则整包为零。
- `absolute_raw_v2` 表示前 9 个物理字段在进入 normalizer 前使用绝对/raw 量；它们仍会做 observation normalization。
- 只有第 10 个二值 mask 绕过 observation normalization。
- 生成代码在 `mec.py:3325-3414`，preserve mask 的归一化代码在 `vec_normalize.py:106-128`。

### 5.4 Actor 实际是双输入头

#### 飞行头

`SpatialFlightEncoder` 接收 251 维 tensor，但只主动读取：

- 自身 `x,y`；
- 每个用户 slot 的前 6 维：`x,y,speed,current_direction,base_direction,remaining_lifetime`；
- 用户可用 mask 和由此得到的 occupancy；
- 8 维 layout context；
- 40 维其他 UAV task/geometry messages。

飞行头明确**不读取** channel gain 和当前 i.i.d. task 三元组 `(D,C,deadline)`。20 个用户先逐个编码再 masked mean-pool，因此对 slot 顺序在聚合层面保持置换不变。

四个 sender 的消息先各自编码；`receiver_gated_sum` 用接收者 self descriptor、编码后的 sender message 和 sender xy 计算独立 sigmoid gate，最后求 gated sum/4。gate bias 初始化为 `-2.197...`，即初始 gate 约 0.1。

代码在 `r_actor_critic.py:10-168`。

#### 关联/带宽/计算资源头

资源头会把 40 维 Actor message block 完整删除，只接收 211 维：

```text
self xy (2) + layout (8) + count (1) + 20 users × 10 = 211
```

因此当前邻机通信**只直接进入飞行动作头**，不直接进入关联、带宽、计算资源分配头。代码在 `r_actor_critic.py:201-226`。

### 5.5 Actor 输出动作：62 维

每架 UAV 的 action space：

| 动作头 | 维度 | 分布/处理 |
|---|---:|---|
| Cartesian flight proposal | 2 | 无界 2D Gaussian proposal |
| continuous association | 20 | Gaussian，0.5 阈值形成服务 mask |
| bandwidth allocation | 20 | Dirichlet |
| computation allocation | 20 | Dirichlet |
| **总计** | **62** | 只对当前 20 个本地候选 slot |

飞行 proposal `p` 被投影：

```text
velocity = v_max * p / max(||p||, 1)
position_xy += velocity * 0.5
position_xy = clip(position_xy, [0,600])
```

因此 proposal 范数小于 1 时可选择低于最大速度，范数大于 1 时投影到最大速度圆盘；不是每步都强制以 30 m/s 飞行。代码在 `mec.py:94-98` 和 `mec.py:2790-2811`。

---

## 6. Critic 输入和价值网络

### 6.1 Critic 每个 UAV token：211 维

环境先调用 `get_critic_local_obs()` 删除 40 维 Actor-only message block，保留：

```text
self xy (2)
+ layout context (8)
+ local active count (1)
+ local 20 users × 10 fields (200)
= 211 dimensions
```

Critic 不直接读取 Actor 的压缩 task-summary message，但会读取每个可见 UAV 自己的完整局部 MD token，包括用户位置/运动/信道/任务，以及同一个布局 context。

### 6.2 R520 k-hop state：1055 维

实际训练日志确认：

```text
share_observation_space = (1055,) = 5 × 211
```

当前设置：

```text
state_is_k_hops=true
all_uav_k_hops=true
max_UAVs_obs_concat=5
neighbor_distance=520
```

对每个 ego UAV：

- token 0 被重排为自己；
- 与 ego 距离 `<=520m` 的其他 UAV token 激活；
- 更远 token 置零且 attention mask=0；
- 600m 地图对角线约 848.5m，因此 R520 不是永久全局可见。

构造代码在 `mec.py:1548-1585` 和 `mec.py:2280-2308`。

### 6.3 ego-query attention critic

每个 211 维 token 先独立编码到 hidden=256，再做 UAV 间 attention。`ego_query_critic=true` 时只取 `attended_features[:,0]` 作为聚合结果，即：

- ego token 是 query anchor；
- 所有 R520 内 token 可作为 keys/values；
- 不做所有 UAV attended token 的简单均值。

随后经过两层 MLP 和 scalar value head 输出当前 UAV 的 `V_i`。代码在 `r_actor_critic_attention.py:226-318`。

虽然 `use_centralized_V=true`，它的语义是“使用 share_obs/state 输入 critic”，**不是无条件全局全状态 critic**；真实可见范围仍受 R520 mask 限制，而且五架 UAV 有五套独立 critic。

---

## 7. 归一化、PPO 和网络超参数

### 7.1 两个反直觉布尔开关

启动器写了：

```text
--share_policy
--use_valuenorm
```

但这两个参数在 parser 中都是 `store_false`，实际结果是：

```text
share_policy=false
use_valuenorm=false
```

这两个真实值必须以 `args.json` 为准，不能按 flag 英文字面解释。

### 7.2 最终归一化

```text
ob_norm=true
ret_norm=true
shared_ret_norm=true
use_valuenorm=false
```

- 每个 UAV 有自己的 observation/state RunningMeanStd；
- Actor message 的前 9 个字段也进入 observation normalization，mask 保留原值；
- layout context 使用 meter raw 值进入 normalizer，不是手工除以 600；
- reward 由 discounted-return 标准差缩放；
- `shared_ret_norm=true` 把五架 UAV 的 discounted returns 合并更新同一统计分布；
- 这不共享网络，也不等于团队 reward/团队 advantage。

### 7.3 PPO/网络参数

| 参数 | 最终值 |
|---|---:|
| hidden size | 256 |
| MLP layers | 2 |
| actor lr | 1e-4 |
| critic lr | 5e-4 |
| PPO clip | 0.15 |
| PPO epochs/update | 4 |
| mini-batches | 1 |
| entropy coefficient | 0 |
| discount `gamma` | 0.99 |
| GAE lambda | 0.95 |
| value loss coefficient | 0.5 |
| max grad norm | 0.5 |
| value loss | clipped Huber，`huber_delta=10` |
| init / activation | orthogonal + ReLU |
| lr decay | disabled |
| CUDA deterministic | enabled |
| training threads | 1 |
| rollout threads | 64 |

---

## 8. 近两周所有主要思路分歧：尝试、结果和今天的处理

| 疑惑/分歧参数 | 近两周尝试 | 今天的选择 | 证据与边界 |
|---|---|---|---|
| 环境地图/热点 | Fixed175、Fixed200、RandomLayout700、Random12-600、Diag4、FixedDiag、MovingHotspot | **速度/主参数选择用 Fixed600-200 index0**；Random12 单列泛化 | 不同环境不可合并绝对 True60；700m/Diag/Moving 未给稳定单调半径 |
| UAV reset curriculum | 无 curriculum、多个 curriculum | **false** | curriculum 明显改变分布，多组破坏半径顺序 |
| MD arrivals | 旧 `1+5`、错误 6/slot、当前 `1+4` | **严格 `[1,4]`** | 当前问题是 1+4 负载结构 |
| lifetime | 10、12 | **12** | 5×12 与 60 slots 对齐；速度矩阵统一 MD12 |
| MD 平均速度 | 3、10 | **3 m/s** | 提高 MD 到 10 未制造可靠通信收益；停止该网格 |
| UAV `v_max` | 10/20/30/40 | **30 m/s（40 待跑满）** | 30 在两个完整 seed 上比 20 高约 13.27%；40 只有早期数据 |
| 飞行动作 | 旧 polar、Cartesian、Spatial Cartesian | **Spatial Cartesian** | 飞行 proposal 可学低速并投影到速度圆盘；空间池化更符合用户集合 |
| 用户排序 | 纯距离、completion priority | **completion priority** | 不能本地完成者优先，减少 slot 语义漂移 |
| Actor 邻居输入 | legacy `actor_neighbor_obs`、消息 disabled/zero、geometry、task-summary | **legacy false；task-summary 开启** | `neighbor_R` legacy 路径关闭；task-summary 同时含几何和负载摘要 |
| 消息池 | mean、receiver-gated sum | **receiver-gated sum** | mean 易稀释；gated 是当前实现，但固定 MD12 下独立 no-message 完整消融尚未完成 |
| 消息数值合同 | `relative_scaled_v1`、`absolute_raw_v2` | **absolute_raw_v2** | 物理量进 ob_norm，只有 mask 保留，避免双重手工缩放 |
| 布局上下文 | 无、`normalized_v1`、`meters_v2` | **开启 meters_v2** | 8 维矩形边界同时给 Actor 资源头/飞行头和 Critic，再由 ob_norm 标准化 |
| 通信半径 | 0/200/260/400/520/600/780/1000 | **520** | R260 后联合机制基本饱和；520 是保守折中，不是 actor-only 最优证明 |
| `neighbor_R` vs `neighbor_distance` | 经常一起修改 | **都写 520，但只有 distance 进当前有效消息/critic/共识路径** | `actor_neighbor_obs=false` 使 `neighbor_R` 只是兼容字段 |
| advantage | local、legacy residual、pure consensus、mixed consensus、externality beta、per-agent noise | **per-agent noise 外壳，实际近 local A_i** | pure/mixed/externality 多次不支持主线；当前 R520 的 m 极小 |
| noise scale | 0、0.2/0.4/0.8、1.4、2、2.5、2.66、2.83、3、4、8、16、30 | **配置 3** | R0 中 2.5–2.66 出现优化阈值；R520 中有效噪声被 m 压到约 1e-6，不能称 noise3 本身更优 |
| consensus iterations K | 1/5/50 | **50** | K50 未证明优于 K1；这里只用于把残差算到近零，不作为独立贡献 |
| critic aggregation | mean-pool、ego-query attention | **ego-query** | ego anchor 更符合每个 UAV 的局部价值；mean-pool没有稳定排序 |
| critic 可见性 | local、全局、k-hop | **R520 k-hop local-centralized** | `use_centralized_V=true` 不等于永久全局 critic |
| 网络共享 | shared policy、separate policy、参数平均 | **五套 separate，不平均参数** | `--share_policy` 是 store_false；不要误写为 parameter sharing MAPPO |
| return/value norm | ValueNorm、独立 return RMS、shared return RMS | **shared return RMS；ValueNorm off** | 共享尺度而不是共享奖励/网络 |
| 错误 `D_min` | 200、200000 | **200000** | `D_min=200` 的长运行已永久隔离 |

---

## 9. 哪些结论已经确定，哪些仍不能写成论文结论

### 9.1 已确定的工程参数

- 固定 speed benchmark 使用 Fixed600-200、1+4、MD12、无 curriculum。
- `v_max=30` 是 10/20/30 完整预算中的明确选择。
- 当前实现使用 R520、task-summary receiver gate、absolute_raw_v2、layout meters_v2。
- 当前是 separated 五策略、Spatial Cartesian actor、ego-query R520 critic。
- Actor 251 维、资源头 211 维、critic 5×211=1055 维。
- PPO、归一化、任务和奖励参数已由 12 份 `args.json` 锁定。

### 9.2 目前只能称为“当前合同”，不能称为被单因素证明

- R520 不能称为 Actor 通信半径的单因素最优值，因为它同时改变三条路径。
- noise=3 不能称为最佳噪声幅度；在当前 R520 中它几乎不实际注入噪声。
- receiver-gated task-summary 是当前默认 Actor 通信实现，但当前 fixed MD12 的 no-message 完整反事实仍未提供最终因果确认。
- Fixed600-200 是速度选择/算法收敛环境；是否把它作为论文唯一主环境，还需与 Random12 泛化协议并列设计，而不是直接抹掉 Random12。
- v40 未满 60M；若后续完整多 seed 明显超过 v30，需要重新打开速度选择，但现在不能提前改参数。

### 9.3 当前最稳妥的算法表述

建议在项目文档/论文中写：

> 我们在固定 600m 双热点动态 MEC 环境中，采用五个独立策略的局部奖励 PPO 后端。每个飞行 Actor 使用自身局部移动用户集合、布局上下文以及 R520 内 receiver-gated UAV 任务/几何摘要；每个 ego-query Critic 使用 R520 内 UAV 的局部 211 维状态 token。训练保留 50 轮 Metropolis 共识残差定义的自适应噪声项，但在 R520 下该残差接近零，因此有效更新接近局部优势。完整的 v10/v20/v30 两 seed 实验选择 v30；v40 尚待完整预算。

不要写成：

> “MAPPO 全局 critic + 共享策略已经证明通信距离越大越好，noise3 最优。”

这句话与实际执行分支、输入切片和实验因果关系都不一致。

---

## 10. 后续只需要补的最小证据

1. 等 v40 三个 seed 到完整 60M，再做与 v30 相同的 tail100/matched-step 比较；在此之前冻结 v30。
2. 等本地 seed2 的 v10/v20/v30 完成，用三 seed 均值复核，但不因早期点改变已完成双 seed 判断。
3. 对最终 v30/R520 checkpoint 做同 checkpoint 的 normal-message vs no-message 评估，确认 Actor 是否实际使用消息。
4. 未来远端正式运行必须保存源码 manifest/SHA；只有 `args.json` 不足以证明本地/远端代码逐字节一致。
5. Random12 作为单独 generalization protocol 汇报，不与 fixed speed curves 直接按绝对 True60 拼接排名。

除上述五项外，不建议重新开启 noise 小数点网格、K 网格、externality beta、pure/mixed consensus 或更大 radius 网格。

---

## 11. 2026-08-10 至 08-11：不用 Actor message 的对照复核

### 11.1 这组实验确实存在

2026-08-10 在远端 RTX 3090 上启动了三条 `no_actor_message` 训练：

| 实验 | 半径 | 冻结事件最后 step（08-11 10:21 快照） |
|---|---:|---:|
| `dcppoR0_..._no_actor_message_seed2_60m_20260810` | 0 | 38.0672M |
| `dcppoR260_..._no_actor_message_seed2_60m_20260810` | 260 | 38.3744M |
| `dcppoR520_..._no_actor_message_seed2_60m_20260810` | 520 | 40.8832M |

冻结证据位于 `analysis/fixed600_no_actor_vs_actor_20260811/`。当时三个远端主进程仍在运行，目标预算均为 60M；本地保存的最后一份快照没有覆盖到最终 60M。当前不能直接 SSH 刷新远端，因此以下结论严格以已冻结的 40.8832M 以内数据为准。

### 11.2 R520 的配置单变量模块级消融

R520 no-message 与当时本地 R520 actor-message/v30/seed2 的两份 `args.json` 逐字段比较，只有：

```text
experiment_name: 不同
actor_message_mode: disabled  <->  task_summary
```

其余环境、seed、v30、R520、MD12、noise3、PPO、Critic、归一化和训练预算完全一致。因此，这是一个 `actor_message_mode` 配置单变量、但同时改变 Actor 输入维度和 message encoder/gate 参数容量的**模块级从头训练消融**；它不是“冻结同一 checkpoint 后仅屏蔽消息内容”的因果反事实。

在两条曲线共同达到的 40.8832M：

| 指标 | no Actor message | Actor message | message − no-message |
|---|---:|---:|---:|
| tail25 True60 | 547,509 | 547,601 | +92 / **+0.017%** |
| tail100 True60 | 548,930 | 551,410 | +2,480 / **+0.452%** |

不同共同预算的 tail100 差异也一直很小：

| 共同 step | message 相对 no-message |
|---:|---:|
| 9.9584M | −0.025% |
| 19.9936M | +0.069% |
| 29.9776M | +0.437% |
| 38.0672M | +0.729% |

38.0672M 的配套诊断同样接近：总 admission 0.9156 vs 0.9177、平均 active MD 52.206 vs 52.336；message 版本的末 100 个 actor-message neighbor fraction 约 0.902，说明它不是因为完全收不到消息才与 no-message 接近。

因此，“当时用不用 Actor message 看起来没什么差别”是有真实实验依据的；更准确的结论是：

> 在 Fixed600-200、MD12、v30、R520、noise3、seed2 的单次训练中，截至共同 40.8832M，加入 task-summary receiver-gated Actor message 只带来约 0%–0.45% 的末窗差异，tail25 几乎相等。

但这仍然不是最终的“Actor message 无用”结论，因为：

- 只有一个训练 seed；
- no-message 本地冻结快照未到 60M；
- 比较的是分别从头训练的两套网络，不是同一 checkpoint 的测试时屏蔽反事实；
- 没有固定评估 episode 的配对置信区间。

### 11.3 与正式速度实验是否一致

**环境和算法主合同一致。** no-message R520 与正式速度矩阵的 v30 配置相比：

- 相同：Fixed600-200 index0、1+4、MD12、MD 3m/s、固定五 UAV 起点、无 curriculum、v30、R520、noise3、K50、Spatial Cartesian Actor、completion-priority、R520 ego-query critic、shared return normalization、60M/64 workers；
- 有意差异：`actor_message_mode=disabled`，正式速度实验为 `task_summary`；
- 统计差异：no-message 只有 seed2；正式完整速度证据目前主要是 seed32/42，seed2 正在补跑。

所以它与速度实验属于同一个 fixed600-v30 算法族，可以用来回答“Actor message 模块是否改善该合同”；但不能把 seed2 no-message 的 40.8832M 直接与 seed32/42 的 60M 最终值按绝对数排名。

### 11.4 `disabled` 到底关闭了什么

启动器不是把 40 维消息置零，而是使用：

```text
--actor_neighbor_obs            # 不传；实际 false
--actor_message_mode disabled
--actor_message_pool receiver_gated_sum
--actor_message_contract absolute_raw_v2
--spatial_flight_actor
```

`mec.py:414-422` 使：

```text
actor_message_dim = 0
actor_only_obs_dim = 0
```

因此环境不生成消息 block，Actor observation 从 251 维变成 211 维：

```text
self xy 2
+ layout context 8
+ local active count 1
+ 20 local users * 10
= 211
```

Spatial flight head 仍然存在，仍读取自身位置、用户移动/lifetime、occupancy 和 8 维布局上下文；只是 `message_dim=0`，不创建 `message_encoder` 和 `message_gate`，forward 也不做 sender pooling。

资源/关联/带宽/计算头在有 message 的正式版本本来就会主动删除 40 维消息，因此它在两种配置下都读取相同的 211 维本地信息。也就是说，这个 ablation 的直接结构变化只发生在**飞行头**。

### 11.5 Critic 和优势仍保留通信

“no Actor message”不等于完全去中心化或完全无通信：

- `neighbor_distance=520` 仍使 Critic 读取 R520 内 5×211=1055 维 k-hop tokens；
- `use_atten_critic=true`、`ego_query_critic=true` 不变；
- 50 轮 Metropolis 图和 `per_agent_noise` 的残差系数仍使用 R520；
- `neighbor_R=520` 仍是兼容字段，因 `actor_neighbor_obs=false` 不进入 legacy Actor 邻居位置路径。

因此该实验隔离的是：

```text
有无 Actor flight-head 的显式 task/geometry 消息
```

而不是：

```text
整个系统有无 UAV 通信
```

如果后续要做同 checkpoint 的反事实，应保留 message-capable checkpoint 和冻结 normalizer，仅在评估时把消息 payload/mask 置零；这与这里 `actor_message_mode=disabled` 从头训练、输入维度从 251 变 211 的实验不是同一种问题。
