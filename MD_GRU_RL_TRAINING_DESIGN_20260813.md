# 不可靠 Type-S 下 MD-GRU 与 DC-PPO 的训练设计

## 结论

MD-GRU 应是 **critic-only 的因果状态估计器**。它不改变 actor 输入，也不参与执行时的 MD 关联；它只在训练期 Type-S 丢包时补齐邻居 UAV 的 MD token，使 critic 能估计价值并构造 local advantage。

每架接收 UAV (i) 有一套本地参数 ((\phi_i,\varphi_i))。该 UAV 维护的所有 MD ID 共享这套参数，但每个 `(env, receiver i, MD session n)` 都有独立的 online hidden、age、剩余 lifetime 和最后一次任务可用性。换句话说，是“每个 MD 一份状态”，不是“每个 MD 一个网络”。

## 每时隙数据流

1. 环境先产生 actor 的本地观测；它与 Type-S 是否丢包无关。
2. sender UAV 将 canonical critic-local MD records 编码为 Type-S。
3. receiver (i) 合并自己的本地记录和当前成功收到的 sender records，按 MD session ID 去重。
4. 对重新观测的 MD：用真实 persistent mobility feature 更新其 hidden，age 置零，并记录一个监督样本。
5. 对本时隙没有观测的已知 MD：hidden 不用未知真值更新；prediction head 根据保存的 hidden 和增大的 age 预测 persistent feature。
6. 使用可靠低速控制面的 sender UAV 位置，判断预测 MD 是否落在该 sender 覆盖区，重算 channel gain 并构造 surrogate token。
7. 任务 `D/C/deadline` 不预测；未知时使用 placeholder，并令 `task_valid=0`。critic 同时读取 `record_valid/task_valid/age`。
8. 成功 Type-S 用 direct token；丢失 Type-S 用 surrogate token。组合后的 1355 维状态只送给 receiver critic。

## 两类存储不能混为一谈

### Online belief memory

- 保存位置：每个环境副本、receiver UAV、MD session。
- 保存内容：hidden、age、lifetime、last/estimated feature、当前任务有效性、原始成功观测历史。
- episode 结束必须清空，因为新 episode 的 MD session 和初始状态是新的因果过程。
- MD lifetime 到零也应释放，避免容量随历史 session 数持续增长。

### Supervised sequence replay

- 保存内容：重观测之前的原始成功观测输入序列、当前 gap/age、重观测真实 feature。
- episode 结束不清空；它是训练数据集，不是环境状态。
- 使用有界 FIFO。当前上限由 `md_gru_max_samples=32768` 控制，每架 UAV 各自一份。
- replay 容量不是每轮训练量；每次 PPO 更新后最多抽 `md_gru_train_samples=512` 条，并做 `md_gru_epochs=4`。达到 `md_gru_min_ready_samples=512` 前，critic 继续使用 last-observation fallback。
- 训练时从零 hidden 用当前 GRU 参数重放序列，不能直接保存旧模型生成的 hidden；否则模型更新后会产生 recurrent-state staleness。

本任务的 MD lifetime 只有 12 slots，所以保存完整短序列比 R2D2 式长 burn-in 更简单；如果以后 lifetime 扩大到数百 slots，再改为固定 window + burn-in。

## 损失与梯度流

当前推荐损失是每个 receiver 独立的监督 Smooth-L1：

\[
L^{pred}_i=\mathbb E_{(n,t)\in\mathcal D_i}
\operatorname{SmoothL1}(\widehat z^{pre}_{in,t},z_{n,t}).
\]

选择 Smooth-L1 而非纯 MSE，是为了减弱边界反弹、长 gap 和少数异常移动样本对梯度的支配。输入/目标对 (x,y,speed,\sin/\cos(direction),\sin/\cos(reference)) 做固定物理范围归一化；日志必须至少报告米制 position RMSE，并按 age 分桶报告。

梯度契约为：

```text
MD replay -> GRU/prediction head -> L_pred -> GRU optimizer only
reconstructed NumPy critic state -> critic -> L_value -> critic optimizer only
local/consensus advantage -> PPO clip -> actor optimizer only
```

没有 critic 梯度穿过 surrogate state 回到 GRU。原因是当前重构在 rollout 期间以 detached NumPy 状态写入 PPO buffer。论文中的

\[
L^{train}_i=L^V_i+\lambda_p L^{pred}_i
\]

若两套参数不共享，就只是两个可分别优化的项；`lambda_p` 不代表 value/prediction 的真实端到端权衡。尤其 Adam 对常数梯度缩放大部分会归一化，旧代码的 `0.1 * loss` 含义很弱。更准确的论文表述应是“每个 rollout 后交替执行 critic PPO update 与 K 次 predictor supervised update”。

## 推荐的交替训练时序

一个 rollout/episode 的稳定顺序为：

1. 固定本轮 predictor 参数，采集整个 400-slot rollout，写入 PPO on-policy buffer 和本地 estimator replay。
2. 用采样时的 critic states 计算 returns/local advantages，再执行不可靠 running-sum ratio consensus。
3. 训练 PPO actor/critic；PPO epoch 内不更新 predictor，避免同一 on-policy batch 的状态表征漂移。
4. 每架 UAV 从自己的 sequence replay 做 `md_gru_epochs` 次监督更新。
5. reset online hidden，使用更新后的 predictor 构造下一 episode 的初始 critic state。
6. checkpoint 五套 predictor、optimizer、ready 状态；评估时冻结 predictor。

当前 checkpoint 不包含 supervised sequence replay。恢复时模型、optimizer 和 ready 状态延续，但 replay 从空队列重新积累；这不会让已 ready 的模型退回随机初始化，却不保证数据采样级 bitwise exact resume。若要做可中断的正式长实验，应把五个 replay 及其 RNG 一并纳入 checkpoint。

runner 已按上述顺序调整为 `compute -> PPO update -> predictor update -> next rollout`。本轮 PPO 使用采样时固定 predictor 产生并 detached 的 critic states；更新后的 predictor 只影响下一 rollout。由于 PPO 的 `after_update()` 已把 terminal 状态复制到 buffer slot 0，predictor 更新后的初始 critic state 也明确刷新到 slot 0。

## 不建议当前就做的事情

- 不把 critic value loss 端到端反传进 GRU。它会让 mobility belief 为短期 value fitting 而漂移，难以解释，也会破坏固定 rollout 表征。
- 不预测独立重采样的 task tuple。历史移动轨迹不包含足够信息，应用 `task_valid=0` 显式告诉 critic 未知。
- 不为每个 MD 建独立 GRU 网络。动态 session 数据太少且不能泛化；应该共享参数、分开 hidden。
- 不使用全 UAV 混合 replay 来声称“完全分散训练”。若以后要提高样本效率，可把“离线共享预训练后本地冻结/微调”作为单独消融并如实标注。
- 不直接上 Dreamer/DVRL。它们的完整 latent world model 和 RL joint objective远超过只补齐 MD mobility state 的需求。

## 2026-08-19：为何默认不制造人工缺失

当前 replay 的自然标签定义为“同一 receiver 再次收到同一 MD session 时，用此前真实已接收历史和当前 age 预测当前 mobility feature”。在固定独立及时接收概率 `q` 且没有 lifetime/episode 截断时，重观测间隔满足 `P(age=k)=q(1-q)^(k-1)`；critic 在随机缺包时实际查询的 age 风险也具有相同几何结构。因此 age=1 占比高通常是实际风险权重，而不是自动成立的数据偏差。

真正需要监控的是有限 lifetime/episode 造成的右删失，以及接收概率随 UAV 距离和状态变化带来的选择性删失。默认均匀人工 mask 不保证修复它们，反而会改变物理信道数据分布。当前主设置因此是 natural-only，并记录：

- `md_prediction_replay_label_age{1,2,3,4plus}_fraction`：当前有界 replay 的自然监督分布；
- `md_prediction_rollout_query_age{1,2,3,4plus}_fraction`：本 rollout 真正进入 surrogate critic block 的消费分布；
- `md_prediction_prequential_rmse_m` 与 `md_last_obs_prequential_rmse_m`：预测器 ready 后，在同一自然重观测事件上的在线比较。

label 是 replay 窗口口径，query 是本 rollout 消费次数口径；两者不能直接当成同一批样本计数。若实际常用 query-age 桶几乎没有自然 label，下一步优先按真实 query-age 对自然样本加权/分桶；使用 simulator truth 给删失轨迹补标签必须明确标成 privileged-training 消融。

## 相关论文和成熟实现给出的边界

- [R2D2 (ICLR 2019)](https://openreview.net/forum?id=r1lyTjAqYX) 专门指出 replay 中旧 recurrent state 会因参数变化而 stale，并使用序列 replay 与 burn-in 缓解；这里采用“保存原始短序列、用当前参数重算 hidden”的同一原则。
- [GRU-D](https://arxiv.org/abs/1606.01865) 把 observation mask 与 time interval 明确作为 missing-time-series 输入；本设计的 `task_valid/record_valid/age` 遵循同样的缺失模式显式化原则。
- [UNREAL](https://arxiv.org/abs/1611.05397) 证明辅助任务可使用短 replay，但也报告单纯 reconstruction 可能改善早期学习却损害最终策略；因此必须把 prediction metric 与 RL return 分开验收，不能只凭预测 loss 宣称 DC-PPO 改进。
- [DreamerV3 官方实现](https://github.com/danijar/dreamerv3) 展示了从 replay 训练 world model、再训练 actor--critic 的成熟分层方式；它支持“模型数据可跨 episode 重用”，但完整 world model 对当前 critic-only MD 补全属于过度设计。
- [Google DeepMind Reverb](https://github.com/google-deepmind/reverb) 提供 FIFO、priority 与 sequence replay 等成熟数据结构；当前 MD lifetime 仅 12，使用本地有界 deque 已足够，不值得增加分布式 replay 依赖。
- [MAPPO 官方实现](https://github.com/marlbenchmark/on-policy) 和 [SB3 Contrib RecurrentPPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) 都把 PPO rollout/epoch 边界当作 recurrent on-policy 更新边界。这支持本设计在整轮 PPO batch 内冻结 estimator，而不是在 PPO epochs 中途改变 critic 输入生成器。

## 论文与代码对齐状态

| 契约 | 论文 IV-D | 2026-08-13 代码 |
|---|---|---|
| 每 UAV 本地参数 | 是 | 已对齐：五套 predictor/optimizer |
| 每 MD 独立 hidden/age/lifetime | 是 | 已对齐：receiver/session memory bank |
| consecutive missing 不读真值 | 是 | 已对齐：hidden 不更新，head 读 age |
| 只预测 persistent mobility | 是 | 已对齐；task 使用 unknown mask |
| surrogate 只给 critic | 是 | 已对齐；actor obs 不变 |
| 重观测监督 | 是 | 已对齐 |
| 预测和 value 真正联合反传 | 文字看似是 | 实际/推荐不是；应在论文澄清交替独立优化 |
| 历史序列训练 | 未定义 buffer | 已补齐：跨 episode 有界 raw-sequence replay |

另有三项实验合同需要在正式论文版本中统一：论文写 MSE，代码采用更稳健的 Smooth-L1；论文 `lambda_p=0.1` 看似联合损失，代码是两个 disjoint optimizer；论文表中的 running-sum 轮数 30 应更新为当前已确定的 50。它们不是代码崩溃，但如果论文方法、表格和实际 args 不一致，就会成为复现性问题。

## 必做验证与消融

先验证预测器本身，再看 RL 总回报。否则 RL 结果无法区分“预测不准”和“critic 不需要预测”。

1. **Estimator offline/online metrics**：position RMSE(m)，speed MAE(m/s)，direction cosine error；按 age=1、2、3、4+ 分桶；与 constant-velocity、last-observation 比。
2. **因果性**：对丢包 token 修改当前模拟器真值，重构输出必须不变。
3. **局部性**：只给 UAV 0 replay，只有 predictor 0 参数变化。
4. **RL ablation**：reliable direct、unreliable zero+metadata、last_obs+metadata、constant-velocity+metadata、MD-GRU+metadata；所有组相同信道随机种子、速度环境和 PPO预算。
5. **必要性判断**：若 MD-GRU 的 age-binned RMSE 优于 last_obs/constant-velocity，但 RL 无显著收益，应报告为“估计模块有效但 critic 对该误差不敏感”，不能宣称算法整体提升。

## CPU 验证记录

测试前检查到约 32.6 GB 可用内存、CPU 22--45%，三个现有 `train_mec.py` 长任务，GPU 为 23863/24576 MiB。全部新测试都设置 `CUDA_VISIBLE_DEVICES=-1`、单线程 BLAS，并通过 Python 的 legacy `--cuda` store-false 参数确认 `device='cpu'`；没有停止或修改任何现有进程。

- `py_compile` 通过。
- 15 个 MD reconstruction 测试函数的全部 Python 断言在隔离 CPU 环境执行通过；为绕开已确认的解释器退出期 native cleanup crash，测试 harness 在断言全部完成后用 `os._exit(0)` 结束，得到明确 `EXIT=0`。这证明 Python 状态机/梯度更新断言通过，但不等价于完整 PPO 训练通过。
- 正式 Fixed600 启动器确认输出 `cuda=False`, `device='cpu'`, actor obs `211`, critic state `1355`。
- 两个现有 PyTorch 环境都在 CPU 原生后端产生 Windows `0xC0000005`；`marl` 环境中的独立 8x8 GRU 脚本也能复现，因此没有把一次完整 PPO update 标记为通过。

在开始正式 CPU 训练前，应新建隔离环境并安装与 Windows/Python 匹配的稳定 PyTorch CPU wheel，然后重跑全 pytest 和 one-update smoke。由于当前 GPU 已满，本轮没有切换到 GPU 验证。
