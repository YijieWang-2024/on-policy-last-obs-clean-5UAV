# 最近实验工作日志

> 覆盖时间：2026-07-22 至 2026-08-05。本文只记录可复用的工程事实、正式实验和决策；启动失败、smoke 与重复导出不会伪装成独立实验。

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

## 2026-08-01：旧算法定义核对与统一半径实验

- 完成 Q2、DC-PPO-260、DC-PPO-520、IPPO 曲线对比，确认当前Q2/IPPO使用纯局部优势 `A_i`，而历史Fig.4所谓MAPPO/DC-PPO实际都启用了 `A_i+A_mean+topology noise`；算法名称相同不代表Actor更新信号相同。
- 重新核对 ValueNorm：当前新实验通过反向命令行开关实际关闭 ValueNorm、开启 `ret_norm`；当前 `returns-value_preds` 不存在量纲错误。发现每UAV独立回报缩放会改变跨UAV优势平均的严格等权含义，后续新实验改用 shared return normalization。
- 修正共识残差实现：R0严格self-only，使用整体向量残差倍率，避免逐元素比例无界；保留episode rollout结束后基于最终拓扑做统一优势共识的实验定义。
- 运行归一化残差 R0/R260/R1000 三路，至约54.7M得到稳定反序，否定“仅靠半径相关随机扰动自动获得单调性能”的设想。

## 2026-08-02：优势、Critic和共识机制矩阵

- 本地启动 Mixed-consensus R0/R260/R1000；超过53M后仍为 R0>R260>R1000，overall与fixed-reset同方向，主要差异伴随任务完成率下降，而非单纯接纳率变化。
- 远程9001完成Local-`A_i` R260/R1000：53.376M时只差`+0.004%`，证明critic信息范围扩大本身几乎没有策略收益。
- Pure-consensus R1000比R260低约16.34%，确认纯团队优势存在严重信用分配困难；两路提前停止并保留模型/日志。
- ego-query与mean-pool critic在R1000只差约0.22%，读出结构不是当前主要瓶颈。
- 远程9012启动R520 K=1/5/50；34M早期K增加有约1%的弱收益，但K1/K50继续到62M后K50反低0.66%，不能形成“轮次越多越好”的稳定结论。K5已停止。
- externality beta=.025/.05/.10在8--9M显示beta越大越差，三路停止，放弃继续扫beta。
- 根据诊断把主线从“修改优势/噪声”转向“Actor执行时获得通信消息”。

## 2026-08-03：Actor消息与receiver-gated候选

- 在远程9001运行均值池化九路：低速task消息A0/A1/A2、zero消息A3/B1、R1000 geometry B2，以及高速V0/B3/V1；全部使用纯`A_i`，把优势融合从Actor消息问题中隔离。
- 29M共同点，mean-pool task序列为R260−R0 `−0.77%`、R1000−R260 `+1.44%`，尚未形成完整顺序。
- zero消息序列表明critic-only的R1000相对R260低`11.76%`；但R1000 task相对zero提高`14.72%`，证明Actor消息是R1000通信收益的关键。
- R1000内容消融显示geometry相对zero提高`13.99%`，完整task仅在geometry上再提高`0.64%`；邻居相对几何是主要可行动信息。
- MD速度提高到10m/s没有形成目标排序，R1000相对R260仍低`5.39%`，不再扩大高速环境网格。
- 实现receiver-conditioned独立sigmoid gate和固定分母消息池化；修复新模块初始化污染全局RNG的问题，通过51项全套测试和真实环境smoke。
- receiver-gated G0/G260/G1000在共同21.9904M首次同时满足目标：overall相邻提升`+0.225%/+1.040%`，fixed-reset为`+0.262%/+1.013%`。
- 同半径比较表明gating主要改善R1000：gated−mean为R0 `−0.54%`、R260 `−0.07%`、R1000 `+1.77%`，与“均值池化稀释远端有效消息”的假设一致。
- 冻结本地与两台远程设备的29条正式实验事件文件，生成11组共同step True60配对图和完整CSV审计；入口为`paper_artifacts/diagnostics/dcppo_radius_review_20260803/README.md`。

## 2026-08-03 当前决策

- 主候选：receiver-gated Actor task/geometry communication，优势保持纯`A_i`。
- 保留：G0/G260/G1000继续到50M并做固定起点评估。
- 暂停扩展：优势混合、externality beta、人工噪声、MD速度、更多任务消息字段、更多共识轮次网格。
- 成功标准：50M固定评估仍满足两个相邻正差，然后至少3个独立训练seed中均值同序且至少2/3 seed同序。
- 理论边界：当前结果说明Actor可执行消息有用，不证明critic信息量或共识精度与最终回报天然单调；论文理论与实验定义仍需据此对齐。

## 2026-08-03 跨设备刷新与停机

- 冻结并按共同TensorBoard step比较本地mixed三路、9001 mean-pool九路、receiver-gated三路和9012共识轮次实验。
- 发现receiver-gated在21.9904M的完整排序未维持到36.1728M；当前为`R260 < R0 < R1000`，只保留其跑到50M。
- 确认本地mixed为反序；mean-pool半径比较无序；R1000 geometry显著优于zero，但task不优于geometry；v10无益；K50不优于K1。
- 停止本地mixed三路、9001 mean-pool九路和9012 K1/K50，保留全部日志/模型/结果；未触碰receiver-gated三路和其他用户进程。
- 新证据入口：`paper_artifacts/diagnostics/all_devices_refresh_20260803/README.md`。

## 2026-08-03 RandomLayout-700正式环境与公平预检

- 将正式regional dynamic-MD环境参数化到600/700m；700m采用中心对称十字起点，热点仍为episode间随机、episode内固定。
- 同一起点、同seed分别运行600/700的10k geometry + 2k paired heuristic预检。
- 700m得到R260-R0 `+7.579%`，但R1000-R260仅`+0.158%`且57.05% episode为正，未通过0.5%/70% gate。
- 旧700m非对称起点的`+0.639%/79.3%`不能视为稳健环境收益。
- 58项测试通过；800-step正式训练smoke通过。根据预先门槛未启动长训练。

## 2026-08-03 MovingHotspot700 communication-radius gate

- Added `episode_moving_template4`: hidden symmetric 175x175 / 400x400 MD birth regions move continuously along two 300 m route edges per 400-slot episode (3 m/s center speed). Existing admitted MDs are neither translated nor removed when the birth region moves; their birth-time reflection bounds remain frozen.
- The hotspot state is diagnostic-only and is not added to actor or critic observations. Added tracking diagnostics: `moving_hotspot_small_nearest_uav_distance` and `moving_hotspot_large_mean_four_uav_distance`.
- Local and remote validation: `tests/test_dynamic_md.py` 9 passed; 800-step moving R400 smoke completed one PPO update without Traceback/NaN/OOM.
- The initial 10M launch was stopped after the budget was judged too short; its logs were preserved. Clean 30M/64-worker/seed2 runs are active on 9001 from `/data/home/tanglanProf_user02/wyj/Projects/on-policy-movinghotspot700-20260803`:
  - R0: PGID 100806, GPU5
  - R200: PGID 22722, GPU5
  - R400: PGID 61580, GPU6
  - R600: PGID 100439, GPU7
- The four runs are matched except `neighbor_distance=neighbor_R=0/200/400/600`. At the intended 1+4 deployment these radii correspond to isolated, disconnected local groups, connected non-complete, and complete communication graphs.
- Do not interpret the first sub-0.6M-step snapshot; the 30M matched curves and fixed-protocol evaluation are the decision evidence.

## 2026-08-04 至 2026-08-05：RandomLayout700-v2、MovingHotspot700 与布局复核

- 新增周级交接文档：`RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md`。后续新智能体先读该文件，再读本时间线和 `HANDOFF.md`。
- RandomLayout700-v2 完成两训练 seed 的 R0/R260/R520/R1000 约 50M 比较；共同约49.024M、tail100 True60 为 `379.8/413.8/429.6/431.4k`，曲线首次形成目标顺序，但 R520→R1000 只有约0.43%，仍需固定评估和多 seed。
- RandomLayout700-v2 的 R1000 轨迹不是“两个区域都覆盖”的充分证据：三个 seed12 checkpoint 诊断 episode 的 small admission 均为0；四布局复核得到 `0/0.875/0.9675/1.0`。后续必须分区域统计，不能只看 True60。
- MovingHotspot700 的三 seed比较只能到共同21.9904M（seed2中断）；True60为 `394.9/424.3/418.2/417.4k`（R0/R200/R400/R600）。通信相对R0有效，但半径不单调，停止把该组当作严格排序证据。
- Diag4无 curriculum在46.464M呈 `R0>R260≈R520<R1000`，R0最高；FixedDiag约23–24M四个半径都在260k左右并向小区挤压，均不支持通信半径故事。12-layout UAV curriculum只到17.8944M，记录为早期快照，不排名。
- 本周可复用的正面结果仍是：Q2主干稳定；Actor几何消息比critic-only更有证据；receiver-gated比mean-pool更能避免远端消息稀释，但其完整排序仍需50M固定评估确认。
- 停止/不再扩展：纯共识、外部性beta、共识轮次、MD高速、mean-pool大网格、FixedDiag。未删除这些实验的日志和模型。

### 本周新证据入口

- `paper_artifacts/randomlayout700_v2_final_20260804/`
- `paper_artifacts/randomlayout_moving_20260804/`
- `analysis/remote9001_randomlayout700_uavcurriculum_vs_diag4_20260805/`
- `analysis/remote9001_randomlayout700_live_20260804/`

- 周总结中的曲线与轨迹可视化入口：`RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md` 第 9–10 节。

## 2026-08-05：七类环境统一曲线与轨迹交接

- 周总结已扩展为 `RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md` 第 11–13 节，统一覆盖 RandomLayout700-v2 Full template12、Full template12+UAV curriculum、MovingHotspot700、700m Diag4、600m Diag4 training-free preflight、FixedDiag 与 Subset29。
- 更正旧快照：UAV curriculum 并非只到 17.9M；远程归档已到共同 `49,996,800`，tail100 True60=`464.0/472.3/463.7/456.1k`（R0/R260/R520/R1000），未形成目标排序。
- 新增 Subset29 归档证据：共同 `39,705,600`，tail100=`399.5/458.1/458.4/455.7k`，R0远低于通信组但R1000略低于R520，且四布局 R1000 评估出现 `593.7/90.9/55.3/395.5k`，记录为训练布局子集泛化失败。
- 600m Diag4 被明确标注为 training-free heuristic preflight，不是 PPO 训练；它支持 R260 相对 R0 的信息价值（+5.514%），但 R1000 相对 R260 仅 +0.207%，不能写成三档 PPO 结果。
- 所有新增曲线/轨迹链接已用 `Test-Path` 检查，文档 `git diff --check` 通过；停止的实验日志和模型未删除。
- 追加最终曲线刷新目录 `paper_artifacts/final_curve_refresh_20260805/`：六组训练曲线均绘制到每个运行自己的最后 TensorBoard 事件，并用 `final_curve_endpoints.csv` 保存最终 tail100；RandomLayout700、Diag4、curriculum、Subset29 不再被共同步数末点截断。MovingHotspot seed=2 仍明确显示为中断，不做外推。

## 2026-08-06：重排 8 月 1–5 日周级实验总结

- 读取三条任务线从 8 月 1 日开始的对话，并核对本地保存的机制诊断、训练曲线、最终 checkpoint 评估、轨迹和速度反事实产物。
- 重写 RECENT_EXPERIMENT_SUMMARY_20260801_20260805.md：把机制诊断、正式环境训练和训练后策略评估拆开；矩阵统一加入目的、数据结果、结论/处理和完成状态；旧版重复的第 9–13 节不再继续追加。
- 更正 Q2 10 episode True60 为 617,738 ± 4,096；更正 12-layout UAV curriculum 已完成约 50M；明确 600m/700m preflight 不是 PPO 结果，MovingHotspot seed=2、Subset29、K1/K50 等属于未完成或诊断状态。
- 当前交接判断：RandomLayout700-v2 Full template12 无 curriculum 两 seed 是候选主线，但必须先做 small/large 分区审计和消息反事实；不再仅凭总 True60 或早期曲线宣布通信半径定律。
- 本日期之后启动的 v_max=10/20/30/40 运行不纳入本周完成结果。

## 2026-08-09: Random12-600 input-v2 implementation

- Preserved the pre-change communication experiment baseline as `44b7575`.
- Implemented the explicit `episode_template12_600_200` mode rather than changing the legacy 700m `episode_template12` semantics.
- Versioned the observation contracts to prevent same-shape old checkpoints from silently receiving new semantics.
- `absolute_raw_v2`: absolute/raw message fields, normal `ob_norm` for fields 0-8, raw binary mask at field 9.
- `meters_v2`: raw meter-valued episode layout boundaries followed by normal `ob_norm`.
- Updated frozen normalization, full-message masking, and distance counterfactuals for both v1 and v2.
- Added the parameterized `run_random12_600_dcppo_noise3p0.ps1` launcher for R0/R260/R520/R780.
- Targeted regression: 53 tests passed; one-step environment/normalizer smoke test and PowerShell parser checks passed.

## 2026-08-12：近两周参数分歧收敛与速度实验代码审计

- 全量对照当前工作树、速度启动器、12 份 local/remote `args.json`、TensorBoard 事件、训练日志 shape、近两周噪声/半径/环境总结。
- 权威详细结果：`FINAL_ENV_ALGORITHM_PARAMETER_AUDIT_20260812.md`；速度图协议：`analysis/fixed600_speed_3seed_20260812/audit.md`。
- 当前冻结合同：Fixed600-200 index0，1+4 arrivals，MD12，MD 3m/s，5 UAV 固定起点，无 curriculum，R520，noise3 residual，50 consensus iterations，Spatial Cartesian actor，absolute_raw_v2 task-summary receiver gate，ego-query attention critic，shared return normalization，v30。
- 60M 完整双 seed tail100：v10=485,025.8，v20=496,693.0，v30=562,608.0；v30 比 v20 高约 13.27%。2026-08-13 01:30 刷新时 v40 三 seed 共同到 21.7856M，仍未满 60M，保持 pending。
- 输入维度已沿代码路径复核：Actor obs 251；resource head 211；critic token 211；R520 k-hop critic state 1055；动作 62=2 flight+20 association+20 bandwidth+20 compute。
- 解释边界：R520 是 Actor/Critic/共识的联合选择；noise3 在 R520 下有效噪声接近零；不能宣称 actor-only 半径最优或 noise3 本身最优。
- 溯源缺口：远端无 Git 元数据；速度 args/事件已保存，但没有本轮 speed-specific 完整源码 SHA manifest。

## 2026-08-12：Actor message 开关单变量审计

- 找到 `analysis/fixed600_no_actor_vs_actor_20260811/` 的远端 R0/R260/R520 no-message 冻结事件和 args，以及 `run_fixed600_200_dcppo_R520_noise3p0_md12_no_actor_message.sh`。
- R520 与本地 actor-message v30/seed2 在 args 层只有 `actor_message_mode` 和实验名不同；共同 40.8832M tail25 差 +0.017%，tail100 差 +0.452%。
- 限制：no-message 快照未到 60M、只有 seed2、不是同 checkpoint 的测试时 mask。因此记录为“近等价的训练诊断”，不升级为 Actor message 无用的最终结论。
- `disabled` 将 Actor message 维度从40删到0，Actor obs 251→211；Critic 和 R520 共识图保持不变。

## 2026-08-13：Actor-message/速度主线归档

- 用 GitHub 插件确认账号 `YijieWang-2024` 对目标仓库具有 push/admin 权限。
- 以 `ACTOR_MESSAGE_SPEED_ARCHIVE_20260813.md` 建立当前主线索引；冻结环境/算法合同、速度选择、no-message 结论边界和后续最小 gate。
- 重新运行 `analysis/fixed600_speed_3seed_20260812/plot.py`，只读刷新当前事件：本地 seed2 v10/v20/v30 到 21.7856M/21.7344M/17.7408M；远端冻结 v40 到 23.1168M/22.5536M/21.7856M。
- `tests/test_dynamic_md.py` 通过 17 项；仅保留两个已有除零 RuntimeWarning。
- Git 归档不包含模型、TensorBoard 原始事件、`training_logs/` 或大批历史中间产物。
# 2026-08-13 unreliable-dataplane speed-base sync

- Common base: `b265913`; old unreliable head: `483db29`; speed archive baseline: `fa53f3a`.
- Integration rule: speed branch owns environment/Actor/Critic/advantage contracts; unreliable Type-S/Type-A and MD reconstruction are orthogonal overlays.
- Added shared roster ordering, layout-aware Type-S reconstruction, independent channel/advantage RNGs, current advantage-mode compatibility, MD-GRU timing metrics, and manifest-bound shared GRU checkpoints.
- Added `EpisodeLength`/`MDLifetime` to the generic launcher and a Fixed600-v30-R520-MD12 unreliable wrapper.
- Fixed smoke-only integration bugs: string `save_dir` path joining and avoiding GRU checkpoint publication in `last_obs` mode.
- Verification: `95 passed`; four 2-step runner smokes passed. Test artifacts under `onpolicy/scripts/results/` remain ignored.
## 2026-08-13 MD-GRU estimator audit

- Read paper IV-D lines 920--1164 and pseudocode 1272--1295 against `md_state_reconstruction.py` and separated MEC runner.
- Confirmed critic-only path: Type-S loss selects direct/surrogate sender token; actor observation is unchanged; reconstructed NumPy state severs PPO gradient into estimator.
- Replaced cross-UAV shared predictor with receiver-local models and optimizers while retaining per-session online hidden.
- Replaced per-rollout stale-hidden reservoir with bounded cross-episode raw-sequence replay; reset online beliefs at episode boundaries only.
- Predictor update is Smooth-L1 supervised, recomputes hidden with current parameters, gradient-clips, and remains separate from PPO.
- Runner order is now `compute returns -> PPO actor/critic -> predictor supervised update -> next rollout`, so the estimator remains frozen for each on-policy batch.
- After PPO `after_update()`, refreshed predictor state is written to buffer slot 0 (the next-rollout initial state), not stale terminal slot -1.
- Added `-CPUOnly` launcher plumbing. Resource audit before testing: 32.56 GB free RAM, 30--45% CPU, three active train_mec jobs, GPU 23863/24576 MiB. No active job was stopped or modified.
- CPU test evidence: all 15 reconstruction test functions passed their Python assertions in the isolated CPU environment; an `os._exit(0)` harness after all assertions produced explicit `EXIT=0` and avoided the known interpreter native cleanup crash. A standalone 8x8 GRU script outside the repo reproduces `0xC0000005` under `CUDA_VISIBLE_DEVICES=-1`; `torch.autograd.grad` reported finite gradients before native-process failure. Treat this as logic-level validation, not an accepted end-to-end PPO training pass.

## 2026-08-13 speed-baseline compatibility re-audit

- Added `UNRELIABLE_SPEED_COMPATIBILITY_AUDIT_20260813.md`, grounded in the executed speed/no-message args, Git ancestry, and current call paths.
- Confirmed both reliable Metropolis and unreliable running-sum use reset-before terminal UAV positions copied from `info`, not auto-reset positions.
- Clarified that speed-compatible `per_agent_noise` uses running-sum only for a residual-scaled perturbation of local advantage; paper-IV-D direct running-sum is `pure_consensus`.
- Exposed `AdvantageMode`, `NoiseScale`, `CommunicationDistance`, and `RunningSumRounds` in the fixed launcher; disabled Actor-message runs now still persist pool/contract args explicitly.
- Fixed the zero-denominator per-agent residual endpoint, the GRU speed codec for the formal `[0,5] m/s` range, and the separated eval 5/8-field interface with frozen normalization statistics.
- CPU-only isolated assertions: 17 MD reconstruction/GRU, 11 reliable consensus, 7 unreliable communication, and 2 eval/normalization tests passed; launcher argv probes passed for both speed-compatible per-agent-noise and paper-style pure-consensus modes. No GPU or full PPO run was started.
- Split GRU replay capacity/train budget/readiness into `32768/512/512` and batched the variable-length sequence unfold; this prevents full-replay training after every PPO update and prevents a one-sample predictor from replacing last-observation.
- Corrected Type-S reception-rate accounting to count only geometry-defined receiver/sender attempts and to reset counters per episode. This was superseded on 2026-08-19 by hard `d_com` gating inside the shared sampler, so neither Type-S nor Type-A now retains an internal all-pairs draw.

## 2026-08-19 reliable-mainline semantic sync

- Audited `agent/dcppo-runtime-optimization` commits from 2026-08-14 through 2026-08-19. Runtime-relevant deltas are threshold/deadline filtering (`73845de`) and generic UAV/resource support (`898bdb6`/`0bcb887`); the remaining commits are launchers, plots, blob normalization, docs, or history reconciliation.
- Ported one configurable association threshold through ACTLayer and MEC execution, the optional offload deadline prefilter, generic UAV counts/explicit starts, heterogeneous per-UAV bandwidth and CPU capacities, and agent-count-aware evaluation snapshots.
- Preserved all pre-existing unreliable Type-S/Type-A, receiver-local GRU replay/training, terminal-position consensus, and advantage processing changes in the dirty worktree.
- Added a generic six-UAV unreliable launcher and exposed threshold/resource/PPO controls in the formal five-UAV launcher.
- Verification used `CUDA_VISIBLE_DEVICES=-1`: Python compile passed, four PowerShell scripts parsed, 5/6-UAV argv probes passed, targeted tests were 74/74, and the complete suite was 107/107 with two existing empty-statistic warnings. No training process or GPU workload was started.

## 2026-08-19 physical-range running-sum and GRU data audit

- Three independent reviews converged on the same communication fix: nominal range adjacency must be explicit, out-degree cannot be inferred from successful receptions, and running-sum must divide by `out_degree+1` rather than UAV count.
- Added one-source `P_c <-> d_com` link-budget resolution with a 5 m consistency guard, defaults 520 m / 1.1809658836 W / 10 dB / 50 rounds, strict R0 no-edge support, and finite-value validation.
- Type-A packet rate now uses only nominal directed range edges. Non-edge forged receptions are filtered inside the consensus primitive without mutating the caller's array.
- Retained natural reobservation replay for MD-GRU. Added distinct replay-label and rollout-query age distributions plus matched prequential GRU/last-observation RMSE; synthetic missingness remains out of the default method.
- Corrected the per-agent-noise zero-denominator endpoint. A consensus estimate displaced from the global mean now yields residual 1 even if that agent's local advantage initially equalled the mean.
- Resource check before tests: 39.57/63.91 GB RAM free, reported CPU load 0%, no Python process. With CUDA hidden, 48 targeted tests passed and the full suite passed 117 tests with two pre-existing warnings; no long training was launched.
- After a second resource check (39.56 GB free, no Python process), ran one 2-slot/1-worker CPU-only formal-launcher smoke. PPO completed one update and published actor/critic/normer files, checkpoint manifest, and receiver-local `md_gru_shared.pt`; resolved args were d_com=520, Pc=1.1809658836179866, K=10, H=50, per_agent_noise, md_gru, cuda=false.

## 2026-08-20 receiver-major Type-S reconstruction optimization

### Purpose

- Remove the remaining per-slot Python loops, repeated bank copies, and small-GRU launch overhead without changing the algorithm contract: Actor uses true local observations; Type-S loss only changes critic state reconstruction; Type-A loss only changes running-sum advantage consensus.

### Setting

- Controlled paired benchmark against clean `25a55bd`: 64 environments, 5 receivers, 20 packet slots, 10 valid records per sender, 10% off-diagonal state-packet loss, 190 timed slots; GRU hidden size 64 on CUDA.
- End-to-end gate: Fixed600-B0, seed 2, 64 workers, episode length 400, R520/Pc=1.1809658836W/K=10dB, H=50, per-agent-noise scale 3, Actor message disabled, PPO epoch 4. A separate user-owned 60M reliable job remained alive throughout, so absolute FPS includes concurrent load.

### Results

- Controlled `last_obs`: 18.133 -> 5.260 ms/slot (71.0% less time; 3.45x throughput).
- Controlled prediction-ready `md_gru`: 39.624 -> 12.444 ms/slot (68.6% less time; 3.18x throughput).
- End-to-end `last_obs`: 879 FPS; MD reconstruction 2.046 s of a 28.874 s rollout/update cycle. The previous same-style run was 832 FPS and 3.588 s reconstruction.
- End-to-end prediction-ready `md_gru`: 821 FPS; MD reconstruction 4.612 s, GRU supervised update 1.226 s, total pre-save cycle 30.537 s. `md_gru / last_obs = 0.934`; the previous same-style GRU run was 705 FPS and 10.148 s reconstruction.
- The predictor trained on natural re-observations only: agent 0 used 517 target samples and 128 validation targets at step 76,800; prediction remained enabled. Full regression: 121 passed, with two pre-existing MEC empty-statistic warnings.

### Analysis

- Kept a single receiver-major dense belief bank while preserving receiver/session isolation. Expiration and allocation now run once per slot with a free-slot stack; received observations are deduplicated and ingested once; direct and missing blocks are assembled directly into sender order.
- Kept five independent predictor parameter sets and five natural session buffers. Online GRU update/prediction stacks those independent weights and executes one receiver-batched `bmm`; an elementwise equivalence test locks it to the original five-module calculation.
- Type-S payload now piggybacks on the environment step response, avoiding a second subprocess RPC and avoiding transfer of the discarded reliable `share_obs`. `last_obs` returns a bank view instead of copying the full feature store.
- Rejected two changes after measurement: CPU shadow inference was slower (17.828 vs 15.229 ms/slot in its gate), and one unified session replay buffer was slower than five receiver-local buffers (10.723 vs 10.347 ms/slot). Both were reverted.

### Next Steps

- The runtime gate is satisfied; no further structural rewrite is required before a full run. Compare reliable and proposed performance only with matched PPO epoch and all other frozen arguments, then use the logged phase timers to detect any long-run drift.

## 2026-08-20 rollout-local MD-GRU target-budget training

### Decision

- Removed cross-rollout replay from the default training path. Each receiver keeps only the current rollout's dense MD sessions; the buffer is cleared after the supervised update while online causal hidden states are refreshed from the updated receiver-local predictor.
- Shuffle whole `(environment, MD session)` trajectories once, partition them without replacement, and preserve complete session prefixes for BPTT. Each of the five receiver-local predictors takes up to 10 optimizer steps, each containing approximately 2,048 natural re-observation targets; validation uses the next disjoint approximately 2,048-target session set.
- `md_gru_epochs` is fixed to 1. Legacy replay-size/session-mini-batch arguments are accepted only for launcher compatibility and no longer control training.

### Verification and CUDA timing

- Python compile, PowerShell AST parsing, 28 targeted MD reconstruction tests, and the complete 124-test CPU suite passed; the full suite retained only two pre-existing MEC empty-statistic warnings.
- Ran Fixed600-B0, seed 2, 64 workers, 76,800 steps, R520/Pc=1.1809658836 W/K=10 dB/H=50, per-agent-noise scale 3, Actor message disabled, PPO epoch 4, and `md_gru_target_batch_size=2048`, `md_gru_batches_per_rollout=10`, `md_gru_epochs=1`. The user's 60M reliable reference remained alive throughout, so the absolute 818 FPS is a concurrent-load result.
- At step 76,800, receiver 0 had 59,282 trainable sessions and 541,811 natural targets. Ten disjoint batches used 2,250 sessions / 20,549 targets (3.793%); validation used 2,049 targets. The five predictors are trained independently, so the configured budget is per receiver, not divided across UAVs.
- One rollout/update cycle took 31.506 s: policy collection 14.657 s, environment step 5.359 s, online Type-S MD reconstruction 4.848 s, normalization 1.933 s, PPO/advantage update 2.505 s, and all five supervised predictor updates 1.861 s. Receiver-0's own ten optimizer steps used 0.313 s; sampling and its two validation passes together used 0.048 s.
- Relative to the immediately preceding 512-target receiver-major MD-GRU benchmark, train targets per receiver increased from 517 to 20,549 (39.75x), while all-five-predictor update time increased from 1.226 s to 1.861 s and total pre-save time from 30.537 s to 31.506 s. FPS was 821 versus 818 under the same concurrent-load class. Large target batches therefore use the GPU efficiently; the supervised update is not the dominant runtime bottleneck.

## 2026-08-20 experiment inventory and last-observation control

- Local formal runs confirmed in `onpolicy/scripts/results/mec/mappo/`: `F600_B0_reliable_reference_seed2_60m_cuda` uses `communication_mode=reliable` and `state_reconstruction=zero`; `F600_B0_proposed_unreliable_mdgru_tb2048x10_seed2_60m_cuda` is the currently active unreliable MD-GRU run. The earlier `F600_B0_proposed_unreliable_mdgru_seed2_60m_cuda` directory is retained as a separate prior formal run.
- Added `run_local_f600_b0_control_unreliable_last_obs_seed2_cuda.ps1`, freezing the same Fixed600-B0/R520/K10/H50/per-agent-noise/PPO-4 contract and changing only `state_reconstruction=last_obs`, with experiment name `F600_B0_control_unreliable_last_obs_seed2_60m_cuda`.
- Removed 24 Codex CPU/performance/profile result directories from the active result root. They were moved to `D:/wyj/Projects/.codex-transfer-trash/unreliable-results-20260820` for recovery rather than touching any formal run.
- Remote access was completed read-only through the historical 3090 workflow using `test@114.212.117.24`. The complete unreliable-zero result was copied and manifest-verified into `onpolicy/scripts/results/mec/mappo/F600_B0_unreliable_control_zero_seed2_60m_cuda`; remote `run2` is the 59.9808M-step run and `run1` is an incomplete 0.0768M duplicate.
- Four-way figure snapshot: `analysis/fixed600_four_way_20260820/plot.png`, with exact plotted rows in `curves.csv`, renderer in `plot.py`, and audit in `audit.md`. Complete endpoints are reliable zero 556,634.75 and remote unreliable zero 558,398.94 (+0.317% for unreliable zero). At the latest refresh the local MD-GRU snapshot reaches 2.0736M steps and last-observation reaches 1.2032M; neither is a final-performance comparison yet.
- The figure is marked `PASSED_WITH_WARNINGS`: current local runs are still in progress, and reliable reference has `critic_md_metadata=false` whereas the unreliable runs have it enabled, so this is not a strict one-variable ablation.

## 2026-08-20 four-run configuration explanation

- Full `args.json` comparison and checkpoint-shape audit are in `analysis/fixed600_four_way_20260820/config-audit.md`. The common Fixed600-B0/PPO contract is matched across all four runs, but Reliable zero has `critic_md_metadata=false`; Unreliable zero, MD-GRU, and last-observation have it enabled.
- This is a real critic-architecture difference, not just a logging flag: Reliable zero critic input is 211 features, while all three unreliable/current critics are 271 (`+3*20` metadata features). Actor input shapes are identical.
- The unreliable launcher always passes `-CriticMDMetadata`; current non-zero reconstruction also forces metadata in `train_mec.py`. The older zero runs retain legacy GRU compatibility fields (`epochs=4`, batch/sample legacy values) while current local runs use rollout-local target-budget fields; those fields are inert for zero but show the saved runs were not serialized under one canonical argument schema.
- In `per_agent_noise`, reliable and unreliable modes use different raw-consensus paths (Metropolis versus packet-loss running-sum) to set the Gaussian perturbation magnitude. Around 5--11M, reliable residual means are often larger and unreliable consensus correlation slightly higher. Therefore an early unreliable lead can be a stochastic/regularization effect, amplified by the metadata confound; it is not evidence that packet loss is intrinsically beneficial.
- Strict next ablation: rerun Reliable zero with `critic_md_metadata=true` and canonical current compatibility fields, or disable metadata in both zero modes if retaining that contract. Use at least three seeds before claiming an ordering.

## 2026-08-21 interrupted local runs and checkpoint resume

- After the local reboot, no Proposed/last-observation trainer process remained. Both formal `run1` checkpoints are intact and manifest-verified: MD-GRU `12,569,600` source steps (`episode_index=490`, includes `md_gru_shared.pt`); last-observation `13,721,600` source steps (`episode_index=535`).
- The standard 60M/64-worker/400-step loop actually ends at `59,980,800` steps because the runner floors the number of rollouts. Remaining warm-start budgets are therefore `47,411,200` for MD-GRU and `46,259,200` for last-observation.
- Added `-ModelDir` plumbing to `onpolicy/scripts/train/run_fixed600_200_unreliable_dataplane.ps1`. Resume commands should use the same experiment name plus `-ModelDir ...\run1\models`; the launcher creates `run2` and preserves the interrupted `run1`.
- Restore is a warm start: actor/critic, normers, and MD-GRU predictor are loaded; PPO optimizer moments, RNG/environment state, rollout buffers, and online GRU hidden/session state are not. Do not pass another 60M after restore, or the cumulative budget will overshoot.
- Follow-up diagnosis is recorded in `analysis/fixed600_four_way_20260820/diagnosis.md`: the 5--11M unreliable-zero lead is primarily explained by the extra 60-dimensional critic metadata (`271` vs `211` critic token features), with a secondary difference from reliable Metropolis versus unreliable running-sum advantage-noise paths. The lead averages `6.683%` in 5--11M but only `0.388%` over 20--60M; it is not evidence that packet loss is intrinsically beneficial.

## 2026-08-20 Reliable zero 对 Fixed600-200 历史基准复核

- 对齐了当前 `F600_B0_reliable_reference_seed2_60m_cuda/run1` 与历史 `dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed2_60m_20260814/run1`；两份 args 的 192 个可比字段全部一致，唯一共同字段差异是实验名。当前新增的通信/GRU/resource 字段在 Reliable zero 中是默认/停用状态。
- 原始 TensorBoard 指标同图见 `analysis/fixed600_reliable_vs_historical_baseline_20260820/plot.png`，数据/脚本/审核/配置说明在同目录。历史曲线只到 57.8304M，未做外推。
- 公共区间末点当前 541,996.06、历史 550,950.31，当前低 8,954.25（1.63%）；5--11M 历史平均高约 30,625.93（约 6.28%），20--40M 平均差约 -0.51%，后段基本重合。
- args 没有解释该单 seed 的中段差异；历史运行没有保存 Git SHA。两次运行来自不同时间的代码/运行环境，故在同一 commit、同一设备上复跑前不能判定为代码回归。

## 2026-08-21 至 2026-08-23：不可靠通信实验跨设备完整账本

### 审计口径

- 本次只依据真实存在的 `args.json`、TensorBoard `events.out.tfevents.*`、checkpoint/eval 产物和 2026-08-23 22:13 的进程/GPU 快照；只写过 launcher、空事件头和 smoke 不计为正式结果。
- 训练曲线统一读取 `agent0/system_performance_true_all_GUs` 原始 scalar，不平滑、不插值、不外推。名义 60M 在 64 workers、400-step rollout 下的实际完整终点是 59,980,800。
- “本地”指 Windows `D:/wyj/Projects/`；“远端”指 `test@114.212.117.24` 的 `/home/test/wyj/Projects/`。远端训练进程会被 `setproctitle` 改名为 `mappo-mec-*`，仅 `ps | grep train_mec.py` 会漏报，必须同时核对 `nvidia-smi` PID。

### A. 本地 unreliable 正式训练

统一协议：Fixed600-200 index 0、严格 1+4、MD12/v3、UAV vmax30、固定 UAV 起点、无 Actor-message、meters_v2、deadline filter OFF、psi=0.5、R520/Pc=1.1809658836W/Kc=10dB/H50、per-agent-noise scale3、local reward、Spatial Cartesian Flight Actor、completion-priority、ego-query attention critic、separated 5-policy PPO、shared return normalization、clip0.15/gamma0.99/PPO epoch4、64 workers/60M。MD-GRU 为 receiver-local `2048×10×1` natural-target 监督更新。

| 实验目录/run | seed | 重构 | curriculum | 终点 | latest | tail20 | 状态 |
|---|---:|---|---|---:|---:|---:|---|
| `F600_B0_proposed_unreliable_mdgru_tb2048x10_seed2_60m_cuda/run2` | 2 | MD-GRU | off | 59.9808M | 552,427.69 | 552,225.89 | 完成，从头重跑 |
| `F600_B0_control_unreliable_last_obs_seed2_60m_cuda/run2` | 2 | last obs | off | 59.9808M | 489,965.31 | 489,771.21 | 完成，从头重跑 |
| `F600_B0_proposed_unreliable_mdgru_tb2048x10_seed32_60m_cuda/run1` | 32 | MD-GRU | off | 59.9808M | 493,602.50 | 492,798.01 | 完成 |
| `F600_B0_control_unreliable_last_obs_seed32_60m_cuda/run1` | 32 | last obs | off | 59.9808M | 494,187.72 | 493,493.62 | 完成 |
| `F600_B0_proposed_unreliable_mdgru_curriculum_p0p7_10m_25m_tb2048x10_seed2_60m_cuda/run1` | 2 | MD-GRU | p0.7 10M--25M | 32.0256M | 545,460.81 | 545,211.80 | 主动停止 |
| `F600_B0_control_unreliable_last_obs_curriculum_p0p7_10m_25m_seed2_60m_cuda/run1` | 2 | last obs | p0.7 10M--25M | 54.0416M | 547,507.31 | 551,480.01 | 主动停止 |

参数逐字段校验：

- MD-GRU seed2 与 last-observation seed2 仅差 `experiment_name/state_reconstruction`；
- MD-GRU seed2 与 seed32、last-observation seed2 与 seed32 均仅差 `experiment_name/seed`；
- 两条 curriculum 实验仅差 `experiment_name/state_reconstruction`；
- 无 curriculum MD-GRU 与 curriculum MD-GRU 仅再差 `uav_reset_curriculum` 和 `uav_reset_curriculum_schedule`。

重启边界：旧 seed-2 `run1` 的 TensorBoard 终点为 MD-GRU 12.5696M、last-observation 13.6960M，checkpoint 分别到 12.5696M/13.7216M。新 `run2` 的 `model_dir` 为空，因此两条都是从零开始的新 60M，旧 run1 不能拼接到新曲线。

### B. MD-GRU 预测质量与策略结果

| seed | GRU validation RMSE tail20 | last-observation RMSE tail20 | 相对降低 | GRU/last-observation 策略 tail20 |
|---:|---:|---:|---:|---:|
| 2 | 0.799 m | 1.910 m | 58.2% | 552.226k / 489.771k（GRU +12.75%） |
| 32 | 0.817 m | 1.951 m | 58.1% | 492.798k / 493.494k（GRU -0.14%） |

- estimator 层面的结果在两个训练 seed 上一致：MD-GRU 确实比保持 last observation 更接近下一次自然重观测位置。
- RL 层面的结果不一致：seed 2 在约 30M 后突破到 550k 档，seed 32 的两种重构都停在约 493k。不能用 seed 2 单条好曲线证明 GRU 稳定提升；两 seed tail20 简单均值虽为 522.512k 对 491.632k（+6.28%），但该均值被一个高 seed 主导。
- 预测 loss 数值本身约为 `1.1e-3`，不应跨特征归一化合同直接解释；位置 RMSE 和 prequential GRU-vs-last-observation 才是可读诊断。

中期行为评测：`run2` 的 26.7776M checkpoint 冻结后，在固定 layout index 0、episode seeds 0/1/2 上得到 True-all 544,403 / 542,913 / 557,552（均值 548,289），完成率 98.81%/98.49%/98.22%，状态及时接收率 86.99%/86.11%/86.73%。产物位于 `.../run2/eval_3episodes_latest_step26777600/`。该评测验证了部署结构，但不是最终 checkpoint 或独立训练重复。

### C. Curriculum 对照

在 MD-GRU curriculum 的实际停止点 32.0256M 进行公共步数比较：

| 曲线 | 公共终点 tail20 | 公共终点 latest |
|---|---:|---:|
| 历史 Fixed600-200 reliable baseline | 547,808.46 | 546,778.06 |
| Proposed MD-GRU + curriculum | 545,211.80 | 545,460.81 |
| unreliable last-observation + curriculum | 546,614.98 | 547,820.12 |

三条 tail20 最大差不到 0.5%。这不是“MD-GRU 好到可以提前停”，而是 curriculum 使 reliable、MD-GRU 和 last-observation 都容易学到高性能部署，从而失去识别邻居预测必要性的能力。对应重绘脚本为 `analysis/fixed600_curriculum_three_way_20260822/plot.py`；主消融继续关闭 curriculum。

### D. 8 月 21 日跨界完成的远端 PPO 敏感性实验

以下均在可靠主线、R520、per-agent-noise scale3、Actor-message disabled、无 UAV curriculum、deadline filter ON 下运行；它们不是当前 deadline-filter OFF 的严格对照。

| 变化 | 设备/run | 终点 | tail20 | 记录 |
|---|---|---:|---:|---|
| clip=0.05 | remote `...clip0p05.../run1` | 59.9808M | 533.110k | 完成 |
| clip=0.30 | remote `...clip0p3.../run1` | 59.9808M | 548.438k | 完成 |
| gamma=0.90 | remote `...gamma0p90.../run2` | 59.9808M | 443.755k | 完成，末段退化 |
| gamma=0.95 | remote `...gamma0p95.../run1` | 59.9808M | 559.350k | 完成 |
| PPO epoch=1 | local `run1` | 36.2752M | 537.331k | 本地中断副本 |
| PPO epoch=1 | remote `run2` | 59.9808M | 546.419k | 从头重跑并完成；remote run1 仅空事件头 |
| PPO epoch=10 | remote `run1` | 59.9808M | 501.054k | 完成 |

这组未提供足够证据替换 clip0.15/gamma0.99/PPO epoch4，尤其 gamma0.95 的单 seed 小差异不能凌驾于主协议一致性。

### E. 可靠侧 advantage/noise/radius 诊断

共同环境已经改为当前 Fixed600-200/MD12/v3/vmax30/meters_v2/no Actor-message/deadline OFF/psi0.5/no curriculum/64 workers/60M。重构均为 `zero`；这些实验只诊断优势合同和可靠通信半径。

#### E1. R0 local-mean noise sweep（远端，均已停止）

| noise scale | 终点 | tail20 |
|---:|---:|---:|
| 0.0 | 19.2256M | 460.116k |
| 0.4 | 20.5056M | 434.137k |
| 0.8 | 19.2256M | 430.494k |
| 1.0 | 20.9664M | 443.976k |
| 2.0 | 20.3008M | 429.021k |
| 4.0 | 19.8912M | 419.509k |

本轮只说明在约 20M 的 R0 `local_mean_per_agent_noise` 筛选中，附加噪声没有改善 scale0；所有曲线都早停，不能报告为 60M 排名。轨迹网格 `analysis/remote_localmean_checkpoint_eval_20260822/trajectories_grid_6noise_3seeds.png` 是同 checkpoint 的环境-seed 评估，不是多个训练 seed。

#### E2. per-agent-noise scale0 半径组

| 设备 | R | 终点 | tail20 | 状态 |
|---|---:|---:|---:|---|
| remote | 0 | 57.3184M | 482.939k | 停止，接近预算但未完成 |
| remote | 260 | 41.1392M | 543.579k | 停止 |
| remote | 520 | 40.6784M | 546.270k | 停止 |

R260/R520 在该单 seed 协议下明显高于 R0，但三条终点不齐；R260 与 R520 很接近。原始快照、配置审计和图见可靠主线 `analysis/radius_noise0_comparison_20260822/`。

#### E3. local-mean scale0 半径组

| 设备 | R | 终点 | tail20 | 状态 |
|---|---:|---:|---:|---|
| remote | 0 | 19.2256M | 460.116k | 早停快照 |
| local | 260 | 43.7504M | 497.139k | 停止 |
| local | 520 | 43.6992M | 499.172k | 停止 |

R0 的训练长度与 R260/R520 不匹配，因此该组三条不能形成最终半径排序。

#### E4. 当前运行的两个 per-agent-noise 半径组

快照时间为 2026-08-23 22:13；下表为异步当前点，后续必须刷新并按共同 step 比较。

| 设备/scale | R0（step/tail20） | R260（step/tail20） | R520（step/tail20） | 状态 |
|---|---|---|---|---|
| local, scale=2 | 32.0768M / 473.292k | 32.0256M / 540.821k | 29.0048M / 528.863k | 三路运行中，PID 10984/40652/53056 |
| remote 3090, scale=0.8 | 34.9952M / 476.917k | 33.9200M / 541.782k | 34.1760M / 539.544k | 三路运行中，GPU PID 2183268/2184235/2184654，各约 6.2 GiB |

同日还存在两个旧 R0 单路筛选：remote per-agent scale0.8 到 18.3552M/tail20 465.453k，scale2 到 17.8944M/tail20 500.808k，均已停止；不要与当前三半径新 run 混为同一连续训练。

远端事实边界：`/home/test/wyj/Projects/on-policy-unreliable-dataplane` 在 8 月 21 日以后没有新的事件文件；上述远端任务全部来自可靠主线 `/home/test/wyj/Projects/on-policy-last-obs-clean-5UAV`。远端当前训练并没有 MD-GRU。

### F. 产物与后续 gate

- 无 curriculum 五路图：`analysis/fixed600_four_way_20260820/`（当前脚本保留历史 Fixed600 baseline、两 seed MD-GRU、两 seed last-observation）；
- curriculum 三路图：`analysis/fixed600_curriculum_three_way_20260822/`；
- reliable deadline/curriculum 三基准：`analysis/fixed600_three_reliable_baselines_20260823/`；
- 可靠侧远端/本地 noise-radius 图：可靠主线 `analysis/remote_r0_localmean_noise_sweep_20260822/`、`remote_r0_noise_mode_comparison_20260822/`、`radius_noise0_comparison_20260822/`、`peragent_noise_radius_compare_20260823/`。

下一 gate：

1. 等当前 local scale2 和 remote scale0.8 三半径实验结束或到预先固定的共同步数，再冻结 event/args；不按当前异步末点排名。
2. MD-GRU vs last-observation 至少补一个固定的第三训练 seed，并对最终 checkpoint 使用共同 episode seeds 做确定性评测。当前两个 seed 只足以报告“预测 RMSE 改善稳定、RL 性能改善不稳定”。
3. 继续关闭 curriculum 做预测必要性消融；若论文希望报告 curriculum，只把它放在独立训练技巧/上界实验中。
4. reliable/unreliable 通信对比必须先统一 `critic_md_metadata` 和 critic token 维度；旧 reliable-zero 211 维与当前 unreliable 271 维不可作为严格通信单变量结论。
