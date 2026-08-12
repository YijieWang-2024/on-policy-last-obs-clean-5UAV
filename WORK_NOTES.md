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
