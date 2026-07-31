# 5-UAV 动态用户场景近期实验总结（2026-07-25—2026-07-28）

> 后续进展：2026-07-29 至 2026-07-31 的 Q0/Q1/Q2 最终结果、动作审计和对比实验启动状态见 [`RECENT_EXPERIMENT_SUMMARY_20260729_20260731.md`](RECENT_EXPERIMENT_SUMMARY_20260729_20260731.md)。

> 更新时点：2026-07-28 22:20 左右
> 范围：最近四天围绕动态用户环境、strict 1+5 候选用户、MAPPO/DC-PPO、初始位置、warm-start、Cartesian 飞行动作、Spatial Flight Actor 和 Reset Curriculum 所做的正式实验、诊断实验、轨迹评估、失败启动与无效测试。
> 重要限制：除旧论文曲线外，本轮绝大多数训练只有 `seed=2`；轨迹图通常是固定起点、确定性策略、`eval_seed=100000` 的一个 episode，只适合诊断，不能单独作为统计结论。
> 可移植性说明：指向 `onpolicy/scripts/results/` 或仓库上级目录 `../` 的旧图片链接是本机溯源路径，clone 后不会存在；Git 可直接查看的阶段评估集中在 [`eval_live_20260728/`](eval_live_20260728/)，可复现论文诊断图集中在 [`paper_artifacts/diagnostics/`](paper_artifacts/diagnostics/)。

本文的“2/3”和“1/4”分别表示 episode 结束时左下热点/右上热点附近的 UAV 数量，不是成功率。`seed=2` 是训练随机种子；`eval_seed=100000` 是固定评估环境随机流。checkpoint 是训练某一步保存的参数，一次 episode 只是从该 checkpoint 做的一条评估轨迹。

## 1. 一页结论

最近四天的结果不能再概括成“算法就是学不会左下 1 架、右上 4 架”。更准确的结论是：

1. 旧初始位置下，GAE λ 为 0.95 和 0.98 的 DC-PPO 在 50M 时都停留在 **左下 2 架、右上 3 架**；λ=0.98 没有解决问题，综合性能反而更差。
2. 将下方四架 UAV 的初始位置改为 `(110,70)、(220,70)、(330,70)、(440,70)` 后，MAPPO 和 DC-PPO 的最终 50M checkpoint 都已经形成 **左下 1 架、右上 4 架**。
3. 从 50M checkpoint warm-start 再训练 50M 后，两种算法继续保持 1/4；DC-PPO 后半程改善尤其明显，说明 50M 对该任务通常不够。
4. Cartesian 动作消除了原极坐标动作的方向周期边界和默认左移倾向，但单独使用 Cartesian 在约 26.5M 时仍未摆脱 2/3，说明动作参数化不是根因。
5. E1（Cartesian + Spatial Flight Actor）在约 42.88M 的固定起点评估中仍为 2/3；E2（E1 + Reset Curriculum）在 43.08M 和 59.72M 的固定起点单 episode 评估中均出现 1/4。59.72M 时 True60 为 508.1k、接纳率为 89.88%、完成率为 92.70%，但仍不足以替代多 seed 统计。
6. 当前最有力的机制解释是：**覆盖外候选用户出生后立即消失，使名义上的 1:5 需求在训练初始阶段被可见性选择反转；固定起点和五套独立网络进一步造成角色锁定。**
7. 目前仍没有证明“右上四架的几何位置已经很好”。1/4 只是数量正确，四架之间仍可能高度重叠。
8. E1/E2 还存在一个代码混杂：开启 `spatial_flight_actor` 时不仅改变飞行网络，也改变了用户排序。因此，E2 相对 E1 的 curriculum 对照有意义，但 E1 相对原始 MAPPO 不能只归因于空间编码器。
9. 为拆分上述混杂，已启动 B0（Cartesian + distance-only 排序 + curriculum，标准 actor）和 B1（B0 + actor-only 邻居观测）。截至 22:20 两者约 32M，尚未达到论文结论门槛。

### 1.1 7月28日晚实时状态

| 运行 | 唯一变量/目的 | 当前训练步数 | 最近固定起点评估 | 当前判断 |
|---|---|---:|---|---|
| E2 | Spatial Flight Actor + curriculum | 约81.18M，运行中 | 59.72M，seed100000，1/4；True60 508.1k，Eq60 517.8k，Covered 511.9k，Admission .8988，Completion .9270 | 最强机制证据，但仍是单训练 seed、单 eval episode |
| B0 | distance-only 排序 + curriculum，标准 actor | 约32.74M，运行中 | 9.91M，seed100000，1/4；True60 441.3k，Admission .8046，Completion .9024 | 早期门控信号；用于判断 Spatial Actor 是否必要 |
| B1 | B0 + actor-only 邻居 UAV 相对位置 | 约32.08M，运行中 | 9.45M，seed100000，1/4；True60 424.2k，Admission .7904，Completion .8947 | 早期门控信号；目前不能声称邻居观测有益 |

三个训练进程在更新时间仍有新 TensorBoard/模型写入，未被清理或中断。B0/B1 的 9–10M 评估预算过小，不能与 E2 的 59.72M 结果作算法排名。

## 2. 实验场景与评价口径

### 2.1 旧动态 3×20 场景

- 每时隙产生 3 个候选用户；
- 每个用户寿命 20 个时隙；
- 稳态名义数量约为 60；
- 候选用户只有出生时处于 UAV 联合覆盖范围内才激活；
- 激活用户离开联合覆盖范围后立即消失，不再追踪。

### 2.2 strict 1+5 场景

- 左下矩形每时隙产生 1 个候选用户；
- 右上矩形每时隙产生 5 个候选用户；
- 寿命 10 个时隙；
- 稳态名义数量仍约为 60；
- 覆盖外出生的候选用户立即丢弃；
- active 用户离开覆盖立即退出系统。

### 2.3 本文常用指标

| 简写 | TensorBoard/环境指标 | 含义与注意事项 |
|---|---|---|
| True60 | `system_performance_true_all_GUs` | 将未覆盖候选按本地计算补入后得到的完整用户性能，便于和旧60用户量级比较 |
| Eq60 | `system_performance_equivalent_full_GUs` | 将当前动态人口等效补齐到60用户的记录指标；只记录，不参与训练奖励 |
| Covered | `system_performance` / `system_performance_coverd_GUs` | 当前实际覆盖/激活用户的性能，更接近现有训练回报所反映的对象 |
| Admission | `md_admission_ratio` | 候选用户被联合覆盖并激活的比例 |
| Completion | `complete_task_ratio` | 被处理任务中按要求完成的比例 |

不能只看一个指标：扩大覆盖通常提高 Admission 和活跃用户数，但会加剧带宽与计算资源竞争，可能降低 Completion 和单位用户收益。

## 3. 四天实验时间线

| 日期 | 实验 | 回答的问题 | 状态与结论 |
|---|---|---|---|
| 7月25日 | 3×20 动态 MAPPO/DC-PPO | 新动态环境能否训练、部署是什么样 | 有效诊断；两者测试大多呈 2/3，MAPPO资源性能更好 |
| 7月25日 | 同随机流的 3/2 vs 4/1 反事实 | 4/1 是否只是主观期望 | 有效诊断；强制4/1能明显提高活跃用户与完整系统指标，但不证明策略能自行学到 |
| 7月25—26日 | strict 1+5，DC-PPO λ=.95/.98，各50M | GAE时间尺度是不是原因 | 正式单变量对照；.95优于.98，两者均为2/3，停止继续扫 λ |
| 7月26日 | 修改四个下方初始位置，MAPPO/DC各50M | 固定初始几何是否造成探索盆地 | 关键有效实验；最终两者都学到1/4 |
| 7月26日 | UAV5 强制到 `(500,500)` / `(520,480)`，20 traces | 1/4内部位置是否仍有改进空间 | 关键反事实；True60改善，但训练reward优势弱，暴露目标错配 |
| 7月26—27日 | 50M模型warm-start再训练50M | 50M是否训练充分 | 有效；两者继续改善，DC-PPO改善更明显 |
| 7月27日 | Cartesian MAPPO | 极坐标动作是否是根因 | 只跑到约26.5M后中断；可作中期诊断，不能作最终算法排名 |
| 7月27—28日 | E1：Cartesian + Spatial Flight Actor | 分离空间决策是否能突破2/3 | 历史运行已停止并被消融取代；约43M固定起点评估仍为2/3，且存在用户排序混杂 |
| 7月27—28日 | E2：E1 + Reset Curriculum | 随机起点课程能否打破角色锁定 | 正在训练至约81.2M；43.08M和59.72M固定起点评估均出现1/4，是当前最积极证据 |
| 7月28日 | B0：Cartesian + distance-only排序 + Curriculum | 排序修正与curriculum能否在标准actor上复现1/4 | 正在训练；约32.7M，9.9M单episode仅作早期门控 |
| 7月28日 | B1：B0 + actor-only邻居观测 | 邻居几何信息是否带来额外收益 | 正在训练；约32.1M，9.4M单episode仅作早期门控 |

## 4. 轨迹演化：一步一步看发生了什么

轨迹图中圆点通常表示起点，叉号表示 episode 结束位置；灰色虚线表示两个用户矩形区域边界。图只是单个确定性测试 episode 的部署诊断。

部分早期评估脚本把图标题硬编码成了“MAPPO Test Episode”，所以个别DC-PPO轨迹图标题仍显示MAPPO；应以本节标题和图片路径中的实验目录为准，这只是绘图标签问题，不代表模型读取错误。

### 4.1 旧 3×20 动态场景：MAPPO 与 DC-PPO

MAPPO 在旧动态场景中的代表轨迹：

![旧3x20 MAPPO轨迹](paper_artifacts/progress_figures_20260728/dynamic_mappo_88550400.png)

DC-PPO 在旧动态场景中的代表轨迹：

![旧3x20 DC-PPO轨迹](paper_artifacts/progress_figures_20260728/dynamic_dcppo_78105600.png)

用途：确认动态环境可以运行，也暴露了两架 UAV 容易共同留在左下、右上部署与资源分配仍不充分。该场景不是 strict 1+5，不能用它直接证明后续场景中的1/4最优性。

### 4.2 strict 1+5，旧初始位置：GAE λ=.95 与 .98

λ=.95 的最终轨迹：

![strict1p5 DC-PPO GAE095](paper_artifacts/progress_figures_20260728/strict1p5_dcppo_gae095_50m.png)

λ=.98 的最终轨迹：

![strict1p5 DC-PPO GAE098](paper_artifacts/progress_figures_20260728/strict1p5_dcppo_gae098_50m.png)

两条轨迹都保留两架 UAV 在左下，最终仍为2/3。对应训练曲线：

![GAE095与098训练曲线](paper_artifacts/progress_figures_20260728/gae095_vs_gae098_training_curves.png)

结论：λ=.98增加的长时间优势传播没有解决空间探索，且带来更高方差与较低完成率；保留λ=.95。

### 4.3 修改初始位置后：从中期疑似2/3到最终1/4

这一阶段最容易被中期图片误导。约23M时仍能看到两架 UAV 停留或徘徊在左下附近：

![MAPPO约23M中期轨迹](paper_artifacts/progress_figures_20260728/mappo_newinit_23m.png)

到约34M，迁移角色进一步形成：

![MAPPO约34M轨迹](paper_artifacts/progress_figures_20260728/mappo_newinit_34m.png)

最终50M MAPPO 已形成左下1架、右上4架：

![MAPPO 50M最终轨迹](paper_artifacts/progress_figures_20260728/mappo_newinit_50m.png)

最终50M DC-PPO 同样形成1/4，但学习更慢、路径和最终资源性能不同：

![DC-PPO 50M最终轨迹](paper_artifacts/progress_figures_20260728/dcppo_newinit_50m.png)

这一步证明了算法和五套独立网络具备表达1/4角色分工的能力。它没有证明从任意不利起点都能稳定发现1/4。

### 4.4 warm-start到累计约100M

MAPPO warm后的最终轨迹：

![MAPPO warm到约100M](paper_artifacts/progress_figures_20260728/mappo_warm_100m.png)

DC-PPO warm后的最终轨迹：

![DC-PPO warm到约100M](paper_artifacts/progress_figures_20260728/dcppo_warm_100m.png)

两者均保持1/4。对应的拼接训练曲线如下。注意 warm-start 没有恢复完整 optimizer/RNG/rollout 状态，因此这里的横轴是“累计交互步数”的诊断拼接，不是严格连续100M复现实验。

![累计约100M等效60曲线](paper_artifacts/diagnostics/continuation100m_stitched_20260727_121020/equivalent60/plot.png)

![累计约100M覆盖用户目标曲线](paper_artifacts/diagnostics/continuation100m_stitched_20260727_121020/covered_objective/plot.png)

### 4.5 Cartesian MAPPO：去掉极坐标动作病态，但单独不够

Cartesian运行在约20.25M checkpoint的轨迹：

![Cartesian MAPPO中期轨迹](paper_artifacts/progress_figures_20260728/cartesian_mappo_20m.png)

该运行最终约26.55M后因工程问题终止，没有跑满预算。它只能说明“仅改变动作参数化不会立刻解决角色锁定”，不能据此否定Cartesian。

### 4.6 E1：Spatial Flight Actor，固定起点

E1在42.88M附近的固定起点轨迹：

![E1 fixed-reset轨迹](paper_artifacts/progress_figures_20260728/e1_fixed_42m.png)

E1仍为左下2架、右上3架，而且UAV在局部存在明显来回摆动。约46.62M时训练曲线tail100：True60约399.7k、Eq60约419.0k、Covered约390.9k、右上接纳率约71.5%、完成率约88.6%。

### 4.7 E2：Spatial Flight Actor + Reset Curriculum

E2在43.08M checkpoint、固定原始起点、确定性策略下的轨迹：

![E2 curriculum固定起点轨迹](paper_artifacts/progress_figures_20260728/e2_fixed_43m.png)

这张图是目前最关键的正面证据：UAV3从下方长距离迁移到右上，最终只有UAV2留在左下。固定起点评估结果：

| 指标 | E2 43.08M 单episode |
|---|---:|
| 分组 | 左下1、右上4 |
| True60 | 493,830 |
| Eq60 | 503,526 |
| Covered | 501,745 |
| 后半段平均active MD | 53.76 |
| 总接纳率 | 89.92% |
| 左下接纳率 | 99.50% |
| 右上接纳率 | 88.00% |
| 完成率 | 91.25% |

完整评估元数据位于 [`episode_summary.json`](../e2_fixedstart_seed100000_step43084800_20260728/episode_summary.json)。

截至本文更新时，E2约46.57M的训练tail100为：True60约451.6k、Eq60约464.8k、Covered约446.1k、右上接纳率约86.0%、完成率约89.4%。训练曲线混合了随机reset与固定reset，不能直接等同于固定起点部署成绩。

E2 在 59.72M checkpoint 的第二次固定起点、确定性 `seed=100000` 评估仍为1/4：True60 508,126，Eq60 517,803，Covered 511,873，后半段平均 active MD 54.04，总接纳率89.88%，左下99.50%，右上87.95%，完成率92.70%。精简后的评估文件保存在 [`eval_live_20260728/e2_s100000_59m/`](eval_live_20260728/e2_s100000_59m/)；这提高了结果的时间一致性，但没有增加独立 seed 数。

### 4.8 B0/B1：拆分排序、邻居观测与Spatial Actor

B0 保留 Cartesian 和 curriculum，显式使用 distance-only 用户排序，但回到标准 actor；B1 只在 B0 上增加 actor-only 的邻居 UAV 相对位置。两者因此比 E1/E2 更适合回答“Spatial Flight Actor 本身是否必要”和“actor 缺少迁移方向是否是瓶颈”。

9–10M 的固定起点单 episode 中，B0 与 B1 都出现1/4，但 B0 的 True60、接纳率和完成率暂时高于 B1。这个时点太早且只有一个 eval seed，正确表述只能是“两个接口均能工作并出现积极早期信号”，不能表述成“邻居观测无效”或“B0优于B1”。精简评估分别位于 [`b0_s100000_10m`](eval_live_20260728/b0_s100000_10m/) 与 [`b1_s100000_9m`](eval_live_20260728/b1_s100000_9m/)。

## 5. 主要数值结果汇总

以下主要为各训练后段均值；轨迹分组来自固定起点确定性测试。不同环境、checkpoint和warm协议之间只能作诊断比较。

| 实验 | 预算/状态 | 分组 | True60 | Eq60 | Covered | 接纳率 | 完成率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3×20 DC-PPO | 约100M完成 | 未按同协议复核 | 401k | 426k | 260k | .669 | .936 |
| 3×20 MAPPO | 约100M完成 | 未按同协议复核 | 426k | 451k | 304k | .658 | .963 |
| strict DC λ=.95，旧位置 | 50M | 2/3 | 416k | 439k | 280k | .690 | .932 |
| strict DC λ=.98，旧位置 | 50M | 2/3 | 400k | 421k | 260k | .705 | .910 |
| strict DC λ=.95，新位置 | 50M | 1/4 | 434k | 454k | 277k | .732 | .928 |
| strict MAPPO，新位置 | 50M | 1/4 | 468k | 487k | 310k | .741 | .951 |
| strict DC warm累计约100M | 完成 | 1/4 | 476k | 491k | 300k | .793 | .934 |
| strict MAPPO warm累计约100M | 完成 | 1/4 | 486k | 504k | 318k | .758 | .958 |
| E1 spatial fixed reset | 46.6M历史快照，已停止 | 最近eval 2/3 | 400k | 419k | 391k | — | .886 |
| E2 spatial + curriculum | 46.6M历史训练快照 | 最近eval 1/4 | 452k | 465k | 446k | — | .894 |
| E2 fixed-reset eval | 59.72M checkpoint | 1/4 | 508k | 518k | 512k | .899 | .927 |
| B0 fixed-reset eval | 9.91M checkpoint | 1/4 | 441k | 457k | 445k | .805 | .902 |
| B1 fixed-reset eval | 9.45M checkpoint | 1/4 | 424k | 441k | 433k | .790 | .895 |

## 6. 为什么名义1+5仍然难学到好布局

### 6.1 初始可见需求被选择性接纳反转

按当前初始几何做覆盖率估计：

- 左下候选覆盖率约95.8%，寿命10时稳态可见active约9.58；
- 右上初始只有一架UAV提供有限覆盖，覆盖率约12.3%；即使每时隙产生5个候选，稳态可见active也只有约6.15。

因此策略训练初期真正看到的不是1:5，而约为：

```text
左下可见active : 右上可见active ≈ 9.6 : 6.1
```

大量右上候选出生后直接消失，不进入actor观测、reward或DC-PPO的`A_mean`。

### 6.2 局部actor没有迁移方向

actor只看自身位置和当前覆盖用户；当前实验关闭邻居UAV观测，也没有候选拒绝计数或覆盖空洞信息。`A_local + A_mean`可以表示团队本次结果好坏，但不能告诉某一架UAV“应向东北移动”。

### 6.3 当前奖励与理想覆盖只有弱对齐

`system_performance_equivalent_full_GUs`只是日志；显式覆盖奖励没有进入训练。把一架UAV强制移动到更合理的右上位置时，True60和接纳率会提高，但实际训练reward的增量较弱，且更多接入用户会加重资源竞争、降低完成率。

### 6.4 1/4数量正确不等于四架几何位置优秀

目前没有覆盖重叠惩罚或边际联合覆盖奖励；碰撞惩罚只约束非常小的安全距离。右上4架即使高度重叠，也可能获得相近局部回报。

### 6.5 五套独立网络形成固定身份角色

为保持当前 DC-PPO/独立策略论文对照合同，本阶段不把“五套独立网络”改成共享网络；但固定起点意味着每个actor长期看到相似的局部数据，容易形成固定身份：某些网络始终留守，某些网络始终迁移。E2随机reset课程打乱了这种位置—身份绑定，因此是目前最符合证据的改进方向。

## 7. 哪些实验有用，具体有什么用

| 实验 | 价值 |
|---|---|
| λ=.95 vs .98 | 排除“GAE时间尺度不够”是主因，确定后续保留.95 |
| 修改初始位置 | 证明1/4不是动作空间或网络表示能力上的不可能 |
| MAPPO vs DC-PPO | 说明DC-PPO空间学习更慢，但不是结构性学不会 |
| warm 50M→约100M | 证明50M经常不足，特别是DC-PPO后半程仍在提升 |
| 3/2 vs 4/1反事实 | 证明4/1在完整性能和活跃人口上确有潜在优势 |
| UAV5目标位置反事实 | 证明数量达到1/4后，内部几何仍有提升空间；同时发现训练目标偏好较弱 |
| Cartesian | 排除极坐标动作是唯一原因，保留其作为更合理飞行动作参数化的可能性 |
| E1 vs E2 | 当前最有价值对照：支持固定起点角色锁定/状态分布是核心因素 |
| B0 vs B1 vs E2 | 解除E1/E2的排序混杂，分别检验标准actor、邻居信息与Spatial Actor；必须等同预算并做多seed评估 |

## 8. 没用、失败或不能形成算法结论的测试

这些目录和运行应保留记录，但不能作为独立实验点，也不能计入“多个种子”。

### 8.1 冒烟测试

- `smoke_e1_spatial_flight_no_neighbor_20260727`
- `smoke_e2_spatial_flight_curriculum_no_neighbor_20260727`

用途仅是验证代码能启动、维度和动作接口不报错。步数太少，不提供收敛或性能结论。

### 8.2 E1/E2首次启动、final、r32尝试

- `regional_dynamic_strict1p5_cartesian_spatialflight_e1_seed2_100m_20260727`
- `regional_dynamic_strict1p5_cartesian_spatialflight_e1_seed2_100m_final_20260727`
- `regional_dynamic_strict1p5_cartesian_spatialflight_e1_seed2_100m_r32_20260727`
- `regional_dynamic_strict1p5_cartesian_spatialflight_curriculum_e2_seed2_100m_20260727`
- `regional_dynamic_strict1p5_cartesian_spatialflight_curriculum_e2_seed2_100m_r32_20260727`

这些运行包含启动失败、资源配置尝试、OOM/页面文件/BrokenPipe或被后续正式r64运行取代的目录。这些旧目录不应再重启。当前正式运行的是 E2 的 `r64_20260727` 目录，以及 7月28日新增的 B0/B1 `r64_20260728` 消融；E1已被后续消融取代，不应把历史失败目录计作独立seed。

### 8.3 Cartesian约26.5M中断运行

`regional_dynamic_strict1p5_cartesian_mappo_seed2_100m_20260727`没有跑满100M。它可以作为“Cartesian单独不能快速解决2/3”的诊断，但不能与50M/100M模型作最终优劣排名。

### 8.4 重复轨迹导出

GAE实验中存在`eval_latest_checkpoint`、`eval_final_checkpoint`、`dcppo_eval_final_checkpoint`等多次导出；它们大多来自相同或相近checkpoint，不是独立实验。修改初始位置后13.6M、23M、27.8M、34.3M、40M、46.2M等轨迹也是同一训练的中间快照，只用于观察演化。

### 8.5 单episode图片

单个`seed=100000`轨迹可发现“2/3”“1/4”“来回摆动”“越界”等问题，但不能估计成功概率。E2当前1/4尤其需要10—20个共同eval seeds复核。

### 8.6 与旧论文Fig.4直接叠图

以下诊断图可以查看量级，但环境和统计协议不同：旧Fig.4是固定60用户、多算法、多seed并含历史后处理；新strict1+5多数只有一个训练seed。不能据此声明新算法优于或劣于论文曲线。

![旧Fig4与strict1p5诊断叠图](paper_artifacts/diagnostics/fig4_strict1p5_final_overlay_20260726_233002/plot.png)

### 8.7 E1相对基础MAPPO的归因混杂

开启`spatial_flight_actor`时，环境把覆盖用户改成纯距离排序；基础模型原本优先放置“本地无法按时完成”的用户，再按距离排序。这会同时改变关联和资源分配头看到的slot语义。因此：

- E2相对E1主要只差curriculum，仍是有意义的对照；
- E1相对原始MAPPO同时改变动作、网络、用户排序和部分mask处理，不能把差异只归因于Spatial Flight Actor；
- 下一版必须恢复资源策略原排序，给飞行编码器单独提供空间视图。

## 9. 当前最稳妥的阶段性结论

1. strict 1+5并非不可学习；1/4已经在多个最终checkpoint出现。
2. 现有证据支持一个候选解释：旧位置下的2/3可能共同受到初始可见需求反转、覆盖外需求不可见和固定身份角色锁定影响；尚未完成逐因素因果拆分。
3. λ=.98对照没有改善结果；尚无独立entropy对照，因此不能排除entropy，只是在完成结构门控前不优先扫描。
4. 50M容易得出过早结论；DC-PPO尤其需要更长预算。
5. Reset Curriculum是当前证据最强的改动，但尚缺多训练seed和多eval episode验证。
6. 当前训练目标对“多覆盖更多候选”的偏好弱于True60/Eq60指标表现出的偏好；这可能限制右上四架进一步摊开。
7. 下一步重点不是再找一个参数，而是建立相同checkpoint、相同eval seeds、相同指标的门控协议。

## 10. 下一步预注册计划

### 10.1 先完成现有E2/B0/B1

- 在50M和100M checkpoint分别评估；
- 固定原始起点；
- 使用相同的10—20个eval seeds；
- 统计1/4成功率、True60、Eq60、训练reward、上下区接纳率、后半段active、完成率、覆盖面积、重叠面积、两两距离和轨迹长度。

### 10.2 零训练成本部署oracle

在strict1+5上，用共同随机流比较：

- learned布局；
- 规则2/3；
- 规则1/4网格；
- 规则资源分配；
- 冻结learned资源头。

如果理想1/4只改善True60而不改善训练reward，应先处理目标错配；如果训练reward也显著提高但策略找不到，才继续研究探索与信用分配。

### 10.3 下一轮最多两个原子实验

在E2/B0/B1完成和门控评估结束前，不再启动新训练。之后最多并行：

1. Cartesian普通actor + Reset Curriculum；
2. Spatial Flight Actor + Reset Curriculum，但恢复原始资源用户排序。

暂时不再扫GAE、初始位置和更多DC-PPO参数；entropy尚未被独立排除，但在完成E2/B0/B1结构门控前不优先。

## 11. 结果与原始目录索引

### 正式或主要实验

- `onpolicy/scripts/results/mec/mappo/dynamic_mappo/run2`
- `onpolicy/scripts/results/mec/mappo/dynamic_dcppo/run3`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_dcppo_gae095_seed2_50m/run5`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_dcppo_gae098_seed2_50m/run3`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_init110_220_330_440_dcppo_gae095_seed2_50m/run1`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_init110_220_330_440_mappo_gae095_seed2_50m/run1`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_init110_220_330_440_dcppo_gae095_seed2_warm50to100m/run1`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_init110_220_330_440_mappo_gae095_seed2_warm50to100m/run1`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_cartesian_mappo_seed2_100m_20260727/run1`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_cartesian_spatialflight_e1_seed2_100m_r64_20260727/run1`
- `onpolicy/scripts/results/mec/mappo/regional_dynamic_strict1p5_cartesian_spatialflight_curriculum_e2_seed2_100m_r64_20260727/run1`

### 关键诊断数据

- `onpolicy/scripts/results/mec/mappo/strict1p5_dcppo_gae_comparison_seed2_50m`
- `paper_artifacts/diagnostics/continuation100m_stitched_20260727_121020`
- `paper_artifacts/diagnostics/fig4_strict1p5_final_overlay_20260726_233002`
- `.tmp_uav5_counterfactual_smoke_20260726/summary.json`
- 各MAPPO/DC-PPO运行目录下的`uav5_target_counterfactual_step34329600_seeds100000-100019_20260726/summary.json`
- `../e2_fixedstart_seed100000_step43084800_20260728/episode_summary.json`

---

本文档记录的是截至2026-07-28 22:20左右的阶段状态。E2/B0/B1仍在训练；达到共同预算并完成多episode、多训练seed固定起点评估后，应更新第5、9、10节，避免继续引用单episode中期结果作为最终结论。

## 12. 仓库清理与Git归档说明

本次只删除可再生且不承载唯一证据的文件：已被`.gitignore`覆盖的IDE配置、pytest/Python缓存、已在本文第7节吸收结论的临时反事实smoke目录，以及三个固定点评估中的逐帧PNG和重复checkpoint权重。共删除约133MB评估冗余；保留每次评估的`episode_summary.json`、`episode_data.npz`、CSV、总览图和`args.json`。

没有删除 `tests/`：这些测试覆盖动态MD人口约束、动作/log-prob契约、Cartesian速度投影、邻居观测隔离、Spatial Actor置换不变性和curriculum训练/评估隔离，属于当前实验可信度的一部分。也没有删除正在写入的训练目录与启动日志；它们被排除在Git归档之外。
