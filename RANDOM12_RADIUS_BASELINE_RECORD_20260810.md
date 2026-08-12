# Random12-600 Noise=3 Radius Baseline Record

记录日期：2026-08-10。该记录冻结昨天远端3090上完成的 Random12 R0/R260/R520 三组实验，作为后续 `1+4` 出生模式下不同通信半径（不同 `k`）的对比基线。

## 实验协议

- 地图：`600 m × 600 m`
- 热点布局：`episode_template12_600_200`；200×200 小区域和 400×400 大区域，从四个角落产生 12 个有序组合
- MD 出生：`md_arrivals_per_region=[1,4]`，每个生成时刻候选 1 个小区 MD + 4 个大区 MD；`n_GUs=60` 为动态 MD 槽位上限。基线持续时间为 10，因此满覆盖稳态最多约 50 个活动 MD；本次 `md12` 才对应最多 60 个活动 MD
- MD 速度：`mean_velocity=3.0`，`md_velocity_init_std=0.6`，初始化倍率 `[0.4,1.6]`
- MD 持续时间：基线为 `md_lifetime_min=10`、`md_lifetime_max=10`
- 噪声：`advantage_mode=per_agent_noise`，`noise_scale=3.0`
- 通信半径：`neighbor_distance=neighbor_R ∈ {0,260,520}`
- 观测上下文：`episode_layout_context=true`，`episode_layout_context_units=meters_v2`
- actor message：`actor_message_mode=task_summary`、`actor_message_contract=absolute_raw_v2`、`actor_message_pool=receiver_gated_sum`
- 训练：seed=2、64 rollout threads、episode_length=400、num_env_steps=60,000,000

R0/R260/R520 三组之间只改变通信半径参数；R0、R260、R520 的完整 `args.json` 已保存于：

- `analysis/random12_radius_baseline_20260810/args/`

## 代码可复现性

远端项目没有 Git 元数据，因此用远端实际文件的 SHA-256 和源码快照固化版本。源码快照位于：

- `analysis/random12_radius_baseline_20260810/remote_code_snapshot/`

关键文件哈希：

| 文件 | 远端实际使用版本 SHA-256 |
|---|---|
| `onpolicy/envs/mec/mec.py` | `AE27D20C7688A0E487CEAFDCCB0DED052EA6B8289117929EC3A572999BFAF3E1` |
| `onpolicy/scripts/train/train_mec.py` | `3835655392EA4E24F743E08759665691F3F59008BBD56C328F1B8EF68A16B3EA` |
| `onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py` | `154296F13C869973730E7CA48F3DF0D7B55F94907CB2716C1FEF2CDDA1912041` |
| `onpolicy/envs/mec/vec_normalize.py` | `7DE9FE2902E2C01A1A84832ABD1CB25A6E6FB0DE224EDF892A83A09F91AE3DFB` |
| `onpolicy/algorithms/utils/mlp.py` | `61434E98FE9BD92C32B7C1664DFDA6C817657E5A7EF0C1254D4BC6BF5E8CA046` |
| `onpolicy/scripts/train/run_random12_600_dcppo_noise3p0.ps1` | `25174080EC068F0AAF74E64426A18DB41D3FEE13FC39C98DE3AEDBD216B9370D` |

本地当前 Git HEAD 为 `716e080ac9cfaeed95a336ae0e14c8e696ed4c89`，但工作区有大量未提交实验产物和改动，不能整体覆盖远端。逐文件比较 `onpolicy` 源码后，只有 4 个共享文件不同；其中 Random12 相关的训练入口、actor/critic、归一化和 Random12 启动脚本一致。`mec.py` 的差异仅是本地为固定布局 `episode_template4_600_200` 增加候选随机数流，昨天 Random12 的 `episode_template12_600_200` 不经过该新增分支。

因此，新的 Random12 实验继续使用远端现有代码，不同步本地固定布局改动；这样可以保证与昨天三组基线的代码路径完全一致。

## 新一组唯一变量

新实验仍使用上述全部协议，仅改为：

```text
md_lifetime_min=12
md_lifetime_max=12
```

其余参数、代码路径、seed、布局、噪声和通信半径保持不变。新实验名称采用 `..._md12_...` 后缀，避免覆盖基线结果。

## 2026-08-10 远端 md12 实验启动记录

启动脚本：`onpolicy/scripts/train/run_random12_600_dcppo_noise3p0_md12.sh`。远端实际使用的脚本 SHA-256 为：

```text
78793048EECA2538088CF96432A7A601779EED5432D1D70BD9725E6682D486F
```

远端3090已按错峰顺序启动以下三个实验：

| 实验 | 通信半径 | 远端日志 |
|---|---:|---|
| `dcppoR0_random12_600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_seed2_60m_20260810` | 0 | `training_logs/dcppoR0_random12_600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_seed2_60m_20260810.log` |
| `dcppoR260_random12_600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_seed2_60m_20260810` | 260 | `training_logs/dcppoR260_random12_600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_seed2_60m_20260810.log` |
| `dcppoR520_random12_600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_seed2_60m_20260810` | 520 | `training_logs/dcppoR520_random12_600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_seed2_60m_20260810.log` |

逐字段比较新实验与昨天对应半径的 `args.json`，每组只有三项差异：`experiment_name`、`md_lifetime_min: 10 → 12`、`md_lifetime_max: 10 → 12`。因此 R0/R260/R520 内部仍只改变通信半径；相对昨天基线只改变 MD 持续时间。

启动核验时三组主训练进程均为存活状态并已进入训练，未见启动异常；GPU 显存约 `22315/24576 MiB`。新实验的 `args.json` 位于各自远端结果目录的 `run1/args.json`。
