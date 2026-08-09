# RandomLayout700-v2 设计与诊断记录

## 目的

检验隐藏服务区域和通信拓扑是否会产生可观测的通信收益。该环境不把热点位置加入观测，热点只在 episode reset 时采样一次，静态 episode 内不跳变。

## 新增代码

- `hotspot_layout_mode=episode_template12`：大区域从四个角落采样，小区域从其余三个角落采样，共 12 种有序布局。
- `five_uav_start_layout=line`：`(100,70),(200,70),(300,70),(400,70),(450,550)`。
- `five_uav_start_layout=staggered`：`(100,70),(200,70),(300,140),(400,140),(450,550)`，作为几何消融。
- 原有 `episode_template4`、`fixed_legacy` 和 `episode_moving_template4` 保持不变。

## 初始通信拓扑

对默认 `line` 和 `staggered` 两种起点，R520 都只让上方 UAV 直接连接下方最近两架；R600 在 reset 初始位置已经连接全部五架。700m 地图对角线约 990m，因此 R780 不等价于整个 episode 始终全连接，R1000才是全图连接基准。

## 首轮实验约束

第一轮只测试 `line` 起点和 12 布局，关闭 `uav_reset_curriculum`，保持当前 Actor/Critic、奖励、消息池化和 PPO 参数不变。建议半径为 R0/R260/R520/R1000，并使用相同 seed 与相同布局随机流。`staggered` 只作为后续几何消融，不与布局随机化和 curriculum 同时引入。

## 判断指标

除 `system_performance_true_all_GUs`、completion、admission 外，记录布局分层曲线、初始大区域是否被观测、第一次跨组消息时刻、直接消息接收比例、通信图连通分量和共识残差。只有在固定起点评估和布局分层结果均支持时，才讨论 `R0<R260<R520<R1000`。

## 本地验证

`tests/test_dynamic_md.py`：11 passed；`git diff --check`：passed。已有两个 `delay_true_coverd_GUs` 除零 warning 未改变。
