# 600 m 1+4 动态 MD 环境实验计划（2026-08-06）

## Purpose

在同一动态 MD 机制下，比较固定双热点、UAV reset curriculum 和四布局对角子集，检查是否能得到 `R0 < R260 < R520`。本记录只定义实验协议，不把结果预先当作结论。

## Common contract

- Map: 600 m × 600 m.
- Small region: lower-left 200 m × 200 m; one candidate per slot.
- Large region: upper-right 400 m × 400 m; four candidates per slot.
- Candidate lifetime: 10 slots; total arrivals 1+4 per slot.
- MD speed: initial `N(3, 0.6^2)` clipped to `[1.2, 4.8]`; subsequent Gauss–Markov speed clipped to `[0, 5]` m/s.
- UAV starts: `(110,180), (220,180), (330,180), (440,180), (400,400)`.
- Reward, action projection, resource allocation, MD admission/removal, PPO budget, and parallel-environment count are unchanged.
- `episode_layout_context` is disabled. The exact algorithm/message contract is copied from the active 9001 baseline and recorded in each `args.json`.

## Protocols

| Protocol | Layout/reset change | Radii |
|---|---|---|
| `fixed2hotspot_nocurr` | One fixed lower-left-small/upper-right-large layout; no UAV reset curriculum | R0/R260/R520 |
| `fixed2hotspot_curr` | Same layout; current verified UAV reset curriculum only | R0/R260/R520 |
| `diag4_nocurr` | `episode_template12` restricted to layout indices `[2,4,7,9]`; no curriculum; preserve the five specified starts | R0/R260/R520 |

Each configuration uses seed 2, 60M environment steps, 64 rollout environments, and a separate result directory. `episode_template4` must not be used if it overrides the specified UAV starts.

The curriculum protocol uses `p_random=0.8` for 0--10M steps, linearly decreases from 0.8 to 0 over 10--35M, and remains zero for 35--60M. The two no-curriculum protocols keep this probability at zero throughout.

## Evaluation and gate

Report common-step training curves and fixed-start post-training episodes separately for each protocol. In addition to overall `system_performance_true_all_GUs`, record small/large-region admission, completion, active-MD count, first-arrival/deployment time, and final UAV positions. A positive ordering in one early window is only a screening result; a final claim requires the same ordering in the fixed evaluation and replication seeds.

## Launch status

Remote 9001 launch is delegated to the existing remote-control task. It must first perform resource and smoke checks, then start runs without stopping or overwriting prior experiments; if only eight GPUs are available, the ninth run is queued independently.
