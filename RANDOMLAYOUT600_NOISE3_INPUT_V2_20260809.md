# 600m Random12 + noise 3.0 input-v2 experiment contract (2026-08-09)

## Decision

The new experiment family uses the 600m map, a 200m x 200m small birth region,
a 400m x 400m large birth region, and all 12 ordered non-overlapping corner
pairs. The requested R260/R520/R780 runs use one matched launcher and differ
only in `neighbor_distance` and `neighbor_R`.

`episode_layout_context` is enabled in the prepared launcher because the
requested policy should receive the episode-level region prior. It is not a
prerequisite for Random12. In fact, the completed RandomLayout700-v2 radius gap
was obtained without a completed layout-context result. Since exact layout
context is common to every UAV, it may reduce the extra value of communication.
If the three curves still overlap, the next diagnostic should be a matched
context-off ablation, not another radius sweep.

An R0 option is also accepted by the launcher. R0 is strongly recommended as a
same-contract no-communication control; without it, R260/R520/R780 only answer
whether larger positive radii differ, not whether communication helps.

## Locked environment and algorithm

- `hotspot_layout_mode=episode_template12_600_200`
- map: `600m x 600m`
- small region: `200m x 200m`; one candidate arrival per slot
- large region: `400m x 400m`; four candidate arrivals per slot
- 12 layouts: choose the large corner, then choose the small region from the
  remaining three corners
- five fixed UAV reset positions: `(110,180)`, `(220,180)`, `(330,180)`,
  `(440,180)`, `(400,400)`
- `advantage_mode=per_agent_noise`, `noise_scale=3.0`, `n_iterations=50`
- `actor_message_mode=task_summary`
- `actor_message_pool=receiver_gated_sum`
- `episode_layout_context=true`
- `episode_layout_context_units=meters_v2`
- `actor_message_contract=absolute_raw_v2`
- `D_min=200000`, `D_max=400000`
- 400-step episode and rollout; 60M environment steps by default

The radius is intentionally a combined communication intervention. In current
code, `neighbor_distance` controls actor-message delivery, k-hop critic tokens,
and the terminal-position Metropolis graph used by `per_agent_noise`.
`neighbor_R` is also matched to the radius for configuration consistency,
although its active observation path is the disabled legacy
`actor_neighbor_obs` block.

## Actor message v2

Each of the four sender slots contains 10 values:

| Index | Value before `ob_norm` |
|---:|---|
| 0-1 | sender UAV absolute `(x,y)` in meters |
| 2 | raw count of GUs in sender coverage |
| 3-4 | absolute GU-cluster centroid `(x,y)` in meters |
| 5-6 | raw mean GU velocity vector `(vx,vy)` |
| 7 | raw RMS spatial spread in meters |
| 8 | hard-task fraction in `[0,1]` |
| 9 | communication mask in `{0,1}` |

The first nine values enter the same running mean/std observation normalizer as
ordinary UAV and MD features. Only index 9 is restored to its raw binary value.
The network removes the mask before the message MLP and uses it to zero invalid
packets during pooling. This remains necessary because an all-zero unavailable
packet can become nonzero after normalization and after biased MLP layers.

The eight layout-context boundaries are emitted in raw meters and then enter
ordinary `ob_norm`. They remain outside the actor-only message block, so both
actor and critic receive the context.

## Compatibility

The old and new observations have the same dimensions but different meanings.
To prevent silent misuse of existing checkpoints, both representations are
versioned:

- old message checkpoints: `actor_message_contract=relative_scaled_v1`
- new experiments: `actor_message_contract=absolute_raw_v2`
- old layout context: `episode_layout_context_units=normalized_v1`
- new experiments: `episode_layout_context_units=meters_v2`

Frozen evaluation and distance-masking utilities understand both versions.

## Launch commands

From the repository root in PowerShell:

```powershell
& .\onpolicy\scripts\train\run_random12_600_dcppo_noise3p0.ps1 -Radius 260
& .\onpolicy\scripts\train\run_random12_600_dcppo_noise3p0.ps1 -Radius 520
& .\onpolicy\scripts\train\run_random12_600_dcppo_noise3p0.ps1 -Radius 780
```

Recommended same-contract R0 control:

```powershell
& .\onpolicy\scripts\train\run_random12_600_dcppo_noise3p0.ps1 -Radius 0
```

Do not start all long runs simultaneously on one GPU. Start one, wait for stable
GPU memory/utilization and at least one completed update, then start the next on
another available GPU or queue it.

## Comparison boundary

R260/R520/R780 (and optional R0) from this launcher are a fair radius family.
They are not directly comparable as a single-variable ablation against existing
fixed-layout curves or RandomLayout700-v2 because the layout distribution,
input contract, layout context, arrival rate, and noise treatment differ.
