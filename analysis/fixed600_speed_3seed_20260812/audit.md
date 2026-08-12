# Experiment Selection Audit

Metric: `agent0/system_performance_true_all_GUs`

## Formal 12-run matrix

| `v_max` | seed 2 | seed 32 | seed 42 |
|---:|---|---|---|
| 10 | local, `...vmax10_seed2_60m_20260812` | remote, `...vmax10_seed32_60m_20260811_retry1/run1` | local, `...vmax10_seed42_60m_20260811` |
| 20 | local, `...vmax20_seed2_60m_20260812` | remote, `...vmax20_seed32_60m_20260811_retry1/run1` | local, `...vmax20_seed42_60m_20260811` |
| 30 | local, `...vmax30_seed2_60m_20260812` | remote, `...vmax30_seed32_60m_20260811_retry2/run1` | local, `...vmax30_seed42_60m_20260811` |
| 40 | remote, `...vmax40_seed2_60m_20260812` | remote, `...vmax40_seed32_60m_20260812` | remote, `...vmax40_seed42_60m_20260812` |

The older local seed-2 runs dated `20260811` were not used. The duplicate remote v10 seed32 `run2` was not used. Remote v30 seed32 uses `retry2`; the earlier failed/empty retry event was not used.

## Parameter audit

All 12 `args.json` files were compared field by field. The only fields that differ are:

- `experiment_name`
- `seed`
- `v_max`

All environment, actor-message, normalization, advantage-noise, communication-radius, PPO, and training-budget fields are otherwise exactly equal. In particular: fixed `episode_template4_600_200`, map `600 x 600`, lifetime 12, arrivals `1+4`, R520, noise 3.0, `absolute_raw_v2`, layout context in meters, and `receiver_gated_sum`.

## Aggregation contract

At every union-grid environment step, each seed is linearly interpolated only inside its observed range. No extrapolation and no smoothing are used. The plotted line is the mean of all seeds that have reached that step. The band is plus/minus one population standard deviation when at least two seeds are available. A one-seed tail is plotted without an uncertainty band.

The frozen 2026-08-12 20:07 remote event files and `args.json` files remain under `data/remote` so the original figure snapshot is reproducible. The six complete remote experiment directory trees were later archived beside local runs under `onpolicy/scripts/results/mec/mappo/`; future executions of `plot.py` read those canonical local archives.

## Figure Audit (2026-08-12 20:08 +08:00)

- script_ran: yes
- png_exists: yes
- chart_type_matches_spec: pass
- data_columns_match_spec: pass
- axis_scales_match_spec: pass
- labels_units_match_spec: pass
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass
- forbidden_elements_absent: pass
- obvious_text_overlap_or_clipping: pass
- repairs_made: refreshed all six remote event/args snapshots, reran the plot, and visually inspected the 300-DPI PNG
- remaining_warnings: the supplied structural validator mis-parses nested Markdown axis fields, but direct spec/code/image inspection passes
- final_status: PASSED

## Refresh Audit (2026-08-13 00:57 +08:00)

- six_active_processes_verified: pass
- remote_snapshots_refreshed: pass; v_max=40 seeds 2/32/42 were pulled through 00:55 +08:00
- data_source_repair: `plot.py` now reads the refreshed `data/remote` snapshots instead of stale local remote archives
- protocol_assertion: pass; all 12 `args.json` files differ only in `experiment_name`, `seed`, and `v_max`
- aggregation_contract_unchanged: pass
- visual_inspection: pass; legend, uncertainty bands, axes, and footnote are readable and unclipped
- final_status: REPAIRED

## Refresh Audit (2026-08-12 22:38 +08:00)

- six_active_processes_verified: pass
- remote_snapshots_refreshed: pass; v_max=40 seeds 2/32/42 were pulled through 22:34 +08:00
- script_ran: yes
- png_exists: yes
- protocol_assertion: pass for all 12 runs
- aggregation_contract_unchanged: pass
- visual_inspection: pass; legend, uncertainty bands, axes, and footnote are readable and unclipped
- final_status: PASSED
