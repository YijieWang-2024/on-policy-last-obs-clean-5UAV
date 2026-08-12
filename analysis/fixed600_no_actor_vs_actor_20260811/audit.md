# Figure Audit

- script_ran: yes
- png_exists: yes (`remote_no_actor_radii.png`, `no_actor_vs_actor_r520.png`, `r520_no_actor_vs_actor.png`, and canonical `plot.png`)
- chart_type_matches_spec: pass
- data_columns_match_spec: pass
- axis_scales_match_spec: pass (both axes linear)
- labels_units_match_spec: pass
- series_category_order_match_spec: pass (R0, R260, R520, then local R520 actor_message)
- legend_complete_and_uncropped: pass; final value and final step are in every legend entry
- annotations_match_spec: pass; no smoothing or extra annotations
- forbidden_elements_absent: pass
- obvious_text_overlap_or_clipping: pass on rendered PNG inspection
- repairs_made: added the requested R520-only output, used purple plus a dashed line for the local actor-message curve in that output, and retained the canonical `plot.png` alias; reran the script using the existing refreshed snapshots
- remaining_warnings: the bundled validator reports false missing axis fields because its regex stops at nested Markdown bullets; the delivered figure-spec uses the required structured axis fields and the rendered axes were checked directly
- final_status: PASSED_WITH_WARNINGS
