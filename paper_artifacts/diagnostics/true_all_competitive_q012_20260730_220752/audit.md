# Figure Audit

- script_ran: yes
- png_exists: yes
- chart_type_matches_spec: pass
- data_columns_match_spec: pass; all five curves use `agent0/system_performance_true_all_GUs`
- axis_scales_match_spec: pass; linear 0-100M environment-step axis
- labels_units_match_spec: pass
- series_category_order_match_spec: pass; Q2, Q0, Q1, MAPPO reference, DC-PPO reference
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass; none required
- forbidden_elements_absent: pass; no equivalent-60 imputation, reward mixing, smoke tests, or invented bands
- obvious_text_overlap_or_clipping: pass by visual inspection
- repairs_made: none required
- remaining_warnings: the generic validator cannot parse the nested Markdown axis fields and reports false axis-label warnings; labels and scales are present in both script and PNG
- final_status: PASSED_WITH_WARNINGS
