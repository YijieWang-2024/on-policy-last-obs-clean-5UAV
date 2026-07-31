# Figure Audit

- script_ran: yes
- png_exists: yes
- chart_type_matches_spec: pass
- data_columns_match_spec: pass; direct `agent0/system_performance_true_all_GUs` values are exported with raw and smoothed columns
- axis_scales_match_spec: pass; both axes are linear and x spans 0-100M
- labels_units_match_spec: pass
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass; live runs are identified in the legend
- forbidden_elements_absent: pass
- obvious_text_overlap_or_clipping: pass by visual inspection
- repairs_made: moved and shortened the legend after the first render obscured the early Q2 curve
- remaining_warnings: the generic validator does not parse the nested axis fields in the prescribed figure-spec format and warns about the intentionally omitted title; structural and visual checks pass
- final_status: REPAIRED
