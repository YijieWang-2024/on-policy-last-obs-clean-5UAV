# Figure Audit

- script_ran: yes
- png_exists: yes
- chart_type_matches_spec: pass
- data_columns_match_spec: pass; all six requested metrics and seven experiments are present
- axis_scales_match_spec: pass; all axes are linear and ratios are formatted as percentages
- labels_units_match_spec: pass
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass; smoothing, endpoint, status, and single-seed notes are visible
- forbidden_elements_absent: pass; raw noise and invented uncertainty are absent
- obvious_text_overlap_or_clipping: pass after original-resolution visual inspection
- repairs_made: replaced the initially redundant sum of five individual-performance tags with the distinct `agent0/system_performance` scalar
- remaining_warnings: the bundled validator mis-parses the nested axis blocks in its own documented spec format; structural and visual checks were completed manually
- final_status: REPAIRED
