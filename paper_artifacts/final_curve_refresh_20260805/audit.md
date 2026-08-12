# Figure audit

- script_ran: yes
- png_exists: yes (six final curve PNGs)
- chart_type_matches_spec: pass
- data_columns_match_spec: pass
- axis_scales_match_spec: pass
- labels_units_match_spec: pass
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass (last event step labels and final tail100 values)
- forbidden_elements_absent: pass (no extrapolation or common-step truncation)
- obvious_text_overlap_or_clipping: pass after visual inspection of representative RandomLayout700, MovingHotspot700, Diag4, curriculum, Subset29 and FixedDiag figures
- repairs_made: corrected MovingHotspot filtering to exclude nested RandomLayout runs; corrected Diag4 source from the early live snapshot to the final 46.464M scalar export
- remaining_warnings: MovingHotspot seed=2 is an interrupted run, so its line ends earlier; this is shown explicitly rather than filled or extrapolated
- final_status: REPAIRED
