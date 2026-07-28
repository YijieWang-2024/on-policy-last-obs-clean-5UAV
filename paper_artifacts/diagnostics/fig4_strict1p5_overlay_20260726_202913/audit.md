# Figure Audit

- script_ran: yes
- png_exists: yes; 2964 by 1945 pixels at 300 dpi
- chart_type_matches_spec: pass
- data_columns_match_spec: pass; 1495 current TensorBoard points exported
- axis_scales_match_spec: pass; calibrated against canonical labelled ticks at x=0/3500 and y=400k/500k
- labels_units_match_spec: pass; canonical labels preserved and the added legend identifies the equivalent-60-MD metric
- series_category_order_match_spec: pass; canonical five-series background unchanged and two current series added
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass
- forbidden_elements_absent: pass; no current uncertainty band or extrapolation
- obvious_text_overlap_or_clipping: pass; current legend occupies the unused lower-right region
- repairs_made: replaced an initially reconstructed historical background after discovering archived F-PPO event logs did not reproduce the canonical manuscript curve; corrected overlay pixel calibration from preview dimensions to the canonical 2964 by 1945 raster; renamed the current lines as stochastic training rollouts so they cannot be confused with deterministic checkpoint evaluation
- remaining_warnings: generic validator flags missing programmatic axis-label calls and `bbox_inches="tight"`; these are intentionally inapplicable because the exact canonical raster already contains the axes and must retain its pixel geometry. This is a diagnostic overlay, not a replacement manuscript figure. Historical F-PPO/IPPO numerical values were not digitized.
- final_status: REPAIRED
