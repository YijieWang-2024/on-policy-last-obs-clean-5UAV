# Figure Audit

- script_ran: yes
- png_exists: yes
- chart_type_matches_spec: pass
- data_columns_match_spec: pass; exact tail-100 means are shown for seven experiments and eight metrics
- axis_scales_match_spec: pass; color is normalized independently within each metric column
- labels_units_match_spec: pass; performance uses thousands and ratios use percentages
- series_category_order_match_spec: pass; rows are ranked by tail-100 True60
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass; exact values and endpoint steps are present
- forbidden_elements_absent: pass; no experiment is omitted and no cross-metric color comparison is implied
- obvious_text_overlap_or_clipping: pass after original-resolution visual inspection
- repairs_made: missing action-projection logs in the two older stitched runs are shown as em dashes instead of zero; the repeated individual-performance sum was replaced with system performance
- remaining_warnings: endpoint budgets differ and are explicitly printed; B0/B1 are running snapshots; the bundled validator warns about heatmap tick labels because they intentionally replace conventional x/y labels
- final_status: REPAIRED
