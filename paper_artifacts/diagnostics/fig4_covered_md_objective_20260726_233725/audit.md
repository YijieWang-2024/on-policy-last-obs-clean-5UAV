# Figure Audit

- script_ran: yes
- png_exists: yes; 2039 by 1287 pixels at 300 dpi
- chart_type_matches_spec: pass
- data_columns_match_spec: pass; 5454 exported rows across four series
- axis_scales_match_spec: pass; both axes are linear and x spans 0 to 3500
- labels_units_match_spec: pass; `Learning episodes` and `Covered-MD system performance` are present
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass; no annotation requested
- forbidden_elements_absent: pass; no equivalent-60 values, collision-augmented cumulative rewards, deterministic claims, or extrapolation
- obvious_text_overlap_or_clipping: pass
- repairs_made: none
- remaining_warnings: generic validator did not parse nested Markdown axis fields and warned about a missing title; structural inspection confirms both labels/scales, and no title was requested. Historical curves contain three seeds while current curves contain only seed2. Fixed-60 and strict-1+5 population processes remain different even though the covered-MD objective is aligned.
- final_status: PASSED
