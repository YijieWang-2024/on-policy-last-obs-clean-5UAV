# Figure Audit

- script_ran: yes
- png_exists: yes
- chart_type_matches_spec: pass
- data_columns_match_spec: pass; stitched curves contain 1,954 points each and live E1/E2 contain all currently available records
- axis_scales_match_spec: pass; linear 0-4000 episode axis
- labels_units_match_spec: pass
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass; 50M boundary shown at episode 1953.125
- forbidden_elements_absent: pass
- obvious_text_overlap_or_clipping: pass by visual inspection
- repairs_made: canonical raster was extended on the right; live E1 and E2 were added at their native from-scratch steps
- remaining_warnings: the generic validator cannot parse labels embedded in the canonical Figure 4 raster; this is a diagnostic overlay rather than a clean vector regeneration of all five unavailable historical series
- final_status: PASSED_WITH_WARNINGS
