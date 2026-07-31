# Figure Audit

- script_ran: yes
- png_exists: yes; 3307 by 1945 pixels at 300 dpi
- chart_type_matches_spec: pass; one Q2 line is overlaid on the canonical five-curve Figure 4 raster
- data_columns_match_spec: pass; all 1953 Q2 TensorBoard points are exported
- axis_scales_match_spec: pass; linear pixel calibration preserves the canonical axes and extends x from 3500 to 4000
- labels_units_match_spec: pass; the canonical `Learning episodes` and `System gain` labels remain visible
- series_category_order_match_spec: pass; historical raster is unchanged and Q2 is drawn last
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass; the historical horizon is marked at episode 3500
- forbidden_elements_absent: pass; no Q2 uncertainty band or cross-environment superiority annotation was added
- obvious_text_overlap_or_clipping: pass
- repairs_made: none after the first render
- remaining_warnings: the generic validator cannot parse labels embedded in the canonical raster and requests `bbox_inches="tight"`; exact pixel calibration requires preserving the raster with `bbox_inches=None`. This is a diagnostic numerical-scale comparison across different environments, not a manuscript-ready superiority result. The archived historical arrays contain three curves per algorithm although the old manuscript text states five seeds.
- final_status: PASSED_WITH_WARNINGS
