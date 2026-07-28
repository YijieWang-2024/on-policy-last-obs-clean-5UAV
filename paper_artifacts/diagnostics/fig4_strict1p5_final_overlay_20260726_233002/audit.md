# Figure Audit

- script_ran: yes
- png_exists: yes; 2964 by 1945 pixels at 300 dpi
- chart_type_matches_spec: pass
- data_columns_match_spec: pass; 977 points per completed run, 1954 exported rows
- axis_scales_match_spec: pass; linear calibration matches canonical x=0/3500 and y=400k/500k grid ticks
- labels_units_match_spec: pass; canonical labels remain visible and the added legend names the equivalent-60-MD metric
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass
- forbidden_elements_absent: pass; no extrapolation, deterministic-evaluation claim, or fabricated band
- obvious_text_overlap_or_clipping: pass
- repairs_made: none after final render
- remaining_warnings: the generic validator cannot detect axes embedded in the canonical raster and requests `bbox_inches="tight"`; preserving exact raster geometry requires `bbox_inches=None`. This remains a diagnostic overlay rather than a replacement vector manuscript figure. The current curves are one training seed and the environment differs from historical fixed-60-MD Fig. 4.
- final_status: PASSED_WITH_WARNINGS
