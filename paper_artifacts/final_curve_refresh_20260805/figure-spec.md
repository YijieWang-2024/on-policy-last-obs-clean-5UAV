# Figure specification

- chart_type: two-panel line plus final-window bar comparison
- data_sources: final TensorBoard event files, exported `true60_metrics_raw.json`, and final scalar CSV exports
- metric: `agent0/system_performance_true_all_GUs` (True60)
- line_panel: every available run is plotted through its own last saved event; faint raw values and 21-event centered rolling means
- bar_panel: each run's own final `tail100` mean, with the exact last logged step retained in `final_curve_endpoints.csv`
- x_axis: environment steps in millions, linear
- y_axis: True60 in thousands, linear
- series: communication radius, with line style distinguishing seed when multiple seeds exist
- forbidden_elements: no extrapolation, no padding after the final event, no common-step truncation in the full-trace figures
- source_note: a separate common-step analysis remains in the original evidence files and is used for fair cross-run ranking; these figures answer the request to show all saved training data
