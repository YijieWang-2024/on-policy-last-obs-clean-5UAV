# Figure specification

- chart_type: Two-panel time-series line chart with mean and one-standard-deviation bands.
- data_sources: Frozen deterministic evaluation rollouts for Q0, Q1, and Q2; ten complete 400-slot episodes per method.
- data_columns: method, slot, coverage_mean, coverage_std, cumulative_admitted_mean, cumulative_admitted_std.
- x_axis:
  - field : slot
  - label : Slot
  - scale : linear
- y_axis:
  - field : demand-weighted union coverage and cumulative admitted MD sessions
  - label : Demand-weighted union coverage (%) / Cumulative admitted MD sessions
  - scale : linear
- series_or_categories: Q0, Q1, and Q2; Q0 blue, Q1 orange, Q2 green; solid mean lines and translucent standard-deviation bands.
- legend: Shared legend below both panels; the admission panel also identifies the candidate upper bound.
- required_annotations: Horizontal 90% coverage reference and Q2 mean first slot sustaining at least 90% coverage for 20 consecutive slots.
- forbidden_elements: No 3D effects, gradients, decorative backgrounds, smoothed interpolation, or dual y-axis.
- assumptions: UAV coverage radius is 120 m; lower rectangle `[0,175] x [0,175]` has weight 1/6; upper rectangle `[200,600] x [200,600]` has weight 5/6; a session is admitted once at its first CSV appearance; `6 x slot` is the candidate-arrival upper bound.

## Interpretation boundary

The figure measures formation and persistence of geometric coverage and the resulting admission. It does not claim that post-arrival motion is energy-optimal, nor that the three checkpoints have exactly equal training budgets.
