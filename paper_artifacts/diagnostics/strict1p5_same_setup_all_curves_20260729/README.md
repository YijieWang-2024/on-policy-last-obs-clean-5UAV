# Strict 1+5 Same-Setup Training Comparison

Snapshot extracted on 2026-07-29. The comparison includes runs that use the same revised fixed UAV start and the strict regional MD process: one lower-left and five upper-right candidate arrivals per slot, lifetime 10.

## Included curves

1. DC-PPO 50M plus its warm-start continuation to 100M (stitched as one curve).
2. MAPPO 50M plus its warm-start continuation to 100M (stitched as one curve).
3. Cartesian MAPPO stopped diagnostic run.
4. E1 spatial-flight diagnostic run without reset curriculum.
5. E2 spatial-flight run with reset curriculum.
6. B0 standard actor with distance-only ordering and reset curriculum.
7. B1, which adds actor-only neighbor positions to B0.

Failed launches, r32/r64 resource trials superseded by formal runs, duplicate restart directories, older initial positions, and the earlier non-regional three-arrival environment are excluded.

## Files

- `training_curves.csv`: all extracted TensorBoard scalar records in long form.
- `tail100_metrics.csv`: arithmetic mean of the last 100 available records for every included run.
- `run_manifest.csv`: source run directories, status, family, and endpoint.
- `core_performance/plot.png`: six-panel smoothed training comparison.
- `tail100_summary/plot.png`: exact tail-100 values with within-column ranking.

The curve plot uses a trailing mean spanning approximately 2.5M environment steps. It does not show uncertainty: every experiment currently has one training seed.
