# Final Status

- Status: PASSED
- Output: `plot.png`
- Reproducible script: `plot.py`
- Machine-readable run and aggregation summary: `progress_summary.json`
- Selection and parameter audit: `audit.md`
- Figure contract: `figure-spec.md`
- Resolution: 300 DPI
- Palette: Okabe-Ito colorblind-safe colors
- Refreshed: 2026-08-13 15:24 +08:00
- Visual inspection: passed; axes, title, legend, uncertainty bands, and footnote are readable and unclipped
- Protocol assertion: passed for all 12 runs; legend explicitly reports available seeds as `x/3`
- Remote archive: six remote run trees synchronized to `onpolicy/scripts/results/mec/mappo/` and verified by `remote_speed_sync_manifest.json`
- Completion boundary: remote v40 seeds 2/32/42 are complete; local v10/v20/v30 seed2 were stopped before 60M, so their terminal aggregate uses two complete seeds
