param([string]$UserName = $env:USERNAME)

$common = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
& $common `
    -NoiseScale 3.0 -Seed 2 -NumEnvSteps 60000000 -RolloutThreads 64 `
    -MdLifetime 12 -UAVMaxSpeed 30 -NeighborDistance 520 -NeighborR 520 `
    -HotspotLayoutMode episode_template4_600_200 -EpisodeLayoutContext `
    -EpisodeLayoutContextUnits meters_v2 -ActorMessageMode disabled `
    -ActorMessageContract absolute_raw_v2 -AssociationThreshold 0.1 `
    -DisableOffloadDeadlineFilter -UserName $UserName `
    -ExperimentName 'dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p1_nofilter_seed2_60m_20260816'
exit $LASTEXITCODE
