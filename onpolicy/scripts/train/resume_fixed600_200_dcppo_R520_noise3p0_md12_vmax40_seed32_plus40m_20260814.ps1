param(
    [string]$UserName = $env:USERNAME
)

$ErrorActionPreference = 'Stop'
$common = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
$modelDir = 'D:\wyj\Projects\on-policy-last-obs-clean-5UAV\onpolicy\scripts\results\mec\mappo\dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed32_60m_20260812\run1\models'

& $common `
    -NoiseScale 3.0 `
    -Seed 32 `
    -NumEnvSteps 40000000 `
    -RolloutThreads 64 `
    -MdLifetime 12 `
    -UAVMaxSpeed 40 `
    -NeighborDistance 520 `
    -NeighborR 520 `
    -HotspotLayoutMode 'episode_template4_600_200' `
    -EpisodeLayoutContext `
    -EpisodeLayoutContextUnits 'meters_v2' `
    -ActorMessageContract 'absolute_raw_v2' `
    -ActorMessageMode 'task_summary' `
    -ModelDir $modelDir `
    -UserName $UserName `
    -ExperimentName 'dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed32_resume40m_20260814'
exit $LASTEXITCODE
