param(
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [string]$UserName = $env:USERNAME,
    [string]$ExperimentName = ''
)

$ErrorActionPreference = 'Stop'
$common = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
if (-not $ExperimentName) {
    $ExperimentName = 'dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_ppoepoch10_seed2_60m_20260813'
}

& $common -NoiseScale 3.0 -Seed 2 -NumEnvSteps $NumEnvSteps `
    -RolloutThreads $RolloutThreads -MdLifetime 12 -UAVMaxSpeed 30 `
    -NeighborDistance 520 -NeighborR 520 `
    -HotspotLayoutMode 'episode_template4_600_200' -EpisodeLayoutContext `
    -EpisodeLayoutContextUnits 'meters_v2' `
    -ActorMessageMode 'disabled' -ActorMessageContract 'absolute_raw_v2' `
    -ClipParam 0.15 -Gamma 0.99 -PpoEpoch 10 `
    -UserName $UserName -ExperimentName $ExperimentName
exit $LASTEXITCODE
