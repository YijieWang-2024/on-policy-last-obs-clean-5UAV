param(
    [int]$Seed = 2,
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [string]$UserName = $env:USERNAME,
    [string]$ExperimentName = ''
)

$ErrorActionPreference = 'Stop'
$common = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
if (-not (Test-Path -LiteralPath $common)) {
    throw "Common launcher not found: $common"
}

if (-not $ExperimentName) {
    $ExperimentName = (
        'dcppoR260_fixed600_200_layoutctx_inputv2_' +
        'peragentnoise_s3p0_seed' + $Seed + '_60m_20260809_retry1'
    )
}

& $common `
    -NoiseScale 3.0 `
    -Seed $Seed `
    -NumEnvSteps $NumEnvSteps `
    -RolloutThreads $RolloutThreads `
    -NeighborDistance 260 `
    -NeighborR 260 `
    -HotspotLayoutMode 'episode_template4_600_200' `
    -EpisodeLayoutContext `
    -EpisodeLayoutContextUnits 'meters_v2' `
    -ActorMessageContract 'absolute_raw_v2' `
    -UserName $UserName `
    -ExperimentName $ExperimentName
exit $LASTEXITCODE
