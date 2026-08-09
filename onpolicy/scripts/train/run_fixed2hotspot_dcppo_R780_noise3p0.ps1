param(
    [int]$Seed = 2,
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [string]$UserName = $env:USERNAME,
    [string]$ExperimentName = ''
)

$common = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
if (-not (Test-Path -LiteralPath $common)) {
    throw "Common launcher not found: $common"
}

if (-not $ExperimentName) {
    $ExperimentName = 'dcppoR780_fixed2hotspot_nocurr_peragentnoise_s3p0_seed' + $Seed + '_60m_20260808'
}

& $common `
    -NoiseScale 3.0 `
    -Seed $Seed `
    -NumEnvSteps $NumEnvSteps `
    -RolloutThreads $RolloutThreads `
    -UserName $UserName `
    -NeighborDistance 780 `
    -NeighborR 780 `
    -ExperimentName $ExperimentName
exit $LASTEXITCODE
