param(
    [int]$Seed = 2,
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [string]$ExperimentName = ''
)

$common = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
if (-not $ExperimentName) {
    $ExperimentName = 'fixed2hotspot_nocurr_A_peragentnoise_s3p0_seed' + $Seed + '_60m_20260808'
}
& $common -NoiseScale 3.0 -Seed $Seed -NumEnvSteps $NumEnvSteps -RolloutThreads $RolloutThreads -ExperimentName $ExperimentName
