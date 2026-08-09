param(
    [int]$Seed = 2,
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [string]$ExperimentName = ''
)

$common = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
& $common -NoiseScale 0.8 -Seed $Seed -NumEnvSteps $NumEnvSteps -RolloutThreads $RolloutThreads -ExperimentName $ExperimentName
