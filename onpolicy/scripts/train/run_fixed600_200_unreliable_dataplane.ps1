param(
    [int]$Seed = 2,
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [int]$EpisodeLength = 400,
    [ValidateSet('zero', 'last_obs', 'md_gru')]
    [string]$StateReconstruction = 'md_gru',
    [ValidateSet('disabled', 'task_summary')]
    [string]$ActorMessageMode = 'disabled',
    [string]$Python = 'C:\Users\wyj2\.conda\envs\marl\python.exe',
    [string]$ExperimentName = ''
)

$ErrorActionPreference = 'Stop'
$common = Join-Path $PSScriptRoot 'run_dynamic_5uav.ps1'
if (-not (Test-Path -LiteralPath $common)) {
    throw "Common launcher not found: $common"
}
if (-not $ExperimentName) {
    $messageTag = if ($ActorMessageMode -eq 'disabled') { 'noactor' } else { 'actor' }
    $ExperimentName = (
        "dcppoR520_fixed600_200_vmax30_md12_unreliable_${StateReconstruction}_${messageTag}_seed${Seed}_60m"
    )
}

& $common `
    -Method dcppo `
    -Seed $Seed `
    -NumEnvSteps $NumEnvSteps `
    -RolloutThreads $RolloutThreads `
    -EpisodeLength $EpisodeLength `
    -MDLifetime 12 `
    -CommunicationMode unreliable `
    -CommunicationDistance 520 `
    -ActorNeighborDistance 520 `
    -RunningSumRounds 50 `
    -StateReconstruction $StateReconstruction `
    -CriticMDMetadata `
    -AdvantageMode per_agent_noise `
    -NoiseScale 3.0 `
    -ActorMessageMode $ActorMessageMode `
    -ActorMessagePool receiver_gated_sum `
    -ActorMessageContract absolute_raw_v2 `
    -HotspotLayoutMode episode_template4_600_200 `
    -MapSize 600 `
    -UAVMaxSpeed 30 `
    -MeanVelocity 3.0 `
    -MDVelocityInitStd 0.6 `
    -MDVelocityInitMinFactor 0.4 `
    -MDVelocityInitMaxFactor 1.6 `
    -MDVelocityUpdateClipMin 0.0 `
    -MDVelocityUpdateClipMax 5.0 `
    -CartesianFlight `
    -SpatialFlightActor `
    -CompletionPriorityUserSort `
    -EpisodeLayoutContext `
    -EpisodeLayoutContextUnits meters_v2 `
    -Python $Python `
    -ExperimentName $ExperimentName
exit $LASTEXITCODE
