param(
    [int]$Seed = 2,
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [int]$EpisodeLength = 400,
    [ValidateSet('zero', 'last_obs', 'md_gru')]
    [string]$StateReconstruction = 'md_gru',
    [ValidateSet('local', 'mixed_consensus', 'pure_consensus', 'legacy_noise', 'per_agent_noise')]
    [string]$AdvantageMode = 'per_agent_noise',
    [double]$NoiseScale = 3.0,
    [Nullable[double]]$CommunicationDistance = $null,
    [Nullable[double]]$CriticNeighborDistance = $null,
    [double]$ActorNeighborDistance = 520,
    [Nullable[double]]$A2ATransmitPowerW = $null,
    [double]$A2ARicianKDb = 10.0,
    [double]$A2ADistanceToleranceM = 5.0,
    [double]$StateDeadlineMs = 13.54,
    [double]$AdvantageDeadlineMs = 21.54,
    [int]$RunningSumRounds = 50,
    [int]$MDGRUTargetBatchSize = 2048,
    [int]$MDGRUBatchesPerRollout = 10,
    [int]$MDGRUMinReadySamples = 2048,
    [ValidateRange(0.0, 1.0)]
    [double]$AssociationThreshold = 0.5,
    [switch]$DisableOffloadDeadlineFilter,
    [ValidateSet('homogeneous', 'heterogeneous')]
    [string]$UAVResourceMode = 'homogeneous',
    [double[]]$UAVResourceScaleFactors = @(),
    [ValidateRange(0.0, 1.0)]
    [double]$ClipParam = 0.15,
    [ValidateRange(0.0, 1.0)]
    [double]$Gamma = 0.99,
    [ValidateRange(1, 100)]
    [int]$PpoEpoch = 4,
    [switch]$UAVResetCurriculum,
    [ValidateSet('legacy', 'p0p7_10m_25m')]
    [string]$UAVResetCurriculumSchedule = 'legacy',
    [string]$ModelDir = '',
    [ValidateSet('disabled', 'task_summary')]
    [string]$ActorMessageMode = 'disabled',
    [switch]$CPUOnly,
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
    $communicationTag = if ($null -ne $CommunicationDistance) {
        "R${CommunicationDistance}"
    } elseif ($null -ne $A2ATransmitPowerW) {
        "Pc${A2ATransmitPowerW}W"
    } else {
        'R520'
    }
    $ExperimentName = (
        "dcppo${communicationTag}_fixed600_200_vmax30_md12_unreliable_${AdvantageMode}_${StateReconstruction}_${messageTag}_seed${Seed}_60m"
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
    -CommunicationDistance $CommunicationDistance `
    -CriticNeighborDistance $CriticNeighborDistance `
    -ActorNeighborDistance $ActorNeighborDistance `
    -RunningSumRounds $RunningSumRounds `
    -A2ATransmitPowerW $A2ATransmitPowerW `
    -A2ARicianKDb $A2ARicianKDb `
    -A2ADistanceToleranceM $A2ADistanceToleranceM `
    -StateDeadlineMs $StateDeadlineMs `
    -AdvantageDeadlineMs $AdvantageDeadlineMs `
    -AssociationThreshold $AssociationThreshold `
    -DisableOffloadDeadlineFilter:$DisableOffloadDeadlineFilter `
    -UAVResourceMode $UAVResourceMode `
    -UAVResourceScaleFactors $UAVResourceScaleFactors `
    -ClipParam $ClipParam -Gamma $Gamma -PpoEpoch $PpoEpoch `
    -ModelDir $ModelDir `
    -StateReconstruction $StateReconstruction `
    -MDGRUTargetBatchSize $MDGRUTargetBatchSize `
    -MDGRUBatchesPerRollout $MDGRUBatchesPerRollout `
    -MDGRUMinReadySamples $MDGRUMinReadySamples `
    -CriticMDMetadata `
    -AdvantageMode $AdvantageMode `
    -NoiseScale $NoiseScale `
    -ActorMessageMode $ActorMessageMode `
    -ActorMessagePool receiver_gated_sum `
    -ActorMessageContract absolute_raw_v2 `
    -HotspotLayoutMode episode_template4_600_200 `
    -UAVStartPositions @(110, 180, 220, 180, 330, 180, 440, 180, 400, 400) `
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
    -UAVResetCurriculum:$UAVResetCurriculum `
    -UAVResetCurriculumSchedule $UAVResetCurriculumSchedule `
    -EpisodeLayoutContext `
    -EpisodeLayoutContextUnits meters_v2 `
    -CPUOnly:$CPUOnly `
    -Python $Python `
    -ExperimentName $ExperimentName
exit $LASTEXITCODE
