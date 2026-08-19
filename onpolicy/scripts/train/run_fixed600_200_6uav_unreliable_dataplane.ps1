param(
    [int]$Seed = 2,
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [ValidateSet('zero', 'last_obs', 'md_gru')]
    [string]$StateReconstruction = 'md_gru',
    [ValidateSet('local', 'mixed_consensus', 'pure_consensus', 'externality_consensus', 'legacy_noise', 'per_agent_noise')]
    [string]$AdvantageMode = 'per_agent_noise',
    [double]$NoiseScale = 3.0,
    [Nullable[double]]$CommunicationDistance = $null,
    [double]$ActorNeighborDistance = 520,
    [Nullable[double]]$A2ATransmitPowerW = $null,
    [double]$A2ARicianKDb = 10.0,
    [double]$A2ADistanceToleranceM = 5.0,
    [int]$RunningSumRounds = 50,
    [ValidateSet('disabled', 'task_summary')]
    [string]$ActorMessageMode = 'disabled',
    [ValidateSet('homogeneous', 'heterogeneous')]
    [string]$UAVResourceMode = 'homogeneous',
    [double[]]$UAVResourceScaleFactors = @(),
    [ValidateRange(0.0, 1.0)]
    [double]$AssociationThreshold = 0.5,
    [switch]$DisableOffloadDeadlineFilter,
    [switch]$UAVResetCurriculum,
    [switch]$CPUOnly,
    [string]$Python = 'C:\Users\wyj2\.conda\envs\marl\python.exe',
    [string]$ExperimentName = ''
)

$ErrorActionPreference = 'Stop'
$common = Join-Path $PSScriptRoot 'run_dynamic_5uav.ps1'
if (-not (Test-Path -LiteralPath $common -PathType Leaf)) {
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
    $ExperimentName = "dcppo${communicationTag}_fixed600_200_6uav_unreliable_${AdvantageMode}_${StateReconstruction}_${messageTag}_seed${Seed}_60m"
}

$runArgs = @{
    Method = 'dcppo'
    Seed = $Seed
    NumEnvSteps = $NumEnvSteps
    RolloutThreads = $RolloutThreads
    NumUAVs = 6
    NumGUs = 72
    MDArrivalsPerRegion = @(1, 5)
    EpisodeLength = 400
    MDLifetime = 12
    CommunicationMode = 'unreliable'
    CommunicationDistance = $CommunicationDistance
    ActorNeighborDistance = $ActorNeighborDistance
    A2ATransmitPowerW = $A2ATransmitPowerW
    A2ARicianKDb = $A2ARicianKDb
    A2ADistanceToleranceM = $A2ADistanceToleranceM
    RunningSumRounds = $RunningSumRounds
    StateReconstruction = $StateReconstruction
    CriticMDMetadata = $true
    AdvantageMode = $AdvantageMode
    NoiseScale = $NoiseScale
    ActorMessageMode = $ActorMessageMode
    ActorMessagePool = 'receiver_gated_sum'
    ActorMessageContract = 'absolute_raw_v2'
    HotspotLayoutMode = 'episode_template4_600_200'
    UAVStartPositions = @(110, 180, 220, 180, 330, 180, 440, 180, 550, 180, 400, 400)
    UAVResourceMode = $UAVResourceMode
    AssociationThreshold = $AssociationThreshold
    MapSize = 600
    UAVMaxSpeed = 30
    MeanVelocity = 3.0
    MDVelocityInitStd = 0.6
    MDVelocityInitMinFactor = 0.4
    MDVelocityInitMaxFactor = 1.6
    MDVelocityUpdateClipMin = 0.0
    MDVelocityUpdateClipMax = 5.0
    CartesianFlight = $true
    SpatialFlightActor = $true
    CompletionPriorityUserSort = $true
    EpisodeLayoutContext = $true
    EpisodeLayoutContextUnits = 'meters_v2'
    UAVResetCurriculum = [bool]$UAVResetCurriculum
    UAVResetCurriculumSchedule = 'p0p7_10m_25m'
    DisableOffloadDeadlineFilter = [bool]$DisableOffloadDeadlineFilter
    CPUOnly = [bool]$CPUOnly
    Python = $Python
    ExperimentName = $ExperimentName
}
if ($UAVResourceMode -eq 'heterogeneous') {
    $runArgs.UAVResourceScaleFactors = $UAVResourceScaleFactors
}

& $common @runArgs
exit $LASTEXITCODE
