param(
    [ValidateSet('mappo', 'dcppo', 'ippo')]
    [string]$Method = 'dcppo',
    [int]$Seed = 2,
    [long]$NumEnvSteps = 100000000,
    [int]$RolloutThreads = 64,
    [int]$CommunicationDistance = 260,
    [int]$ActorNeighborDistance = 260,
    [int]$ConsensusRounds = 50,
    [ValidateSet('local', 'mixed_consensus', 'pure_consensus', 'externality_consensus', 'legacy_noise', 'per_agent_noise')]
    [string]$AdvantageMode = 'mixed_consensus',
    [double]$NoiseScale = 0.12,
    [double]$ConsensusAlpha = 0.5,
    [double]$ExternalityBeta = 0.1,
    [ValidateSet('disabled', 'zero', 'geometry', 'task_summary')]
    [string]$ActorMessageMode = 'disabled',
    [ValidateSet('mean', 'receiver_gated_sum')]
    [string]$ActorMessagePool = 'mean',
    [ValidateSet('fixed_legacy', 'episode_template4', 'episode_template4_600_200', 'episode_template12', 'episode_template12_600_200', 'episode_moving_template4')]
    [string]$HotspotLayoutMode = 'fixed_legacy',
    [int[]]$HotspotLayoutIndices = @(),
    [ValidateSet('line', 'staggered')]
    [string]$FiveUAVStartLayout = 'line',
    [ValidateSet(600, 700)]
    [int]$MapSize = 600,
    [int]$UAVMaxSpeed = 30,
    [double]$MeanVelocity = 0.5,
    [double]$MDVelocityInitStd = 0.3,
    [double]$MDVelocityInitMinFactor = 0.7,
    [double]$MDVelocityInitMaxFactor = 1.3,
    [Nullable[double]]$MDVelocityUpdateClipMin = $null,
    [Nullable[double]]$MDVelocityUpdateClipMax = $null,
    [string]$Python = 'python',
    [string]$ExperimentName = '',
    [switch]$CartesianFlight,
    [switch]$ActorNeighborObs,
    [switch]$SpatialFlightActor,
    [switch]$DistanceOnlyUserSort,
    [switch]$CompletionPriorityUserSort,
    [switch]$UAVResetCurriculum,
    [ValidateSet('legacy', 'p0p7_10m_25m')]
    [string]$UAVResetCurriculumSchedule = 'legacy',
    [switch]$EpisodeLayoutContext,
    [ValidateSet('normalized_v1', 'meters_v2')]
    [string]$EpisodeLayoutContextUnits = 'normalized_v1',
    [ValidateSet('relative_scaled_v1', 'absolute_raw_v2')]
    [string]$ActorMessageContract = 'relative_scaled_v1',
    [switch]$LegacyMeanPoolCritic,
    [switch]$IndependentReturnNorm
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$trainScript = Join-Path $repoRoot 'onpolicy\scripts\train\train_mec.py'
if ($MapSize -eq 700 -and $HotspotLayoutMode -notin @('episode_template4', 'episode_template12', 'episode_moving_template4')) {
    throw 'MapSize=700 requires episode_template4, episode_template12, or episode_moving_template4; fixed_legacy is the original 600m environment.'
}
if ($HotspotLayoutMode -eq 'episode_template12_600_200' -and $MapSize -ne 600) {
    throw 'episode_template12_600_200 requires MapSize=600.'
}
if ($MapSize -eq 600 -and $HotspotLayoutMode -in @('episode_template12', 'episode_moving_template4')) {
    throw "$HotspotLayoutMode requires MapSize=700. Use episode_template12_600_200 for the 600m Random12 protocol."
}
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
if (-not $ExperimentName) {
    $ExperimentName = "regional_dynamic_$Method"
}
$is600mOnePlusFour = $HotspotLayoutMode -in @(
    'episode_template4_600_200', 'episode_template12_600_200'
)
$regionArrivals = if ($is600mOnePlusFour) {
    @('1', '4')
} else {
    @('1', '5')
}
$regionalArrivalTotal = [int]$regionArrivals[0] + [int]$regionArrivals[1]

$arguments = @(
    $trainScript,
    '--env_name', 'mec',
    '--algorithm_name', $(if ($Method -eq 'ippo') { 'ippo' } else { 'mappo' }),
    '--experiment_name', $ExperimentName,
    '--user_name', $env:USERNAME,
    '--seed', $Seed,
    '--share_policy',
    '--n_UAVs', '5',
    '--n_GUs', '60',
    '--max_GUs_in_range', '20',
    '--dynamic_md',
    '--md_arrivals_min', $regionalArrivalTotal,
    '--md_arrivals_max', $regionalArrivalTotal,
    '--md_arrivals_per_region', $regionArrivals[0], $regionArrivals[1],
    '--hotspot_layout_mode', $HotspotLayoutMode,
    '--five_uav_start_layout', $FiveUAVStartLayout,
    '--md_lifetime_min', '10',
    '--md_lifetime_max', '10',
    '--x_max', $MapSize,
    '--x_min_uav', '0', '--x_max_uav', $MapSize,
    '--y_min_uav', '0', '--y_max_uav', $MapSize,
    '--x_min_gu', '0', '--x_max_gu', $MapSize,
    '--y_min_gu', '0', '--y_max_gu', $MapSize,
    '--fix_hotspot',
    '--max_UAVs_in_neighbor', '5',
    '--max_UAVs_obs_concat', '5',
    '--neighbor_R', $ActorNeighborDistance,
    '--state_is_k_hops',
    '--all_uav_k_hops',
    '--local_reward',
    '--continuous_associate',
    '--not_served_rew_to_nearest',
    '--n_rollout_threads', $RolloutThreads,
    '--episode_length', '400',
    '--num_env_steps', $NumEnvSteps,
    '--hidden_size', '256',
    '--layer_N', '2',
    '--lr', '0.0001',
    '--critic_lr', '0.0005',
    '--clip_param', '0.15',
    '--ppo_epoch', '4',
    '--num_mini_batch', '1',
    '--entropy_coef', '0',
    '--B', '30000000',
    '--F_m', '20000000000',
    '--v_max', $UAVMaxSpeed,
    '--mean_velocity', $MeanVelocity,
    '--md_velocity_init_std', $MDVelocityInitStd,
    '--md_velocity_init_min_factor', $MDVelocityInitMinFactor,
    '--md_velocity_init_max_factor', $MDVelocityInitMaxFactor,
    '--alpha_r', '32',
    '--gamma_r', '26',
    '--delta_r', '32',
    '--lambda_r', '0.000001',
    '--mu_r', '64',
    '--use_valuenorm'
)

if ($HotspotLayoutIndices.Count -gt 0) {
    $arguments += @('--hotspot_layout_indices') + $HotspotLayoutIndices
}
if ($EpisodeLayoutContext) {
    $arguments += '--episode_layout_context'
    $arguments += @('--episode_layout_context_units', $EpisodeLayoutContextUnits)
}

if ($Method -ne 'ippo') {
    $arguments += '--use_atten_critic'
}

if ($CartesianFlight) {
    $arguments += '--cartesian_flight'
}
if ($ActorNeighborObs) {
    $arguments += '--actor_neighbor_obs'
}
if ($ActorMessageMode -ne 'disabled') {
    $arguments += @('--actor_message_mode', $ActorMessageMode)
    $arguments += @('--actor_message_pool', $ActorMessagePool)
    $arguments += @('--actor_message_contract', $ActorMessageContract)
}
if ($SpatialFlightActor) {
    $arguments += '--spatial_flight_actor'
}
if ($DistanceOnlyUserSort) {
    $arguments += '--distance_only_user_sort'
}
if ($CompletionPriorityUserSort) {
    $arguments += '--completion_priority_user_sort'
}
if ($UAVResetCurriculum) {
    $arguments += '--uav_reset_curriculum'
    $arguments += @('--uav_reset_curriculum_schedule', $UAVResetCurriculumSchedule)
}
if ($null -ne $MDVelocityUpdateClipMin) {
    $arguments += @('--md_velocity_update_clip_min', $MDVelocityUpdateClipMin)
}
if ($null -ne $MDVelocityUpdateClipMax) {
    $arguments += @('--md_velocity_update_clip_max', $MDVelocityUpdateClipMax)
}

if ($Method -eq 'dcppo') {
    $arguments += @(
        '--neighbor_distance', $CommunicationDistance,
        '--advantage_mode', $AdvantageMode,
        '--consensus_alpha', $ConsensusAlpha,
        '--externality_beta', $ExternalityBeta,
        '--n_iterations', $ConsensusRounds
    )
    if ($AdvantageMode -eq 'per_agent_noise') {
        $arguments += @('--noise_scale', $NoiseScale)
    }
    if ($AdvantageMode -ne 'legacy_noise' -and -not $LegacyMeanPoolCritic) {
        $arguments += '--ego_query_critic'
    }
    if ($AdvantageMode -ne 'legacy_noise' -and -not $IndependentReturnNorm) {
        $arguments += '--shared_ret_norm'
    }
} elseif ($Method -eq 'mappo') {
    $arguments += @('--neighbor_distance', '1000')
} else {
    $arguments += @('--neighbor_distance', $CommunicationDistance)
}

Push-Location $repoRoot
try {
    & $Python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
