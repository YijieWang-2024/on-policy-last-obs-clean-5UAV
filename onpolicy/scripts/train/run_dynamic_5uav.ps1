param(
    [ValidateSet('mappo', 'dcppo', 'ippo')]
    [string]$Method = 'dcppo',
    [int]$Seed = 2,
    [long]$NumEnvSteps = 100000000,
    [int]$RolloutThreads = 64,
    [int]$CommunicationDistance = 260,
    [int]$ActorNeighborDistance = 260,
    [int]$ConsensusRounds = 50,
    [ValidateSet('local', 'mixed_consensus', 'pure_consensus', 'externality_consensus', 'legacy_noise')]
    [string]$AdvantageMode = 'mixed_consensus',
    [double]$ConsensusAlpha = 0.5,
    [double]$ExternalityBeta = 0.1,
    [string]$Python = 'python',
    [string]$ExperimentName = '',
    [switch]$CartesianFlight,
    [switch]$ActorNeighborObs,
    [switch]$SpatialFlightActor,
    [switch]$DistanceOnlyUserSort,
    [switch]$CompletionPriorityUserSort,
    [switch]$UAVResetCurriculum,
    [switch]$LegacyMeanPoolCritic,
    [switch]$IndependentReturnNorm
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$trainScript = Join-Path $repoRoot 'onpolicy\scripts\train\train_mec.py'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
if (-not $ExperimentName) {
    $ExperimentName = "regional_dynamic_$Method"
}

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
    '--md_arrivals_min', '6',
    '--md_arrivals_max', '6',
    '--md_arrivals_per_region', '1', '5',
    '--md_lifetime_min', '10',
    '--md_lifetime_max', '10',
    '--x_min_uav', '0', '--x_max_uav', '600',
    '--y_min_uav', '0', '--y_max_uav', '600',
    '--x_min_gu', '0', '--x_max_gu', '600',
    '--y_min_gu', '0', '--y_max_gu', '600',
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
    '--v_max', '30',
    '--mean_velocity', '0.5',
    '--alpha_r', '32',
    '--gamma_r', '26',
    '--delta_r', '32',
    '--lambda_r', '0.000001',
    '--mu_r', '64',
    '--use_valuenorm'
)

if ($Method -ne 'ippo') {
    $arguments += '--use_atten_critic'
}

if ($CartesianFlight) {
    $arguments += '--cartesian_flight'
}
if ($ActorNeighborObs) {
    $arguments += '--actor_neighbor_obs'
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
}

if ($Method -eq 'dcppo') {
    $arguments += @(
        '--neighbor_distance', $CommunicationDistance,
        '--advantage_mode', $AdvantageMode,
        '--consensus_alpha', $ConsensusAlpha,
        '--externality_beta', $ExternalityBeta,
        '--n_iterations', $ConsensusRounds
    )
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
