param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(0.0, 100.0)]
    [double]$NoiseScale,
    [int]$Seed = 2,
    [long]$NumEnvSteps = 60000000,
    [int]$RolloutThreads = 64,
    [double]$NeighborDistance = 0,
    [int]$NeighborR = 0,
    [string]$UserName = $env:USERNAME,
    [string]$ExperimentName = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$trainScript = Join-Path $repoRoot 'onpolicy\scripts\train\train_mec.py'

if (-not (Test-Path -LiteralPath $trainScript)) {
    throw "Training entry point not found: $trainScript"
}

$python = $env:MARL_PYTHON
if (-not $python) {
    $python = Join-Path $env:USERPROFILE '.conda\envs\marl\python.exe'
}
if (-not (Test-Path -LiteralPath $python)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $python = $pythonCommand.Source
    } else {
        throw "Python not found. Set MARL_PYTHON to the marl environment's python.exe."
    }
}

$noiseTag = ('{0:00}' -f [int][math]::Round($NoiseScale * 100))
if (-not $ExperimentName) {
    $ExperimentName = 'fixed2hotspot_nocurr_A_peragentnoise_s' + $noiseTag + '_seed' + $Seed + '_60m_20260807'
}

$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($ExperimentName) })
if ($running.Count -gt 0) {
    throw "An experiment with this name is already running: $ExperimentName"
}

# Keep Python's output UTF-8 so the existing Chinese startup message cannot
# terminate the process under Windows PowerShell's legacy code page.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

$logDir = Join-Path $repoRoot 'training_logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logDir ($ExperimentName + '_' + $stamp + '.log')

$trainArgs = @(
    $trainScript,
    '--env_name', 'mec',
    '--algorithm_name', 'mappo',
    '--user_name', $UserName,
    '--seed', $Seed,
    '--share_policy',
    '--n_training_threads', '1',
    '--n_rollout_threads', $RolloutThreads,
    '--n_UAVs', '5',
    '--max_UAVs_in_neighbor', '5',
    '--max_UAVs_obs_concat', '5',
    '--neighbor_distance', $NeighborDistance,
    '--neighbor_R', $NeighborR,
    '--d_optimal', '210',
    '--n_GUs', '60',
    '--max_GUs_in_range', '20',
    '--dynamic_md',
    '--md_arrivals_min', '5',
    '--md_arrivals_max', '5',
    '--md_arrivals_per_region', '1', '4',
    '--hotspot_layout_mode', 'fixed_legacy',
    '--five_uav_start_layout', 'line',
    '--uav_start_positions', '110', '180', '220', '180', '330', '180', '440', '180', '400', '400',
    '--md_lifetime_min', '10',
    '--md_lifetime_max', '10',
    '--x_max', '600',
    '--x_min_uav', '0',
    '--x_max_uav', '600',
    '--y_min_uav', '0',
    '--y_max_uav', '600',
    '--x_min_gu', '0',
    '--x_max_gu', '600',
    '--y_min_gu', '0',
    '--y_max_gu', '600',
    '--fix_hotspot',
    '--B', '30000000',
    '--H_UAV', '120',
    '--H_GU', '1',
    '--alpha_r', '32',
    '--beta_r', '0.5',
    '--q1', '1',
    '--gamma_r', '26',
    '--delta_r', '32',
    '--q2', '1',
    '--epsilon_r', '0',
    '--q3', '1',
    '--lambda_r', '0.000001',
    '--q4', '1',
    '--mu_r', '64',
    '--q5', '1',
    '--Delta_t', '0.5',
    '--F_m', '20000000000',
    '--F_n', '1500000000',
    '--D_min', '200000',
    '--D_max', '400000',
    '--C_min', '675000000',
    '--C_max', '800000000',
    '--delay_min', '0.499',
    '--delay_max', '0.5',
    '--Dis_min', '3',
    '--Cover_R', '120',
    '--w1', '20',
    '--w2', '1',
    '--p3', '500',
    '--v_max', '30',
    '--mean_velocity', '3',
    '--md_velocity_init_std', '0.6',
    '--md_velocity_init_min_factor', '0.4',
    '--md_velocity_init_max_factor', '1.6',
    '--md_velocity_update_clip_min', '0',
    '--md_velocity_update_clip_max', '5',
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
    '--use_valuenorm',
    '--shared_ret_norm',
    '--state_is_k_hops',
    '--all_uav_k_hops',
    '--use_atten_critic',
    '--ego_query_critic',
    '--local_reward',
    '--continuous_associate',
    '--not_served_rew_to_nearest',
    '--cartesian_flight',
    '--actor_message_mode', 'task_summary',
    '--actor_message_pool', 'receiver_gated_sum',
    '--spatial_flight_actor',
    '--completion_priority_user_sort',
    '--n_iterations', '50',
    '--advantage_mode', 'per_agent_noise',
    '--experiment_name', $ExperimentName,
    '--noise_scale', $NoiseScale
)

Write-Host "Experiment : $ExperimentName"
Write-Host "Noise scale: $NoiseScale"
Write-Host "Seed       : $Seed"
Write-Host "Rollouts   : $RolloutThreads"
Write-Host "Log        : $logPath"
Write-Host "Python     : $python"
Write-Host "Press Ctrl+C to stop this run."

Push-Location $repoRoot
try {
    & $python @trainArgs 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Training exited with code $exitCode. See $logPath"
    }
} finally {
    Pop-Location
}
