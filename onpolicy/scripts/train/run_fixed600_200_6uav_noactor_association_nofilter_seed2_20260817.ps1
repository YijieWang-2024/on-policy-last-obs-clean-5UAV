param(
    [int]$Seed = 2,
    [string]$UserName = $env:USERNAME,
    [string]$ExperimentName = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$trainScript = Join-Path $repoRoot 'onpolicy\scripts\train\train_mec.py'

if (-not (Test-Path -LiteralPath $trainScript -PathType Leaf)) {
    throw "Training entry point not found: $trainScript"
}

$python = $env:MARL_PYTHON
if (-not $python) {
    $python = Join-Path $env:USERPROFILE '.conda\envs\marl\python.exe'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $python = $pythonCommand.Source
    } else {
        throw "Python not found. Set MARL_PYTHON to the marl environment's python.exe."
    }
}

if (-not $ExperimentName) {
    $ExperimentName = 'dcppoR520_fixed600_200_6uav_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed' + $Seed + '_60m_20260817'
}

$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($ExperimentName) })
if ($running.Count -gt 0) {
    throw "An experiment with this name is already running: $ExperimentName"
}

# Keep Python output UTF-8.  The launcher also keeps stderr visible without
# letting an ordinary Python warning terminate a healthy training process.
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

# This is the fixed six-UAV protocol.  Keep all experiment-defining values in
# this one array so the saved args.json is an auditable record of the run.
$trainArgs = @(
    $trainScript,
    '--env_name', 'mec',
    '--algorithm_name', 'mappo',
    '--user_name', $UserName,
    '--seed', $Seed,
    '--share_policy',
    '--n_training_threads', '1',
    '--n_rollout_threads', '64',
    '--n_UAVs', '6',
    '--max_UAVs_in_neighbor', '6',
    '--max_UAVs_obs_concat', '6',
    '--neighbor_distance', '520',
    '--neighbor_R', '520',
    '--d_optimal', '210',
    '--n_GUs', '72',
    '--max_GUs_in_range', '20',
    '--dynamic_md',
    '--md_arrivals_min', '6',
    '--md_arrivals_max', '6',
    '--md_arrivals_per_region', '1', '5',
    '--hotspot_layout_mode', 'episode_template4_600_200',
    '--uav_start_positions',
        '110', '180',
        '220', '180',
        '330', '180',
        '440', '180',
        '550', '180',
        '400', '400',
    '--md_lifetime_min', '12',
    '--md_lifetime_max', '12',
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
    '--num_env_steps', '60000000',
    '--hidden_size', '256',
    '--layer_N', '2',
    '--lr', '0.0001',
    '--critic_lr', '0.0005',
    '--clip_param', '0.15',
    '--gamma', '0.99',
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
    '--association_threshold', '0.5',
    '--not_served_rew_to_nearest',
    '--cartesian_flight',
    '--actor_message_mode', 'disabled',
    '--actor_message_pool', 'receiver_gated_sum',
    '--actor_message_contract', 'absolute_raw_v2',
    '--spatial_flight_actor',
    '--completion_priority_user_sort',
    '--n_iterations', '50',
    '--advantage_mode', 'per_agent_noise',
    '--noise_scale', '3.0',
    '--episode_layout_context',
    '--episode_layout_context_units', 'meters_v2',
    '--disable_offload_deadline_filter',
    '--experiment_name', $ExperimentName
)

Write-Host "Experiment : $ExperimentName"
Write-Host "Environment: Fixed600-200 / 6 UAV / 1+5 MD / lifetime=12"
Write-Host "UAV starts : (110,180) (220,180) (330,180) (440,180) (550,180) (400,400)"
Write-Host "Algorithm  : separated MAPPO / no actor message / R520 critic"
Write-Host "Noise      : per_agent_noise, scale=3.0, K=50"
Write-Host "Association: psi=0.5 / deadline filter disabled"
Write-Host "Seed       : $Seed"
Write-Host "Rollouts   : 64"
Write-Host "Log        : $logPath"
Write-Host "Python     : $python"
Write-Host "Press Ctrl+C to stop this run."

Push-Location $repoRoot
try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python @trainArgs 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Training exited with code $exitCode. See $logPath"
    }
} finally {
    Pop-Location
}
