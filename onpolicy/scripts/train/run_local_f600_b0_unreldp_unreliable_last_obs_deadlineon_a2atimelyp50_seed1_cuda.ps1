$ErrorActionPreference = 'Stop'

$python = if ($env:MARL_PYTHON) { $env:MARL_PYTHON } else { Join-Path $env:USERPROFILE '.conda\envs\marl\python.exe' }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "CUDA Python not found: $python" }
& $python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"CUDA: {torch.cuda.get_device_name(0)}")'
if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed; this script intentionally has no CPU fallback.' }

$launcher = Join-Path $PSScriptRoot 'run_fixed600_200_unreliable_dataplane.ps1'
& $launcher `
    -Seed 1 -NumEnvSteps 60000000 -RolloutThreads 64 `
    -StateReconstruction 'last_obs' `
    -AdvantageMode 'per_agent_noise' -NoiseScale 3.0 `
    -CommunicationDistance 520 -ActorNeighborDistance 520 `
    -A2ATransmitPowerW 1.1809658836179866 -A2ARicianKDb 10.0 -A2ADistanceToleranceM 5.0 `
    -StateDeadlineMs 10.175730095548 -AdvantageDeadlineMs 18.175730095548 `
    -RunningSumRounds 50 -AssociationThreshold 0.5 `
    -UAVResourceMode 'homogeneous' `
    -ClipParam 0.15 -Gamma 0.99 -PpoEpoch 4 `
    -ActorMessageMode 'disabled' -Python $python `
    -ExperimentName 'F600_B0_unreldp_unreliable_last_obs_nocurr_deadlineon_a2atimelyp50_pa_s3_seed1_60m_cuda'
if ($LASTEXITCODE -ne 0) { throw "Unreliable last-observation A2A-timely-p50 seed-1 training exited with code $LASTEXITCODE." }
