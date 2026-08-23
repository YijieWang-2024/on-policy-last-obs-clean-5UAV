$ErrorActionPreference = 'Stop'

# F600-B0 control: the same unreliable data plane as Proposed, but missing MD
# critic tokens use the causal last observation instead of the GRU predictor.
$python = if ($env:MARL_PYTHON) {
    $env:MARL_PYTHON
} else {
    Join-Path $env:USERPROFILE '.conda\envs\marl\python.exe'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "CUDA Python not found: $python"
}

& $python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"CUDA: {torch.cuda.get_device_name(0)}")'
if ($LASTEXITCODE -ne 0) {
    throw 'CUDA preflight failed; this script intentionally has no CPU fallback.'
}

$launcher = Join-Path $PSScriptRoot 'run_fixed600_200_unreliable_dataplane.ps1'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Unreliable launcher not found: $launcher"
}

$exitCode = 1
& $launcher `
    -Seed 32 `
    -NumEnvSteps 60000000 `
    -RolloutThreads 64 `
    -StateReconstruction 'last_obs' `
    -AdvantageMode 'per_agent_noise' `
    -NoiseScale 3.0 `
    -CommunicationDistance 520 `
    -ActorNeighborDistance 520 `
    -A2ATransmitPowerW 1.1809658836179866 `
    -A2ARicianKDb 10.0 `
    -A2ADistanceToleranceM 5.0 `
    -RunningSumRounds 50 `
    -AssociationThreshold 0.5 `
    -DisableOffloadDeadlineFilter `
    -ClipParam 0.15 `
    -Gamma 0.99 `
    -PpoEpoch 4 `
    -ActorMessageMode 'disabled' `
    -Python $python `
    -ExperimentName 'F600_B0_control_unreliable_last_obs_seed32_60m_cuda'
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "Unreliable last-observation training exited with code $exitCode."
}
