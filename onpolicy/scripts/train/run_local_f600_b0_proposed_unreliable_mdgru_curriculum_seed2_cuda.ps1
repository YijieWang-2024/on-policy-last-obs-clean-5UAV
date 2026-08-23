$ErrorActionPreference = 'Stop'

# F600-B0 proposed method with training-only UAV reset curriculum.
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
    -Seed 2 `
    -NumEnvSteps 60000000 `
    -RolloutThreads 64 `
    -StateReconstruction 'md_gru' `
    -AdvantageMode 'per_agent_noise' `
    -NoiseScale 3.0 `
    -CommunicationDistance 520 `
    -ActorNeighborDistance 520 `
    -A2ATransmitPowerW 1.1809658836179866 `
    -A2ARicianKDb 10.0 `
    -A2ADistanceToleranceM 5.0 `
    -RunningSumRounds 50 `
    -MDGRUTargetBatchSize 2048 `
    -MDGRUBatchesPerRollout 10 `
    -MDGRUMinReadySamples 2048 `
    -AssociationThreshold 0.5 `
    -DisableOffloadDeadlineFilter `
    -ClipParam 0.15 `
    -Gamma 0.99 `
    -PpoEpoch 4 `
    -UAVResetCurriculum `
    -UAVResetCurriculumSchedule 'p0p7_10m_25m' `
    -ActorMessageMode 'disabled' `
    -Python $python `
    -ExperimentName 'F600_B0_proposed_unreliable_mdgru_curriculum_p0p7_10m_25m_tb2048x10_seed2_60m_cuda'
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "Unreliable MD-GRU curriculum training exited with code $exitCode."
}
