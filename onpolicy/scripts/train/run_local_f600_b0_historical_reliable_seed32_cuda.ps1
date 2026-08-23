$ErrorActionPreference = 'Stop'

# Exact seed-32 continuation of the 2026-08-14 Fixed600-200 historical
# reliable baseline: no curriculum and no offload-deadline filter.
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

$launcher = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Reliable launcher not found: $launcher"
}

$previousMarlPython = $env:MARL_PYTHON
$env:MARL_PYTHON = $python
$exitCode = 1
try {
    & $launcher `
        -NoiseScale 3.0 `
        -Seed 32 `
        -NumEnvSteps 60000000 `
        -RolloutThreads 64 `
        -MdLifetime 12 `
        -UAVMaxSpeed 30 `
        -NeighborDistance 520 `
        -NeighborR 520 `
        -HotspotLayoutMode 'episode_template4_600_200' `
        -EpisodeLayoutContext `
        -EpisodeLayoutContextUnits 'meters_v2' `
        -ActorMessageMode 'disabled' `
        -ActorMessageContract 'absolute_raw_v2' `
        -AssociationThreshold 0.5 `
        -DisableOffloadDeadlineFilter `
        -ClipParam 0.15 `
        -Gamma 0.99 `
        -PpoEpoch 4 `
        -ExperimentName 'dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed32_60m_20260823'
    $exitCode = $LASTEXITCODE
}
finally {
    if ($null -eq $previousMarlPython) {
        Remove-Item Env:MARL_PYTHON -ErrorAction SilentlyContinue
    } else {
        $env:MARL_PYTHON = $previousMarlPython
    }
}
if ($exitCode -ne 0) {
    throw "Historical reliable training exited with code $exitCode."
}
