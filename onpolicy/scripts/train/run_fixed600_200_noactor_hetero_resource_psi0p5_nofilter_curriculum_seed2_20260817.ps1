param([string]$UserName = $env:USERNAME)

$common = Join-Path $PSScriptRoot 'run_fixed2hotspot_per_agent_noise.ps1'
if (-not (Test-Path -LiteralPath $common -PathType Leaf)) {
    throw "Common launcher not found: $common"
}

& $common `
    -NoiseScale 3.0 -Seed 2 -NumEnvSteps 60000000 -RolloutThreads 64 `
    -MdLifetime 12 -UAVMaxSpeed 30 -NeighborDistance 520 -NeighborR 520 `
    -HotspotLayoutMode episode_template4_600_200 -EpisodeLayoutContext `
    -EpisodeLayoutContextUnits meters_v2 -ActorMessageMode disabled `
    -ActorMessageContract absolute_raw_v2 -AssociationThreshold 0.5 `
    -DisableOffloadDeadlineFilter `
    -UAVResourceMode heterogeneous `
    -UAVResourceScaleFactors 1.0,1.3,0.7,1.3,0.7 `
    -UAVResetCurriculum -UAVResetCurriculumSchedule p0p7_10m_25m `
    -UserName $UserName `
    -ExperimentName 'dcppoR520_fixed600_200_het1_13_07_13_07_psi05_nf_cur_s2_60m_20260817'
exit $LASTEXITCODE
