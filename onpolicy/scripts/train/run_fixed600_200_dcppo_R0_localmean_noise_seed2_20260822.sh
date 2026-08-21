#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 1.0|2.0|4.0" >&2
  exit 2
fi

case "$1" in
  1|1.0) noise_scale="1.0"; noise_tag="1p0" ;;
  2|2.0) noise_scale="2.0"; noise_tag="2p0" ;;
  4|4.0) noise_scale="4.0"; noise_tag="4p0" ;;
  *) echo "noise scale must be 1.0, 2.0, or 4.0" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runner="$repo_root/onpolicy/scripts/train/run_fixed600_200_dcppo_R520_noise3p0_md12_no_actor_message.sh"
experiment="dcppoR0_fixed600_200_layoutctx_noactor_localmean_peragentnoise_s${noise_tag}_md12_vmax30_psi0p5_nofilter_seed2_60m_20260822"

exec bash "$runner" \
  0 "$experiment" local_mean_per_agent_noise "$noise_scale" disabled
