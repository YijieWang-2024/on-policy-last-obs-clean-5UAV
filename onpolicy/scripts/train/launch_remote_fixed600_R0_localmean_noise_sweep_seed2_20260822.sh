#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runner="$repo_root/onpolicy/scripts/train/run_fixed600_200_dcppo_R0_localmean_noise_seed2_20260822.sh"
log_dir="$repo_root/training_logs"
mkdir -p "$log_dir"

gpu_process_count() {
  local pids count=0
  if ! pids="$(nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null)"; then
    echo 999
    return
  fi
  while IFS= read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] && ((count += 1))
  done <<< "$pids"
  echo "$count"
}

while (( $(gpu_process_count) > 0 )); do
  sleep 60
done

for noise_scale in 1.0 2.0 4.0; do
  noise_tag="${noise_scale/./p}"
  session="lm_r0_s${noise_tag}_seed2_20260822"
  experiment="dcppoR0_fixed600_200_layoutctx_noactor_localmean_peragentnoise_s${noise_tag}_md12_vmax30_psi0p5_nofilter_seed2_60m_20260822"
  log="$log_dir/${experiment}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already active: $session"
    continue
  fi
  tmux new-session -d -s "$session" \
    "cd '$repo_root' && exec bash '$runner' '$noise_scale' > '$log' 2>&1"
  echo "started: $session ($experiment)"
  sleep 30
done

echo "all local-mean R0 noise experiments have been launched"
