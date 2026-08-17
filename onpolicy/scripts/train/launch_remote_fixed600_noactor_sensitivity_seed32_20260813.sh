#!/usr/bin/env bash
set -u -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runner="$repo_root/onpolicy/scripts/train/run_fixed600_200_noactor_sensitivity_seed32.sh"
log_dir="$repo_root/training_logs"
mkdir -p "$log_dir"

parameters=(clip gamma ppo_epoch clip gamma ppo_epoch)
values=(0.05 0.95 1 0.3 0.90 10)
tags=(clip0p05 gamma0p95 ppoepoch1 clip0p3 gamma0p90 ppoepoch10)
max_parallel=3

active_count() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -c '^hp_s32_' || true
}

launch_one() {
  local index="$1"
  local tag="${tags[$index]}"
  local session="hp_s32_${tag}_20260813"
  local experiment="dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_${tag}_seed32_60m_20260813"
  local log="$log_dir/${experiment}.log"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already active: $session"
    return
  fi
  tmux new-session -d -s "$session" \
    "cd '$repo_root' && exec bash '$runner' '${parameters[$index]}' '${values[$index]}' 32 '$experiment' > '$log' 2>&1"
  echo "started: $session ($experiment)"
}

# One representative value from each sensitivity family starts immediately.
for index in 0 1 2; do
  launch_one "$index"
done

# The second value from each family starts in order whenever a slot is free.
for index in 3 4 5; do
  while (( $(active_count) >= max_parallel )); do
    sleep 30
  done
  launch_one "$index"
done

echo "all queued experiments have been launched"
