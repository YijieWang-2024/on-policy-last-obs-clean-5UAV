#!/usr/bin/env bash
set -u -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runner="$repo_root/onpolicy/scripts/train/run_fixed600_200_noactor_association_seed2.sh"
log_dir="$repo_root/training_logs"
queue_session="assoc_s2_psi_queue_20260814"

current_sessions=(
  hp_s32_clip0p3_20260813
  hp_s32_gamma0p90_20260813
  hp_s32_ppoepoch10_20260813
)

thresholds=(0.3 0.7 0.9)
tags=(psi0p3 psi0p7 psi0p9)

if [[ "${ASSOC_QUEUE_WORKER:-0}" != 1 ]] && tmux has-session -t "$queue_session" 2>/dev/null; then
  echo "already queued: $queue_session"
  exit 0
fi

mkdir -p "$log_dir"

launch_one() {
  local index="$1"
  local threshold="${thresholds[$index]}"
  local tag="${tags[$index]}"
  local session="assoc_s2_${tag}_20260814"
  local experiment="dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_${tag}_seed2_60m_20260814"
  local result_dir="$repo_root/onpolicy/scripts/results/mec/mappo/$experiment/run1"
  local log="$log_dir/${experiment}.log"

  if [[ -e "$result_dir" ]]; then
    echo "refusing duplicate experiment: $experiment" >&2
    return 1
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already active: $session"
    return 0
  fi
  tmux new-session -d -s "$session" \
    "cd '$repo_root' && exec bash '$runner' '$threshold' '$experiment' > '$log' 2>&1"
  echo "started: $session ($experiment)"
}

echo "queue=$queue_session waiting for current sensitivity sessions"
while :; do
  active=0
  for session in "${current_sessions[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      active=1
      break
    fi
  done
  if (( active == 0 )); then
    break
  fi
  sleep 60
done

for index in 0 1 2; do
  launch_one "$index"
done

echo "association-threshold queue launched"
