#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runner="$repo_root/onpolicy/scripts/train/run_fixed600_200_seed.sh"
python_bin="${MARL_PYTHON:?set MARL_PYTHON to the CUDA training interpreter}"
log_dir="$repo_root/training_logs"
pid_dir="$log_dir/pids"
mkdir -p "$log_dir" "$pid_dir"

seeds=(1 11 21)
gpus=(0 1 2)
state_deadline_ms=10.175730095548
advantage_deadline_ms=18.175730095548

for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  gpu="${gpus[$index]}"
  experiment="F600_B0_unreldp_unreliable_mdgru_nocurr_deadlineon_a2atimelyp50_pa_s3_tb2048x10_seed${seed}_60m_cuda"
  log="$log_dir/${experiment}.log"
  pid_file="$pid_dir/${experiment}.pid"

  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "GPU $gpu already has a compute process; refusing to oversubscribe" >&2
    exit 1
  fi
  if pgrep -f -- "$experiment" >/dev/null; then
    echo "already active: $experiment" >&2
    exit 1
  fi

  nohup env CUDA_VISIBLE_DEVICES="$gpu" MARL_PYTHON="$python_bin" \
    bash "$runner" unreliable md_gru "$seed" "$experiment" off enabled \
    "$state_deadline_ms" "$advantage_deadline_ms" \
    >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  sleep 5
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "failed to start: $experiment; inspect $log" >&2
    exit 1
  fi
  echo "started seed=$seed gpu=$gpu pid=$pid log=$log"
done
