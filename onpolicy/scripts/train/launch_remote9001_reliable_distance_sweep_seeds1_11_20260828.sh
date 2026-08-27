#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runner="$repo_root/onpolicy/scripts/train/run_fixed600_200_seed.sh"
python_bin="${MARL_PYTHON:-/data/home/tanglanProf_user02/miniconda3/envs/marl/bin/python}"
log_dir="$repo_root/training_logs"
stamp="$(date +%Y%m%d_%H%M%S)"

jobs=(
  "0 1 260 260"
  "1 1 520 520"
  "2 1 780 780"
  "3 1 260 0"
  "4 11 260 260"
  "5 11 520 520"
  "6 11 780 780"
  "7 11 260 0"
)

experiment_name() {
  local seed="$1" dcom="$2" critic="$3"
  printf 'F600_B0_unreldp_reliable_nocurr_deadlineon_pa_s3_dcom%s_critic%s_seed%s_60m_cuda' \
    "$dcom" "$critic" "$seed"
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for job in "${jobs[@]}"; do
    read -r gpu seed dcom critic <<<"$job"
    name="$(experiment_name "$seed" "$dcom" "$critic")"
    printf 'PLAN\t%s\t%s\t%s\t%s\t%s\n' "$gpu" "$seed" "$dcom" "$critic" "$name"
  done
  exit 0
fi

[[ -x "$python_bin" ]] || { echo "Python not executable: $python_bin" >&2; exit 1; }
[[ -x "$runner" ]] || { echo "Runner not executable: $runner" >&2; exit 1; }
cd "$repo_root"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to launch from a repository with tracked changes" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "Refusing to launch because at least one GPU compute process is already active" >&2
  exit 1
fi

mkdir -p "$log_dir"
pids=()
for job in "${jobs[@]}"; do
  read -r gpu seed dcom critic <<<"$job"
  name="$(experiment_name "$seed" "$dcom" "$critic")"
  if pgrep -af "$name" >/dev/null; then
    echo "Experiment is already running: $name" >&2
    exit 1
  fi
  log="$log_dir/${name}_${stamp}.log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" MARL_PYTHON="$python_bin" \
    "$runner" reliable zero "$seed" "$name" off enabled 13.54 21.54 \
    "$critic" "$dcom" >"$log" 2>&1 </dev/null &
  pid=$!
  pids+=("$pid")
  printf 'STARTED\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$gpu" "$pid" "$seed" "$dcom" "$critic" "$name" "$log"
done

sleep 8
for pid in "${pids[@]}"; do
  kill -0 "$pid" 2>/dev/null || { echo "Launch failed for PID $pid" >&2; exit 1; }
done
