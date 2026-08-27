#!/usr/bin/env bash
set -euo pipefail

communication_mode="${1:?usage: $0 reliable|unreliable zero|last_obs|md_gru seed experiment_name [off|p0p7_10m_25m] [enabled|disabled] [state_deadline_ms] [advantage_deadline_ms] [critic_neighbor_distance] [communication_distance]}"
state_reconstruction="${2:?missing state reconstruction}"
seed="${3:?missing seed}"
experiment_name="${4:?missing experiment name}"
curriculum="${5:-off}"
deadline_filter="${6:-disabled}"
state_deadline_ms="${7:-13.54}"
advantage_deadline_ms="${8:-21.54}"
critic_neighbor_distance="${9:-}"
communication_distance="${10:-520}"

case "$communication_mode:$state_reconstruction" in
  reliable:zero|unreliable:zero|unreliable:last_obs|unreliable:md_gru) ;;
  *) echo "invalid mode/reconstruction pair: $communication_mode/$state_reconstruction" >&2; exit 2 ;;
esac
case "$curriculum" in
  off|p0p7_10m_25m) ;;
  *) echo "invalid curriculum: $curriculum" >&2; exit 2 ;;
esac
case "$deadline_filter" in
  enabled|disabled) ;;
  *) echo "invalid deadline filter: $deadline_filter" >&2; exit 2 ;;
esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${MARL_PYTHON:-/home/test/miniconda3/envs/marl/bin/python}"
train_script="$repo_root/onpolicy/scripts/train/train_mec.py"

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

args=(
  "$train_script"
  --env_name mec
  --algorithm_name mappo
  --experiment_name "$experiment_name"
  --user_name test
  --seed "$seed"
  --share_policy
  --n_UAVs 5
  --n_GUs 60
  --max_GUs_in_range 20
  --dynamic_md
  --md_arrivals_min 5
  --md_arrivals_max 5
  --md_arrivals_per_region 1 4
  --hotspot_layout_mode episode_template4_600_200
  --five_uav_start_layout line
  --uav_start_positions 110 180 220 180 330 180 440 180 400 400
  --md_lifetime_min 12
  --md_lifetime_max 12
  --x_max 600
  --x_min_uav 0 --x_max_uav 600
  --y_min_uav 0 --y_max_uav 600
  --x_min_gu 0 --x_max_gu 600
  --y_min_gu 0 --y_max_gu 600
  --fix_hotspot
  --max_UAVs_in_neighbor 5
  --max_UAVs_obs_concat 5
  --neighbor_R 520
  --state_is_k_hops
  --all_uav_k_hops
  --local_reward
  --continuous_associate
  --association_threshold 0.5
  --uav_resource_mode homogeneous
  --not_served_rew_to_nearest
  --n_rollout_threads 64
  --n_training_threads 1
  --episode_length 400
  --num_env_steps 60000000
  --hidden_size 256
  --layer_N 2
  --lr 0.0001
  --critic_lr 0.0005
  --clip_param 0.15
  --gamma 0.99
  --ppo_epoch 4
  --num_mini_batch 1
  --entropy_coef 0
  --B 30000000
  --F_m 20000000000
  --v_max 30
  --mean_velocity 3
  --md_velocity_init_std 0.6
  --md_velocity_init_min_factor 0.4
  --md_velocity_init_max_factor 1.6
  --md_velocity_update_clip_min 0
  --md_velocity_update_clip_max 5
  --alpha_r 32
  --gamma_r 26
  --delta_r 32
  --lambda_r 0.000001
  --mu_r 64
  --use_valuenorm
  --use_atten_critic
  --cartesian_flight
  --actor_message_mode disabled
  --actor_message_pool receiver_gated_sum
  --actor_message_contract absolute_raw_v2
  --spatial_flight_actor
  --completion_priority_user_sort
  --episode_layout_context
  --episode_layout_context_units meters_v2
  --advantage_mode per_agent_noise
  --consensus_alpha 0.5
  --externality_beta 0.1
  --communication_mode "$communication_mode"
  --neighbor_distance "$communication_distance"
  --a2a_rician_k_db 10
  --a2a_distance_tolerance_m 5
  --noise_scale 3
  --ego_query_critic
  --shared_ret_norm
)

if [[ "$deadline_filter" == disabled ]]; then
  args+=(--disable_offload_deadline_filter)
fi

if [[ -n "$critic_neighbor_distance" ]]; then
  args+=(--critic_neighbor_distance "$critic_neighbor_distance")
fi

if [[ "$curriculum" != off ]]; then
  args+=(--uav_reset_curriculum --uav_reset_curriculum_schedule "$curriculum")
fi

if [[ "$communication_mode" == reliable ]]; then
  args+=(--n_iterations 50)
else
  args+=(
    --critic_md_metadata
    --running_sum_rounds 50
    --state_payload_bits 8000
    --state_deadline_ms "$state_deadline_ms"
    --advantage_payload_bits 16000
    --advantage_deadline_ms "$advantage_deadline_ms"
    --state_reconstruction "$state_reconstruction"
  )
  if [[ "$state_reconstruction" == md_gru ]]; then
    args+=(
      --md_gru_hidden_dim 64
      --md_gru_lr 0.001
      --md_gru_target_batch_size 2048
      --md_gru_batches_per_rollout 10
      --md_gru_epochs 1
      --md_gru_min_ready_samples 2048
      --md_prediction_loss_coef 1
    )
  fi
fi

cd "$repo_root"
exec "$python_bin" "${args[@]}"
