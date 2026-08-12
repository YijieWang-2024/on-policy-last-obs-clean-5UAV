#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 10|20|30|40 [experiment_name] [seed]" >&2
  exit 2
fi
v_max="$1"
seed="${3:-32}"
experiment_name="${2:-dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax${v_max}_seed${seed}_60m_20260812}"
case "$v_max" in
  10|20|30|40) ;;
  *) echo "v_max must be 10, 20, 30, or 40" >&2; exit 2 ;;
esac
case "$seed" in
  ''|*[!0-9]*) echo "seed must be a non-negative integer" >&2; exit 2 ;;
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

# Do not use shell errexit around Python's output: warnings are ordinary
# stderr records.  The exec exit code remains the authoritative failure signal.
exec "$python_bin" "$train_script" \
  --env_name mec \
  --algorithm_name mappo \
  --user_name wyj2 \
  --seed "$seed" \
  --share_policy \
  --n_training_threads 1 \
  --n_rollout_threads 64 \
  --n_UAVs 5 \
  --max_UAVs_in_neighbor 5 \
  --max_UAVs_obs_concat 5 \
  --neighbor_distance 520 \
  --neighbor_R 520 \
  --d_optimal 210 \
  --n_GUs 60 \
  --max_GUs_in_range 20 \
  --dynamic_md \
  --md_arrivals_min 5 \
  --md_arrivals_max 5 \
  --md_arrivals_per_region 1 4 \
  --hotspot_layout_mode episode_template4_600_200 \
  --five_uav_start_layout line \
  --uav_start_positions 110 180 220 180 330 180 440 180 400 400 \
  --md_lifetime_min 12 \
  --md_lifetime_max 12 \
  --x_max 600 \
  --x_min_uav 0 \
  --x_max_uav 600 \
  --y_min_uav 0 \
  --y_max_uav 600 \
  --x_min_gu 0 \
  --x_max_gu 600 \
  --y_min_gu 0 \
  --y_max_gu 600 \
  --fix_hotspot \
  --B 30000000 \
  --H_UAV 120 \
  --H_GU 1 \
  --alpha_r 32 \
  --beta_r 0.5 \
  --q1 1 \
  --gamma_r 26 \
  --delta_r 32 \
  --q2 1 \
  --epsilon_r 0 \
  --q3 1 \
  --lambda_r 0.000001 \
  --q4 1 \
  --mu_r 64 \
  --q5 1 \
  --Delta_t 0.5 \
  --F_m 20000000000 \
  --F_n 1500000000 \
  --D_min 200000 \
  --D_max 400000 \
  --C_min 675000000 \
  --C_max 800000000 \
  --delay_min 0.499 \
  --delay_max 0.5 \
  --Dis_min 3 \
  --Cover_R 120 \
  --w1 20 \
  --w2 1 \
  --p3 500 \
  --v_max "$v_max" \
  --mean_velocity 3 \
  --md_velocity_init_std 0.6 \
  --md_velocity_init_min_factor 0.4 \
  --md_velocity_init_max_factor 1.6 \
  --md_velocity_update_clip_min 0 \
  --md_velocity_update_clip_max 5 \
  --episode_length 400 \
  --num_env_steps 60000000 \
  --hidden_size 256 \
  --layer_N 2 \
  --lr 0.0001 \
  --critic_lr 0.0005 \
  --clip_param 0.15 \
  --ppo_epoch 4 \
  --num_mini_batch 1 \
  --entropy_coef 0 \
  --use_valuenorm \
  --shared_ret_norm \
  --state_is_k_hops \
  --all_uav_k_hops \
  --use_atten_critic \
  --ego_query_critic \
  --local_reward \
  --continuous_associate \
  --not_served_rew_to_nearest \
  --cartesian_flight \
  --actor_message_mode task_summary \
  --actor_message_pool receiver_gated_sum \
  --actor_message_contract absolute_raw_v2 \
  --spatial_flight_actor \
  --completion_priority_user_sort \
  --n_iterations 50 \
  --advantage_mode per_agent_noise \
  --experiment_name "$experiment_name" \
  --noise_scale 3.0 \
  --episode_layout_context \
  --episode_layout_context_units meters_v2
