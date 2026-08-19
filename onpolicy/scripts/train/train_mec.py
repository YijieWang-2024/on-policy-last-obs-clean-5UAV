#!/usr/bin/env python
import sys
import os
# import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import json
import copy


# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
print(current_file)
# 获取项目根目录（假设main.py在项目根目录下的src目录中）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
print(project_root)
sys.path.insert(0, project_root)


from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from onpolicy.utils.unreliable_communication import resolve_communication_parameters


"""Train script for MEC."""
os.environ["WANDB_API_KEY"] = "2fc0c89cd0d6a198e3f53c9466aaf610699e6457"


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mec":
                from onpolicy.envs.mec.env_maker import WrappedMECEnv
                env_args = copy.deepcopy(all_args)
                env_args.uav_reset_curriculum_training = True
                env = WrappedMECEnv(args=env_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mec":
                from onpolicy.envs.mec.env_maker import WrappedMECEnv
                env_args = copy.deepcopy(all_args)
                env_args.uav_reset_curriculum_training = False
                env = WrappedMECEnv(args=env_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--n_UAVs', type=int, default=3, help="total number of uav-servers")
    parser.add_argument('--max_UAVs_in_neighbor', type=int, default=3, help="max number of UAVs in neighbor, in own obs' info")
    parser.add_argument(
        '--neighbor_distance', '--d_com', dest='neighbor_distance', type=float,
        default=None,
        help=(
            "A2A communication radius d_com in metres. If omitted, it is "
            "derived from --a2a_transmit_power_w; if both are omitted, d_com=520 m."
        ),
    )
    parser.add_argument("--neighbor_R", type=int, default=260, help="1-hop distance, Use to one-hop range drone position sharing")
    parser.add_argument('--d_optimal', type=float, default=210, help="(m), Distance between desired drones based on area size and number of drones")
    parser.add_argument('--n_GUs', type=int, default=20, help="total number of groud users")
    parser.add_argument('--max_GUs_in_range', type=int, default=20, help="max number of groud users in per UAV's range")
    parser.add_argument("--dynamic_md", action="store_true", default=False,
                        help="Use bounded exogenous MD arrivals and finite access lifetimes.")
    parser.add_argument("--md_arrivals_min", type=int, default=2,
                        help="Minimum candidate MD arrivals per slot in dynamic_md mode.")
    parser.add_argument("--md_arrivals_max", type=int, default=4,
                        help="Maximum candidate MD arrivals per slot in dynamic_md mode.")
    parser.add_argument("--md_arrivals_per_region", type=int, nargs=2, default=None,
                        metavar=("LOWER_LEFT", "UPPER_RIGHT"),
                        help="Fixed candidate arrivals in the two regional hotspot rectangles.")
    parser.add_argument(
        "--hotspot_layout_mode",
        choices=(
            "fixed_legacy", "episode_template4", "episode_template4_600_200",
            "episode_template12", "episode_template12_600_200",
            "episode_moving_template4"
        ),
        default="fixed_legacy",
        help=(
            "Regional dynamic-MD layout. fixed_legacy preserves the original "
            "lower-left/upper-right rectangles; episode_template4 samples one "
            "of four area-preserving diagonal layouts per episode; "
            "episode_template4_600_200 is the fixed 600m 200m/400m "
            "double-hotspot protocol; "
            "episode_template12 samples all ordered non-overlapping 700m "
            "corner pairs with a 175m/400m region pair; "
            "episode_template12_600_200 applies the same 12 ordered corner "
            "pairs to a 600m map with a 200m/400m region pair; "
            "episode_moving_template4 moves a hidden pair of birth regions "
            "continuously along a symmetric square route."
        ),
    )
    parser.add_argument(
        "--hotspot_layout_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help=(
            "Optional subset of a 12-layout mode's indices to sample uniformly. "
            "Use one index for a fixed layout."
        ),
    )
    parser.add_argument(
        "--episode_layout_context",
        action="store_true",
        default=False,
        help=(
            "Expose reset-time regional hotspot geometry to every actor and "
            "critic as 8 [xmin,xmax,ymin,ymax] values; no "
            "active-user or future-arrival information is included."
        ),
    )
    parser.add_argument(
        "--episode_layout_context_units",
        choices=("normalized_v1", "meters_v2"),
        default="normalized_v1",
        help=(
            "Representation of episode_layout_context. normalized_v1 preserves "
            "old checkpoints; meters_v2 emits raw map coordinates and lets "
            "ob_norm standardize them."
        ),
    )
    parser.add_argument(
        "--five_uav_start_layout",
        choices=("line", "staggered"),
        default="line",
        help=(
            "Initial geometry for the 700m five-UAV episode_template12 "
            "diagnostic. line uses y=70 for the lower four; staggered "
            "uses (300,140) and (400,140) for the last two."
        ),
    )
    parser.add_argument(
        "--uav_start_positions",
        type=float,
        nargs="+",
        default=None,
        metavar="XY",
        help=(
            "Optional explicit UAV reset coordinates as 2*n_UAVs values "
            "(x0 y0 x1 y1 ...). When supplied, this overrides the built-in "
            "reset geometry without changing the hotspot layout."
        ),
    )
    parser.add_argument("--md_lifetime_min", type=int, default=15,
                        help="Minimum active access lifetime in slots.")
    parser.add_argument("--md_lifetime_max", type=int, default=25,
                        help="Maximum active access lifetime in slots.")
    parser.add_argument("--x_max", type=int, default=500, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--x_min_uav", type=int, default=0, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--x_max_uav", type=int, default=600, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--y_min_uav", type=int, default=0, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--y_max_uav", type=int, default=600, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--x_min_gu", type=int, default=0, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--x_max_gu", type=int, default=300, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--y_min_gu", type=int, default=0, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--y_max_gu", type=int, default=300, help="A maximum map range of 1 km × 1 km.")
    parser.add_argument("--B", type=int, default=20 * 10 ** 6, help="Channel bandwidth in Hz, default is 20MHz")
    parser.add_argument(
        "--uav_resource_mode",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="Use one common UAV resource limit or per-UAV scale factors.",
    )
    parser.add_argument(
        "--uav_resource_scale_factors",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Per-UAV positive scale factors applied to both B and F_m; "
            "required when uav_resource_mode=heterogeneous."
        ),
    )
    parser.add_argument("--H_UAV", type=int, default=120, help="UAV server fixed flight height in meters")
    parser.add_argument("--H_GU", type=int, default=1, help="Ground user height in meters")
    parser.add_argument("--fix_hotspot", action='store_true', default=False, help=" if true, has a fixed hotpot at upper_right corner")
    parser.add_argument("--random_hotspot", action='store_true', default=False, help="If true, has a random hotpot")

    parser.add_argument("--alpha_r", type=float, default=0.0, help="Weight of coverage")      # When 1，覆盖用户数目的奖励。 正常0到10的数字。
    parser.add_argument("--beta_r", type=float, default=0.5, help="Weight of overlapping")  # When 1， 再减去0.5*重复用户数目的惩罚。
    parser.add_argument("--q1", type=float, default=1, help="Weight of R_coverage")

    parser.add_argument("--gamma_r", type=float, default=1, help="Weight of delta")         # When 1， 每个用户 0到0.5剩余时延的奖励
    parser.add_argument("--delta_r", type=float, default=1, help="Weight of over delta")    # When 1， 每个用户 超时-1的惩罚。
    parser.add_argument("--q2", type=float, default=1, help="Weight of R_task")

    parser.add_argument("--epsilon_r", type=float, default=0.0, help="Weight of optimal distance. When 1, Mean/optimal distance. Only -1 to 0 penalty")
    parser.add_argument("--q3", type=float, default=1, help="Weight of R_distribution")     # When 1，邻居到最优距离的均值/最优距离。仅仅-1到0的惩罚。

    parser.add_argument("--lambda_r", type=float, default=0.01, help="Weight of energy. When 0.01, cai scale to [0.63, 0.9]")
    parser.add_argument("--q4", type=float, default=1, help="Weight of R_fly")

    parser.add_argument("--mu_r", type=float, default=100, help="Weight of collision")      # When 100，每个碰撞-100的惩罚。
    parser.add_argument("--q5", type=float, default=1, help="Weight of R_collision")

    # parser.add_argument("--Delta_t", type=float, default=0.5,help="time interval of UAV server in seconds. step interval")
    # parser.add_argument("--F_m", type=int, default=15 * 10 ** 9, help="Maximum available computation resources of UAV server in 15 Gigacycles")
    # parser.add_argument("--F_n", type=int, default=1.5 * 10 ** 9, help="Maximum available computation resources of ground user in 1.5 Gigacycles")
    # parser.add_argument("--D_min", type=int, default=1 * 10 ** 6, help="Minimum data size of generated tasks in 1 MB")
    # parser.add_argument("--D_max", type=int, default=3 * 10 ** 6, help="Maximum data size of generated tasks in 3MB")
    # parser.add_argument("--C_min", type=int, default=300 * 10 ** 6, help="Minimum resource demand of generated tasks in 300 Megacycles")
    # parser.add_argument("--C_max", type=int, default=500 * 10 ** 6, help="Maximum resource demand of generated tasks in 500 Megacycles")
    # parser.add_argument("--delay_min", type=float, default=0.25, help="Minimum delay requirement of generated tasks in 0.25 seconds")
    # parser.add_argument("--delay_max", type=float, default=0.3, help="Maximum delay requirement of generated tasks in 0.3 seconds")

    parser.add_argument("--Delta_t", type=float, default=0.5,help="time interval of UAV server in seconds. step interval")
    parser.add_argument("--F_m", type=int, default=15 * 10 ** 9, help="Maximum available computation resources of UAV server in 15 Gigacycles")
    parser.add_argument("--F_n", type=int, default=1.5 * 10 ** 9, help="Maximum available computation resources of ground user in 1.5 Gigacycles")
    parser.add_argument("--D_min", type=int, default=0.2 * 10 ** 6, help="Minimum data size of generated tasks in 0.3 MB")
    parser.add_argument("--D_max", type=int, default=0.4 * 10 ** 6, help="Maximum data size of generated tasks in 0.5 MB")
    parser.add_argument("--C_min", type=int, default=675 * 10 ** 6, help="Minimum resource demand of generated tasks in 900 Megacycles")
    parser.add_argument("--C_max", type=int, default=800 * 10 ** 6, help="Maximum resource demand of generated tasks in 1100 Megacycles")
    parser.add_argument("--delay_min", type=float, default=0.499, help="Minimum delay requirement of generated tasks in 0.25 seconds")
    parser.add_argument("--delay_max", type=float, default=0.5, help="Maximum delay requirement of generated tasks in 0.3 seconds")
    # 设置的经验：delay*F_n=750*10**6，所以设置[C_min,C_max]围绕着750。 D_min和D_max设置的越小，越有助于卸载任务到无人机。

    parser.add_argument("--Dis_min", type=int, default=3, help="Minimum collision avoidance distance between UAV servers in meters")
    parser.add_argument("--Cover_R", type=int, default=120, help="2D communication coverage radius of UAV server in meters")
    parser.add_argument("--w1", type=float, default=20, help="Weight of delay part in reward calculation")
    parser.add_argument("--w2", type=float, default=1, help="Weight of energy consumption part in reward calculation")
    parser.add_argument("--p3", type=float, default=500, help="Penalty value for collision avoidance in reward calculation")
    parser.add_argument("--v_max", type=int, default=20, help="Maximum flight speed of UAV in m/s")
    parser.add_argument("--mean_velocity", type=float, default=5, help="Average moving speed of ground user in m/s")
    parser.add_argument(
        "--md_velocity_init_std",
        type=float,
        default=0.3,
        help="Standard deviation of newly initialized MD speeds in m/s",
    )
    parser.add_argument(
        "--md_velocity_init_min_factor",
        type=float,
        default=0.7,
        help="Lower initial MD speed bound as a multiple of mean_velocity",
    )
    parser.add_argument(
        "--md_velocity_init_max_factor",
        type=float,
        default=1.3,
        help="Upper initial MD speed bound as a multiple of mean_velocity",
    )
    parser.add_argument(
        "--md_velocity_update_clip_min",
        type=float,
        default=None,
        help="Optional lower bound for Gauss-Markov MD speed updates",
    )
    parser.add_argument(
        "--md_velocity_update_clip_max",
        type=float,
        default=None,
        help="Optional upper bound for Gauss-Markov MD speed updates",
    )

    # 占位，从tf代码里搞过来的，先不用这个，所以default改为了False。因为torch代码里有value_norm。
    # 在config.py里有"--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards."
    parser.add_argument("--ret_norm", action='store_false', default=True, help=" if true, scale the reward according to averaged discounted returns. do not use it with use_valuenorm==True")
    parser.add_argument("--ob-norm", action='store_false', default=True, help="If true, normalize the observation using running mean and std")
    parser.add_argument("--shared_ret_norm", action='store_true', default=False,
                        help="Use one pooled discounted-return scale for all UAVs while keeping observation statistics independent")

    parser.add_argument("--ob_state_with_timestep", action='store_true', default=False, help="If true, observation and state with timestep in first dimension")
    parser.add_argument("--use_kl_threshold", action='store_true', default=False, help="If true, r_mappo.py use kl threshold to stop update early")
    parser.add_argument("--perform_with_local_state", action='store_true', default=False, help="If true, self.state=self.obs in mec.py")
    parser.add_argument("--state_is_k_hops", action='store_true', default=False, help="If true, self.state=M_i^k in mec.py. paper_0707")
    parser.add_argument("--all_uav_k_hops", action='store_true', default=False,
                        help="If true, The state is a fixed form of all drones, corresponding to the positional complementary data of the k-hop neighbors")
    parser.add_argument("--concat_neighbor_obs", action='store_true', default=False, help="If true, concat neighbor's obs")
    parser.add_argument('--max_UAVs_obs_concat', type=int, default=3, help="max number of UAVs' obs to concat")
    parser.add_argument("--use_atten_actor", action='store_true', default=False, help="If true, use R_Actor_Attention")
    parser.add_argument("--use_atten_critic", action='store_true', default=False, help="If true, use R_Critic_Attention")
    parser.add_argument("--ego_query_critic", action='store_true', default=False,
                        help="For attention critics, keep the first (self) query output instead of mean-pooling all query outputs")
    parser.add_argument("--local_reward", action='store_true', default=False,
                        help="If true, calculate different reward in calculate_reward() of mec.py")
    parser.add_argument("--discrete_associate", action='store_true', default=False, help="If true, offloading_actions is Binary, and Network's output_layer is MultiBornrlia")
    parser.add_argument("--continuous_associate", action='store_true', default=False, help="If true, offloading_actions is continuous, and Network's output_layer is Gaussian")
    parser.add_argument(
        "--association_threshold", type=float, default=0.5,
        help=(
            "Threshold psi for converting each continuous association score into "
            "a candidate UAV-GU link. Scores greater than or equal to psi are kept."
        ),
    )
    parser.add_argument(
        "--disable_offload_deadline_filter",
        action="store_false",
        dest="offload_deadline_filter",
        default=True,
        help=(
            "Disable the pre-execution filter that cancels offloads whose predicted "
            "transmission plus execution delay misses the task deadline. Deadline "
            "success/failure and penalties in the reward remain enabled."
        ),
    )
    parser.add_argument("--nearest_associate", action='store_true', default=False, help="If true, without offloading_actions and Network's output_layer")
    parser.add_argument("--nearest_avail_actions", action='store_true', default=False, help="If true, get avail_actions for max_GUs_in_range and nearest itself")
    parser.add_argument("--not_process_action", action='store_true', default=False, help="If true, without process_action in env_maker.py and act.py")
    parser.add_argument("--fix_uav_pos", action='store_true', default=False, help="If true, uav's pos is fixed. action_space don't have fly_action")
    parser.add_argument("--cartesian_flight", action='store_true', default=False,
                        help="Use an unconstrained Gaussian (vx, vy) proposal followed by speed-disk projection")
    parser.add_argument("--actor_neighbor_obs", action='store_true', default=False,
                        help="Add one-hop relative UAV positions and masks to each actor observation")
    parser.add_argument(
        "--actor_message_mode",
        choices=["disabled", "zero", "geometry", "task_summary"],
        default="disabled",
        help=(
            "Radius-gated actor-only UAV message block. zero/geometry/task_summary "
            "share one fixed architecture; the block is removed from critic inputs."
        ),
    )
    parser.add_argument(
        "--actor_message_pool",
        choices=["mean", "receiver_gated_sum"],
        default="mean",
        help=(
            "Permutation-invariant aggregation for actor UAV messages. "
            "receiver_gated_sum independently gates each sender using the "
            "receiver local descriptor and uses a fixed n_UAVs-1 denominator."
        ),
    )
    parser.add_argument(
        "--actor_message_contract",
        choices=["relative_scaled_v1", "absolute_raw_v2"],
        default="relative_scaled_v1",
        help=(
            "Message feature contract. relative_scaled_v1 preserves old "
            "checkpoints and bypasses ob_norm for the full message block; "
            "absolute_raw_v2 uses absolute/raw physical fields, applies ob_norm "
            "to the first 9 values of each packet, and preserves only its mask."
        ),
    )
    parser.add_argument("--spatial_flight_actor", action='store_true', default=False,
                        help="Use an independent task-free DeepSets encoder for the flight action")
    parser.add_argument("--distance_only_user_sort", action='store_true', default=False,
                        help="Sort observed users only by UAV distance")
    parser.add_argument("--completion_priority_user_sort", action='store_true', default=False,
                        help="Sort all covered users by local infeasibility first, then by distance within each group")
    parser.add_argument("--uav_reset_curriculum", action='store_true', default=False,
                        help="Train with target-free random UAV resets, annealed to fixed resets by 50%%")
    parser.add_argument(
        "--uav_reset_curriculum_schedule",
        choices=["legacy", "p0p7_10m_25m"],
        default="legacy",
        help=(
            "UAV reset curriculum schedule. legacy is the existing 0.5/20M->50M "
            "schedule; p0p7_10m_25m holds p=0.7 to 10M and linearly anneals "
            "to zero at 25M."
        ),
    )
    parser.add_argument("--ave_resource", action='store_true', default=False, help="If true, allocate the resources of UAV equally to the connected users")
    parser.add_argument("--ave_bandwidth", action='store_true', default=False, help="If true, allocate the bandwidth resources only of UAV equally to the connected users")
    parser.add_argument("--not_served_rew_to_ave", action='store_true', default=False, help="If true, Rewards for unserved users are split evenly between covered drones")
    parser.add_argument("--not_served_rew_to_nearest", action='store_true', default=False, help="If true, Rewards for unserved users are given to the nearest drone that covers them")

    parser.add_argument("--average_local_advantage_timely", action='store_true', default=False, help="If true, Execute the average of the local advantage functions every timestep, rather than after sampling B buffer")
    parser.add_argument("--average_local_advantage", action='store_true', default=False, help="If true, Execute the average of the local advantage functions in the mec_runner.py")
    parser.add_argument("--whether_local_add_direct_ave_adv", action='store_true', default=False, help="If true, local add actual mean advantage in the mec_runner.py")
    parser.add_argument("--whether_local_add_ave_adadvantage", action='store_true', default=False, help="If true, local add averaged advantage in the mec_runner.py")
    parser.add_argument("--local_add_T_ave_adv", type=int, default=0, help="how many steps of the advantage to average")
    parser.add_argument("--average_neighbor_advantage", action='store_true', default=False, help="If true, Execute the average of all neighbor's Advantage in the mec_runner.py with neighbor_weights")
    parser.add_argument("--n_iterations", type=int, default=50, help="Number of iterations for weighted summation when finding the global advantage function")
    parser.add_argument(
        "--advantage_mode",
        choices=[
            "default",
            "local",
            "mixed_consensus",
            "pure_consensus",
            "externality_consensus",
            "legacy_noise",
            "per_agent_noise",
        ],
        default="default",
        help="Actor advantage contract. Consensus modes use the rollout-ending communication graph.",
    )
    parser.add_argument(
        "--noise_scale", type=float, default=0.12,
        help=(
            "Base Gaussian-noise scale for per_agent_noise. The final actor "
            "advantage is A_local + noise_magnitude_i * noise_scale * "
            "std(A_local) * N(0,1)."
        ),
    )
    parser.add_argument(
        "--consensus_alpha", type=float, default=0.5,
        help="Team-consensus weight for mixed_consensus: (1-alpha)*A_local + alpha*A_consensus",
    )
    parser.add_argument(
        "--externality_beta", type=float, default=0.1,
        help=(
            "Weight for externality_consensus: A_local + beta*(n_UAVs*"
            "A_consensus - c_self*A_local); PPO applies the final normalization"
        ),
    )
    parser.add_argument("--communication_mode", choices=("reliable", "unreliable"), default="reliable",
                        help="Reliable finite-round consensus or unreliable running-sum for the advantage-noise estimator")
    parser.add_argument("--running_sum_rounds", type=int, default=50,
                        help="Post-rollout running-sum rounds under unreliable communication")
    parser.add_argument(
        "--a2a_transmit_power_w", type=float, default=None,
        help=(
            "A2A transmit power P_c in watts. If omitted, it is derived from "
            "d_com; if supplied alone, d_com is derived from it."
        ),
    )
    parser.add_argument(
        "--a2a_distance_tolerance_m", type=float, default=5.0,
        help="Maximum permitted d_com mismatch when both distance and power are supplied.",
    )
    parser.add_argument("--a2a_bandwidth_hz", type=float, default=2e6)
    parser.add_argument("--a2a_reference_gain_db", type=float, default=-38.46)
    parser.add_argument("--a2a_reference_distance_m", type=float, default=1.0)
    parser.add_argument("--a2a_path_loss_exponent", type=float, default=2.2)
    parser.add_argument("--a2a_rician_k_db", type=float, default=10.0)
    parser.add_argument("--a2a_noise_psd_dbm_hz", type=float, default=-130.0)
    parser.add_argument("--a2a_spectral_efficiency", type=float, default=0.5)
    parser.add_argument("--a2a_decoding_threshold_db", type=float, default=-0.5)
    parser.add_argument("--a2a_gamma_shape", type=float, default=2.5)
    parser.add_argument("--a2a_gamma_scale_ms", type=float, default=1.0)
    parser.add_argument("--state_payload_bits", type=float, default=8000.0)
    parser.add_argument("--state_deadline_ms", type=float, default=13.54)
    parser.add_argument(
        "--state_reconstruction", choices=("zero", "last_obs", "md_gru"),
        default="zero",
        help="Missing Type-S state handling in the main training process",
    )
    parser.add_argument(
        "--critic_md_metadata", action="store_true", default=False,
        help="Append record-valid, task-valid, and information-age features to critic MD slots",
    )
    parser.add_argument("--md_gru_hidden_dim", type=int, default=64)
    parser.add_argument("--md_gru_lr", type=float, default=1e-3)
    parser.add_argument("--md_gru_epochs", type=int, default=4)
    parser.add_argument("--md_gru_batch_size", type=int, default=512)
    parser.add_argument("--md_gru_max_samples", type=int, default=32768)
    parser.add_argument(
        "--md_gru_train_samples", type=int, default=512,
        help="Maximum replay sequences trained per receiver and PPO update.",
    )
    parser.add_argument(
        "--md_gru_min_ready_samples", type=int, default=512,
        help="Receiver-local replay size required before GRU predictions replace last-observation fallback.",
    )
    parser.add_argument(
        "--md_prediction_loss_coef", type=float, default=1.0,
        help=(
            "Deprecated compatibility option. The MD estimator has a separate "
            "optimizer and its supervised loss is not mixed with PPO gradients."
        ),
    )
    parser.add_argument("--advantage_payload_bits", type=float, default=16000.0)
    parser.add_argument("--advantage_deadline_ms", type=float, default=21.54)
    parser.add_argument("--whether_average_network_parameters", action='store_true', default=False, help="If true, Execute the average of all network's parameters in the mec_runner.py")
    # 测试使用tanh来处理下动作会不会有影响。
    parser.add_argument("--tanh_gaussian", action='store_true', default=False, help="If true, act.py use (tanh(u)+1)/2 to process action")
    parser.add_argument("--average_network_parameters_interval", type=int, default=10, help="interval of average_network_parameters, /episodes")
    parser.add_argument("--ob_state_with_id", action='store_true', default=False, help="whether the state and obs with agent_id in env_maker.py")

    default_parser = parser.parse_args([])
    assert default_parser.alpha_r==0 and default_parser.epsilon_r==0, "这两个参数要默认为0。哪个输入了新的值，不为零，就是考虑了对应的惩罚。"
    assert default_parser.local_add_T_ave_adv == 0, "这个参数要默认为0，不为0的话，就是使用了这样一个平均方式。"
    all_args = parser.parse_known_args(args)[0]
    try:
        resolve_communication_parameters(all_args)
    except ValueError as error:
        parser.error(str(error))
    if not 0.0 <= all_args.consensus_alpha <= 1.0:
        parser.error("--consensus_alpha must be in [0, 1]")
    if not 0.0 <= all_args.externality_beta <= 1.0:
        parser.error("--externality_beta must be in [0, 1]")
    if all_args.noise_scale < 0.0:
        parser.error("--noise_scale must be non-negative")
    if not 0.0 <= all_args.association_threshold <= 1.0:
        parser.error("--association_threshold must be in [0, 1]")
    if all_args.md_prediction_loss_coef <= 0.0:
        parser.error("--md_prediction_loss_coef must be positive")
    if all_args.md_gru_hidden_dim <= 0 or all_args.md_gru_lr <= 0:
        parser.error("--md_gru_hidden_dim and --md_gru_lr must be positive")
    if all_args.md_gru_epochs <= 0 or all_args.md_gru_batch_size <= 0:
        parser.error("--md_gru_epochs and --md_gru_batch_size must be positive")
    if all_args.md_gru_max_samples <= 0 or all_args.md_gru_train_samples <= 0:
        parser.error("--md_gru_max_samples and --md_gru_train_samples must be positive")
    if not 0 < all_args.md_gru_min_ready_samples <= all_args.md_gru_max_samples:
        parser.error(
            "--md_gru_min_ready_samples must be in [1, --md_gru_max_samples]"
        )
    if all_args.ego_query_critic and not all_args.use_atten_critic:
        parser.error("--ego_query_critic requires --use_atten_critic")
    if all_args.shared_ret_norm and not all_args.ret_norm:
        parser.error("--shared_ret_norm requires ret_norm to be enabled")
    return all_args


def main(args):
    # Keep torch out of Windows spawn-based environment workers. They import
    # this module to reconstruct the subprocess target but never train models.
    import torch
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.critic_md_metadata = bool(
        all_args.critic_md_metadata or all_args.state_reconstruction != "zero"
    )
    if all_args.state_reconstruction != "zero" and all_args.share_policy:
        raise ValueError(
            "state reconstruction currently requires the separated runner; "
            "pass --share_policy (this legacy flag selects separate policies)"
        )
    assert all_args.use_valuenorm != all_args.ret_norm, 'torch中的valuenorm默认True和tf的retnorm默认False不能一起用'

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False         # naive的RNN一直是False，沒True過，不知道幹嘛的
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec" or all_args.algorithm_name=="belief_transformer":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False           # 默認是True，使用RNN網絡
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False          # 默認是True   用不用share_obs。
        all_args.use_recurrent_policy = False  # 默認是True，使用RNN網絡
    elif all_args.algorithm_name == "happo" or all_args.algorithm_name == "hatrpo":
        # can or cannot use recurrent network?
        print("using", all_args.algorithm_name, 'without recurrent network')
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    all_args.device = str(device)   # 添加使用的device，记录下来。

    if all_args.algorithm_name == "rmappo":
        # pytorch1.5.1的rnn部分与cuDNN存在冲突，只能这样。虽然降低了性能，但是还可以正常运行。
        torch.backends.cudnn.enabled = False

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                        #  group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
        all_args = wandb.config     # for wandb sweep
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # Save arguments to file
    with open(os.path.join(str(run_dir), 'args.json'), 'w') as f:
        json.dump(vars(all_args), f, indent=4, sort_keys=True, default=str)
    print(f"Arguments saved to {os.path.join(str(run_dir), 'args.json')}")

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.n_UAVs

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,   # 在separated的base_runner里边用到了，提示agent的数目
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.mec_runner import MECRunner as Runner
    else:
        from onpolicy.runner.separated.mec_runner import MECRunner as Runner

    if all_args.algorithm_name == "happo" or all_args.algorithm_name == "hatrpo":
        from onpolicy.runner.separated.smac_runner import SMACRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
