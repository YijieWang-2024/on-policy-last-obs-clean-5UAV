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
    parser.add_argument('--neighbor_distance', type=float, default=260, help="(240+20)*k=260,520,780,1040(m), the neighbor distance of UAVs in meters")
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
                        help="Fixed candidate arrivals in the two 5-UAV hotspot rectangles.")
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

    # 占位，从tf代码里搞过来的，先不用这个，所以default改为了False。因为torch代码里有value_norm。
    # 在config.py里有"--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards."
    parser.add_argument("--ret_norm", action='store_false', default=True, help=" if true, scale the reward according to averaged discounted returns. do not use it with use_valuenorm==True")
    parser.add_argument("--ob-norm", action='store_false', default=True, help="If true, normalize the observation using running mean and std")

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
    parser.add_argument("--local_reward", action='store_true', default=False,
                        help="If true, calculate different reward in calculate_reward() of mec.py")
    parser.add_argument("--discrete_associate", action='store_true', default=False, help="If true, offloading_actions is Binary, and Network's output_layer is MultiBornrlia")
    parser.add_argument("--continuous_associate", action='store_true', default=False, help="If true, offloading_actions is continuous, and Network's output_layer is Gaussian")
    parser.add_argument("--nearest_associate", action='store_true', default=False, help="If true, without offloading_actions and Network's output_layer")
    parser.add_argument("--nearest_avail_actions", action='store_true', default=False, help="If true, get avail_actions for max_GUs_in_range and nearest itself")
    parser.add_argument("--not_process_action", action='store_true', default=False, help="If true, without process_action in env_maker.py and act.py")
    parser.add_argument("--fix_uav_pos", action='store_true', default=False, help="If true, uav's pos is fixed. action_space don't have fly_action")
    parser.add_argument("--cartesian_flight", action='store_true', default=False,
                        help="Use an unconstrained Gaussian (vx, vy) proposal followed by speed-disk projection")
    parser.add_argument("--actor_neighbor_obs", action='store_true', default=False,
                        help="Add one-hop relative UAV positions and masks to each actor observation")
    parser.add_argument("--spatial_flight_actor", action='store_true', default=False,
                        help="Use an independent task-free DeepSets encoder for the flight action")
    parser.add_argument("--distance_only_user_sort", action='store_true', default=False,
                        help="Sort observed users only by UAV distance")
    parser.add_argument("--completion_priority_user_sort", action='store_true', default=False,
                        help="Sort all covered users by local infeasibility first, then by distance within each group")
    parser.add_argument("--uav_reset_curriculum", action='store_true', default=False,
                        help="Train with target-free random UAV resets, annealed to fixed resets by 50%%")
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
    parser.add_argument("--communication_mode", choices=("reliable", "unreliable"), default="reliable",
                        help="Reliable finite-round consensus or unreliable running-sum for the advantage-noise estimator")
    parser.add_argument("--running_sum_rounds", type=int, default=30,
                        help="Post-rollout running-sum rounds under unreliable communication")
    parser.add_argument("--a2a_transmit_power_w", type=float, default=2.0)
    parser.add_argument("--a2a_bandwidth_hz", type=float, default=2e6)
    parser.add_argument("--a2a_reference_gain_db", type=float, default=-38.46)
    parser.add_argument("--a2a_reference_distance_m", type=float, default=1.0)
    parser.add_argument("--a2a_path_loss_exponent", type=float, default=2.2)
    parser.add_argument("--a2a_rician_k_db", type=float, default=6.0)
    parser.add_argument("--a2a_noise_psd_dbm_hz", type=float, default=-130.0)
    parser.add_argument("--a2a_spectral_efficiency", type=float, default=0.5)
    parser.add_argument("--a2a_decoding_threshold_db", type=float, default=-0.5)
    parser.add_argument("--a2a_gamma_shape", type=float, default=2.5)
    parser.add_argument("--a2a_gamma_scale_ms", type=float, default=1.0)
    parser.add_argument("--state_payload_bits", type=float, default=8000.0)
    parser.add_argument("--state_deadline_ms", type=float, default=13.54)
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
    return all_args


def main(args):
    # Keep torch out of Windows spawn-based environment workers. They import
    # this module to reconstruct the subprocess target but never train models.
    import torch
    parser = get_config()
    all_args = parse_args(args, parser)
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
