import widowx_env
import gym
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv
from stable_baselines.common import set_global_seeds
import stable_baselines
import numpy as np
import utils.import_envs  # pytype: disable=import-error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pkg_resources
import importlib
import warnings
import pandas as pd
import time
from collections import OrderedDict

# added by Pierre
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# from rlkit.envs.wrappers import NormalizedBoxEnv


# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                        type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=False,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--load-best', action='store_true', default=False,
                        help='Load best model instead of last model if available')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[],
                        help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument(
        '--render-pybullet', help='Slow down Pybullet simulation to render', default=False)  # added by Pierre
    parser.add_argument('--random-pol', help='Random policy', default=False)  # added by Pierre
    args = parser.parse_args()

    plot_bool = False
    plot_dim = 2
    log_bool = False

    if plot_bool:

        if plot_dim == 2:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(5, 10))
        elif plot_dim == 3:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

    if log_bool:
        log_df = pd.DataFrame()
        log_dict = OrderedDict()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    if not args.random_pol:  # added by Pierre
        model_path = find_saved_model(algo, log_path, env_id, load_best=args.load_best)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        args.n_envs = 1

    set_global_seeds(args.seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(
        stats_path, norm_reward=args.norm_reward, test_mode=True)

    log_dir = args.reward_log if args.reward_log != '' else None

    env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                          should_render=not args.no_render,
                          hyperparams=hyperparams)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    if not args.random_pol:  # added by Pierre
        model = ALGOS[algo].load(model_path, env=load_env)

    # if not args.no_render:
        # env.render(mode="human")  # added by Pierre (to work with ReachingJaco-v1)

    obs = env.reset()

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in [
        'dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    episode = 0
    success_threshold_50 = 0.05
    success_threshold_20 = 0.02
    success_threshold_10 = 0.01
    success_threshold_5 = 0.005
    success_threshold_2 = 0.002
    success_threshold_1 = 0.001
    success_threshold_05 = 0.0005
    episode_success_list_50 = []
    episode_success_list_20 = []
    episode_success_list_10 = []
    episode_success_list_5 = []
    episode_success_list_2 = []
    episode_success_list_1 = []
    episode_success_list_05 = []
    success_list_50 = []
    success_list_20 = []
    success_list_10 = []
    success_list_5 = []
    success_list_2 = []
    success_list_1 = []
    success_list_05 = []

    # For HER, monitor success rate
    successes = []
    state = None

    ##############
    def calc_ep_success(success_threshold, episode_success_list):
        """update episode_success_list for the current timestep"""

        if infos[0]['total_distance'] <= success_threshold:
            episode_success_list.append(1)
        else:
            episode_success_list.append(0)
        return episode_success_list

    def calc_success_list(episode_success_list, success_list):
        """ Append the last element of the episode success list when episode is done """
        success_list.append(episode_success_list[-1])
        return success_list

    def calc_reach_time(episode_success_list):
        """ If the episode is successful and it starts from an unsucessful step, calculate reach time """
        reachtime_list = []
        if episode_success_list[-1] == True and episode_success_list[0] == False:
            idx = 0
            while episode_success_list[idx] == False:
                idx += 1
            reachtime_list.append(idx)
        return reachtime_list
    ##############

    for t in range(args.n_timesteps):
        if args.random_pol:
            # Random Agent
            action = [env.action_space.sample()]
        else:
            action, state = model.predict(obs, state=state, deterministic=deterministic)

        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)

        if args.render_pybullet:
            time.sleep(1./30.)     # added by Pierre (slow down Pybullet for rendering)

        # update episode success list
        episode_success_list_50 = calc_ep_success(success_threshold_50, episode_success_list_50)
        episode_success_list_20 = calc_ep_success(success_threshold_20, episode_success_list_20)
        episode_success_list_10 = calc_ep_success(success_threshold_10, episode_success_list_10)
        episode_success_list_5 = calc_ep_success(success_threshold_5, episode_success_list_5)
        episode_success_list_2 = calc_ep_success(success_threshold_2, episode_success_list_2)
        episode_success_list_1 = calc_ep_success(success_threshold_1, episode_success_list_1)
        episode_success_list_05 = calc_ep_success(success_threshold_05, episode_success_list_05)

        episode_reward += reward[0]
        ep_len += 1

        if plot_bool:
            goal = infos[0]['goal position']
            tip = infos[0]['tip position']

            if plot_dim == 2:
                ax1.cla()
                ax1.plot(goal[0], goal[2], marker='x', color='b',
                         linestyle='', markersize=10, label="goal", mew=3)
                ax1.plot(tip[0], tip[2], marker='o', color='r',
                         linestyle='', markersize=10, label="end effector")

                circ_1_50 = plt.Circle((goal[0], goal[2]), radius=success_threshold_50,
                                       edgecolor='g', facecolor='w', linestyle='--', label="50 mm")
                circ_1_20 = plt.Circle((goal[0], goal[2]), radius=success_threshold_20,
                                       edgecolor='b', facecolor='w', linestyle='--', label="20 mm")
                circ_1_10 = plt.Circle((goal[0], goal[2]), radius=success_threshold_10,
                                       edgecolor='m', facecolor='w', linestyle='--', label="10 mm")
                circ_1_5 = plt.Circle((goal[0], goal[2]), radius=success_threshold_5,
                                      edgecolor='r', facecolor='w', linestyle='--', label="5 mm")
                ax1.add_patch(circ_1_50)
                ax1.add_patch(circ_1_20)
                ax1.add_patch(circ_1_10)
                ax1.add_patch(circ_1_5)

                ax1.set_xlim([-0.25, 0.25])
                ax1.set_ylim([0, 0.5])
                ax1.set_xlabel("x (m)", fontsize=15)
                ax1.set_ylabel("z (m)", fontsize=15)

                ax2.cla()
                ax2.plot(goal[1], goal[2], marker='x', color='b',
                         linestyle='', markersize=10, mew=3)
                ax2.plot(tip[1], tip[2], marker='o', color='r', linestyle='', markersize=10)

                circ_2_50 = plt.Circle(
                    (goal[1], goal[2]), radius=success_threshold_50, edgecolor='g', facecolor='w', linestyle='--')
                circ_2_20 = plt.Circle(
                    (goal[1], goal[2]), radius=success_threshold_20, edgecolor='b', facecolor='w', linestyle='--')
                circ_2_10 = plt.Circle(
                    (goal[1], goal[2]), radius=success_threshold_10, edgecolor='m', facecolor='w', linestyle='--')
                circ_2_5 = plt.Circle((goal[1], goal[2]), radius=success_threshold_5,
                                      edgecolor='r', facecolor='w', linestyle='--')
                ax2.add_patch(circ_2_50)
                ax2.add_patch(circ_2_20)
                ax2.add_patch(circ_2_10)
                ax2.add_patch(circ_2_5)

                ax2.set_xlim([-0.25, 0.25])
                ax2.set_ylim([0, 0.5])
                ax2.set_xlabel("y (m)", fontsize=15)
                ax2.set_ylabel("z (m)", fontsize=15)

                ax1.legend(loc='upper left', bbox_to_anchor=(
                    0, 1.2), ncol=3, fancybox=True, shadow=True)

            elif plot_dim == 3:
                ax.cla()
                ax.plot([tip[0]], [tip[1]], zs=[tip[2]], marker='x', color='b')
                ax.plot([goal[0]], [goal[1]], zs=[goal[2]], marker='o', color='r', linestyle="None")
                ax.set_xlim([-0.2, 0.2])
                ax.set_ylim([-0.2, 0.2])
                ax.set_zlim([0, 0.5])
                ax.set_xlabel("x (m)", fontsize=15)
                ax.set_ylabel("y (m)", fontsize=15)
                ax.set_zlabel("z (m)", fontsize=15)

            fig.suptitle("timestep "+str(ep_len)+" | distance to target: " +
                         str(round(infos[0]['total_distance']*1000, 1))+" mm")
            plt.pause(0.01)
            # plt.show()

        if log_bool:

            log_dict['episode'] = episode
            log_dict['timestep'] = t
            log_dict['action_1'] = action[0][0]
            log_dict['action_2'] = action[0][1]
            log_dict['action_3'] = action[0][2]
            log_dict['action_4'] = action[0][3]
            log_dict['action_5'] = action[0][4]
            log_dict['action_6'] = action[0][5]
            log_dict['current_joint_pos_1'] = infos[0]['current_joint_pos'][0]
            log_dict['current_joint_pos_2'] = infos[0]['current_joint_pos'][1]
            log_dict['current_joint_pos_3'] = infos[0]['current_joint_pos'][2]
            log_dict['current_joint_pos_4'] = infos[0]['current_joint_pos'][3]
            log_dict['current_joint_pos_5'] = infos[0]['current_joint_pos'][4]
            log_dict['current_joint_pos_6'] = infos[0]['current_joint_pos'][5]
            log_dict['new_joint_pos_1'] = infos[0]['new_joint_pos'][0]
            log_dict['new_joint_pos_2'] = infos[0]['new_joint_pos'][1]
            log_dict['new_joint_pos_3'] = infos[0]['new_joint_pos'][2]
            log_dict['new_joint_pos_4'] = infos[0]['new_joint_pos'][3]
            log_dict['new_joint_pos_5'] = infos[0]['new_joint_pos'][4]
            log_dict['new_joint_pos_6'] = infos[0]['new_joint_pos'][5]
            log_dict['joint_vel_1'] = infos[0]['joint_vel'][0]
            log_dict['joint_vel_2'] = infos[0]['joint_vel'][1]
            log_dict['joint_vel_3'] = infos[0]['joint_vel'][2]
            log_dict['joint_vel_4'] = infos[0]['joint_vel'][3]
            log_dict['joint_vel_5'] = infos[0]['joint_vel'][4]
            log_dict['joint_vel_6'] = infos[0]['joint_vel'][5]
            log_dict['joint1_min'] = -3.1
            log_dict['joint1_max'] = 3.1
            log_dict['joint2_min'] = -1.571
            log_dict['joint2_max'] = 1.571
            log_dict['joint3_min'] = -1.571
            log_dict['joint3_max'] = 1.571
            log_dict['joint4_min'] = -1.745
            log_dict['joint4_max'] = 1.745
            log_dict['joint5_min'] = -2.617
            log_dict['joint5_max'] = 2.617
            log_dict['joint6_min'] = 0.003
            log_dict['joint6_max'] = 0.03
            log_dict['action_low1'] = env.action_space.low[0]
            log_dict['action_low2'] = env.action_space.low[1]
            log_dict['action_low3'] = env.action_space.low[2]
            log_dict['action_low4'] = env.action_space.low[3]
            log_dict['action_low5'] = env.action_space.low[4]
            log_dict['action_low6'] = env.action_space.low[5]
            log_dict['action_high1'] = env.action_space.high[0]
            log_dict['action_high2'] = env.action_space.high[1]
            log_dict['action_high3'] = env.action_space.high[2]
            log_dict['action_high4'] = env.action_space.high[3]
            log_dict['action_high5'] = env.action_space.high[4]
            log_dict['action_high6'] = env.action_space.high[5]
            log_dict['reward'] = reward[0]
            log_dict['return'] = episode_reward
            log_dict['dist'] = infos[0]['total_distance']
            log_dict['target_x'] = infos[0]['goal position'][0]
            log_dict['target_y'] = infos[0]['goal position'][1]
            log_dict['target_z'] = infos[0]['goal position'][2]
            log_dict['tip_y'] = infos[0]['tip position'][1]
            log_dict['tip_x'] = infos[0]['tip position'][0]
            log_dict['tip_z'] = infos[0]['tip position'][2]
            log_dict['done'] = done[0]
            # log_dict['obs'] = obs
            # log_dict['obs_space_low'] = env.observation_space.low
            # log_dict['obs_space_high'] = env.observation_space.high

            log_df = log_df.append(log_dict, ignore_index=True)

        # if not args.no_render:
        #     env.render('human')

        if args.n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                    print("Atari Episode Length", episode_infos['l'])

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print("Episode nb: {} | Episode Reward: {:.2f} | Episode Length: {}".format(
                    episode, episode_reward, ep_len))
                # print("Episode Length", ep_len) # commented by Pierre
                state = None
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)

                # append the last element of the episode success list when episode is done
                success_list_50 = calc_success_list(episode_success_list_50, success_list_50)
                success_list_20 = calc_success_list(episode_success_list_20, success_list_20)
                success_list_10 = calc_success_list(episode_success_list_10, success_list_10)
                success_list_5 = calc_success_list(episode_success_list_5, success_list_5)
                success_list_2 = calc_success_list(episode_success_list_2, success_list_2)
                success_list_1 = calc_success_list(episode_success_list_1, success_list_1)
                success_list_05 = calc_success_list(episode_success_list_05, success_list_05)

                # If the episode is successful and it starts from an unsucessful step, calculate reach time
                reachtime_list_50 = calc_reach_time(episode_success_list_50)
                reachtime_list_20 = calc_reach_time(episode_success_list_20)
                reachtime_list_10 = calc_reach_time(episode_success_list_10)
                reachtime_list_5 = calc_reach_time(episode_success_list_5)
                reachtime_list_2 = calc_reach_time(episode_success_list_2)
                reachtime_list_1 = calc_reach_time(episode_success_list_1)
                reachtime_list_05 = calc_reach_time(episode_success_list_05)

                if log_bool:
                    log_df = log_df[log_dict.keys()]  # sort columns

                    # add estimated tip velocity and acceleration (according to the documentation, 1 timestep = 240 Hz)
                    log_df['est_vel'] = log_df['dist'].diff()*240
                    log_df['est_vel'].loc[0] = 0    # initial velocity is 0
                    log_df['est_acc'] = log_df['est_vel'].diff()*240
                    log_df['est_acc'].loc[0] = 0    # initial acceleration is 0

                    log_df.to_csv(log_path+"/res_episode_"+str(episode)+".csv", index=False)  # slow
                    # log_df.to_pickle(log_path+"/res_episode_"+str(episode)+".pkl")  # fast

                # reset for new episode
                episode_reward = 0.0
                ep_len = 0
                episode_success_list_50 = []
                episode_success_list_20 = []
                episode_success_list_10 = []
                episode_success_list_5 = []
                episode_success_list_2 = []
                episode_success_list_1 = []
                episode_success_list_05 = []
                episode += 1

            # Reset also when the goal is achieved when using HER
            if done or infos[0].get('is_success', False):
                if args.algo == 'her' and args.verbose > 1:
                    print("Success?", infos[0].get('is_success', False))
                # Alternatively, you can add a check to wait for the end of the episode
                # if done:
                obs = env.reset()
                if args.algo == 'her':
                    successes.append(infos[0].get('is_success', False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards),
                                                      np.std(episode_rewards)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(
            success_threshold_50, np.mean(success_list_50), np.mean(reachtime_list_50)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(
            success_threshold_20, np.mean(success_list_20), np.mean(reachtime_list_20)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(
            success_threshold_10, np.mean(success_list_10), np.mean(reachtime_list_10)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(
            success_threshold_5, np.mean(success_list_5), np.mean(reachtime_list_5)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(
            success_threshold_2, np.mean(success_list_2), np.mean(reachtime_list_2)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(
            success_threshold_1, np.mean(success_list_1), np.mean(reachtime_list_1)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(
            success_threshold_05, np.mean(success_list_05), np.mean(reachtime_list_05)))

        # added by Pierre
        print("path:", log_path)
        d = {
            "Eval mean reward": np.mean(episode_rewards),
            "Eval std": np.std(episode_rewards),
            "success ratio 50mm": np.mean(success_list_50),
            "Average reach time 50mm": np.mean(reachtime_list_50),
            "success ratio 20mm": np.mean(success_list_20),
            "Average reach time 20mm": np.mean(reachtime_list_20),
            "success ratio 10mm": np.mean(success_list_10),
            "Average reach time 10mm": np.mean(reachtime_list_10),
            "success ratio 5mm": np.mean(success_list_5),
            "Average reach time 5mm": np.mean(reachtime_list_5),
            "success ratio 2mm": np.mean(success_list_2),
            "Average reach time 2mm": np.mean(reachtime_list_2),
            "success ratio 1mm": np.mean(success_list_1),
            "Average reach time 1mm": np.mean(reachtime_list_1),
            "success ratio 0.5mm": np.mean(success_list_05),
            "Average reach time 0.5mm": np.mean(reachtime_list_05)
        }
        df = pd.DataFrame(d, index=[0])

        if args.random_pol:
            df.to_csv("logs/random_policy_0.2M/"+env_id+"/stats.csv",
                      index=False)  # make path naming more robust
        else:
            df.to_csv(log_path+"/stats.csv", index=False)

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(
            "Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()


if __name__ == '__main__':
    main()
