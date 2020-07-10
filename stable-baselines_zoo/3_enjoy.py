import os
import sys
import argparse
import pkg_resources
import importlib
import warnings
import pandas as pd
import time

# added by Pierre
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# from rlkit.envs.wrappers import NormalizedBoxEnv
import gym, widowx_env
import utils.import_envs  # pytype: disable=import-error
import numpy as np
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model

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
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--render-pybullet', help='Slow down Pybullet simulation to render', default=False) # added by Pierre
    parser.add_argument('--random-pol', help='Random policy', default=False) # added by Pierre
    args = parser.parse_args()

    plot_bool = True
    plot_dim = 2
    log_bool = False

    if plot_bool:

        if plot_dim == 2:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(5, 10))
        elif plot_dim == 3:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

    if log_bool:
        output_df = pd.DataFrame()

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
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

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
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    episode = 0

    # success_threshold_001 = 0.01
    # success_list_001, reachtime_list_001, episode_success_list_001 = [], [], []
    # success_threshold_0002 = 0.002
    # success_list_0002, reachtime_list_0002, episode_success_list_0002 = [], [], []
    # success_threshold_0001 = 0.001
    # success_list_0001, reachtime_list_0001, episode_success_list_0001 = [], [], []
    # success_threshold_00005 = 0.0005
    # success_list_00005, reachtime_list_00005, episode_success_list_00005 = [], [], []

    # changed for the paper
    success_threshold_50 = 0.05
    success_list_50, reachtime_list_50, episode_success_list_50 = [], [], []
    success_threshold_20 = 0.02
    success_list_20, reachtime_list_20, episode_success_list_20 = [], [], []
    success_threshold_10 = 0.01
    success_list_10, reachtime_list_10, episode_success_list_10 = [], [], []
    success_threshold_5 = 0.005
    success_list_5, reachtime_list_5, episode_success_list_5 = [], [], []


    # For HER, monitor success rate
    successes = []
    state = None
    
    for _ in range(args.n_timesteps):
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
        
        if infos[0]['total_distance'] <= success_threshold_50:
            episode_success_list_50.append(1)
        else:
            episode_success_list_50.append(0)

        if infos[0]['total_distance'] <= success_threshold_20:
            episode_success_list_20.append(1)
        else:
            episode_success_list_20.append(0)

        if infos[0]['total_distance'] <= success_threshold_10:
            episode_success_list_10.append(1)
        else:
            episode_success_list_10.append(0)

        if infos[0]['total_distance'] <= success_threshold_5:
            episode_success_list_5.append(1)
        else:
            episode_success_list_5.append(0)
        

        if plot_bool:
            goal = infos[0]['goal position']
            tip = infos[0]['tip position']

            if plot_dim == 2:
                ax1.cla()
                ax1.plot(goal[0], goal[2], marker='x', color='b', linestyle='', markersize=10, label="goal", mew=3)
                ax1.plot(tip[0], tip[2], marker='o', color='r', linestyle='', markersize=10, label="end effector")

                circ_1_50 = plt.Circle((goal[0], goal[2]), radius=success_threshold_50, edgecolor='g', facecolor='w', linestyle='--', label="50 mm")
                circ_1_20 = plt.Circle((goal[0], goal[2]), radius=success_threshold_20, edgecolor='b', facecolor='w', linestyle='--', label="20 mm")
                circ_1_10 = plt.Circle((goal[0], goal[2]), radius=success_threshold_10, edgecolor='m', facecolor='w', linestyle='--', label="10 mm")
                circ_1_5 = plt.Circle((goal[0], goal[2]), radius=success_threshold_5, edgecolor='r', facecolor='w', linestyle='--', label="5 mm")
                ax1.add_patch(circ_1_50)
                ax1.add_patch(circ_1_20)
                ax1.add_patch(circ_1_10)
                ax1.add_patch(circ_1_5)

                ax1.set_xlim([-0.25, 0.25])
                ax1.set_ylim([0, 0.5])
                ax1.set_xlabel("x (m)", fontsize=15)
                ax1.set_ylabel("z (m)", fontsize=15)

                ax2.cla()
                ax2.plot(goal[1], goal[2], marker='x', color='b', linestyle='', markersize=10, mew=3)
                ax2.plot(tip[1], tip[2], marker='o', color='r', linestyle='', markersize=10)

                circ_2_50 = plt.Circle((goal[1], goal[2]), radius=success_threshold_50, edgecolor='g', facecolor='w', linestyle='--')
                circ_2_20 = plt.Circle((goal[1], goal[2]), radius=success_threshold_20, edgecolor='b', facecolor='w', linestyle='--')
                circ_2_10 = plt.Circle((goal[1], goal[2]), radius=success_threshold_10, edgecolor='m', facecolor='w', linestyle='--')
                circ_2_5 = plt.Circle((goal[1], goal[2]), radius=success_threshold_5, edgecolor='r', facecolor='w', linestyle='--')
                ax2.add_patch(circ_2_50)
                ax2.add_patch(circ_2_20)
                ax2.add_patch(circ_2_10)
                ax2.add_patch(circ_2_5)

                ax2.set_xlim([-0.25, 0.25])
                ax2.set_ylim([0, 0.5])
                ax2.set_xlabel("y (m)", fontsize=15)
                ax2.set_ylabel("z (m)", fontsize=15)

                ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.2), ncol=3, fancybox=True, shadow=True)

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

            fig.suptitle("timestep "+str(ep_len)+" | distance to target: "+str(round(infos[0]['total_distance']*1000, 1))+" mm")
            plt.pause(0.01)
            # plt.show()

        if log_bool:
            dict_log = infos[0]
            dict_log['action'] = action[0]
            dict_log['obs'] = obs[0]
            dict_log['reward'] = reward[0]
            dict_log['done'] = done[0]
            dict_log['timestep'] = ep_len
            dict_log['episode'] = episode
            output_df = output_df.append(dict_log, ignore_index=True)


        # if not args.no_render:
        #     env.render('human')

        episode_reward += reward[0]
        ep_len += 1

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
                print("Episode nb: {} | Episode Reward: {:.2f} | Episode Length: {}".format(episode, episode_reward, ep_len))
                # print("Episode Length", ep_len) # commented by Pierre
                state = None
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)

                # append the last element of the episode success list when episode is done
                success_list_50.append(episode_success_list_50[-1]) 
                success_list_20.append(episode_success_list_20[-1]) 
                success_list_10.append(episode_success_list_10[-1]) 
                success_list_5.append(episode_success_list_5[-1])  

                # if the episode is successful and it starts from an unsucessful step, calculate reach time
                if episode_success_list_50[-1] == True and episode_success_list_50[0] == False:
                    idx = 0
                    while episode_success_list_50[idx] == False:
                        idx += 1
                    reachtime_list_50.append(idx)

                if episode_success_list_20[-1] == True and episode_success_list_20[0] == False:
                    idx = 0
                    while episode_success_list_20[idx] == False:
                        idx += 1
                    reachtime_list_20.append(idx)

                if episode_success_list_10[-1] == True and episode_success_list_10[0] == False:
                    idx = 0
                    while episode_success_list_10[idx] == False:
                        idx += 1
                    reachtime_list_10.append(idx)

                if episode_success_list_5[-1] == True and episode_success_list_5[0] == False:
                    idx = 0
                    while episode_success_list_5[idx] == False:
                        idx += 1
                    reachtime_list_5.append(idx)


                if log_bool:
                    # output_df.to_csv(log_path+"/res_episode_"+str(episode)+".csv", index=False)  # slow
                    output_df.to_pickle(log_path+"/res_episode_"+str(episode)+".pkl")

                # reset for new episode
                episode_reward = 0.0
                ep_len = 0
                episode_success_list_50 = []  
                episode_success_list_20 = []  
                episode_success_list_10 = []  
                episode_success_list_5 = []  
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
        print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(success_threshold_50, np.mean(success_list_50), np.mean(reachtime_list_50)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(success_threshold_20, np.mean(success_list_20), np.mean(reachtime_list_20)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(success_threshold_10, np.mean(success_list_10), np.mean(reachtime_list_10)))
        print("success threshold: {} | success ratio: {:.2f} | Average reach time: {:.2f}".format(success_threshold_5, np.mean(success_list_5), np.mean(reachtime_list_5)))

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
            }
        df = pd.DataFrame(d, index=[0])

        if args.random_pol:
            df.to_csv("logs/random_policy_0.2M/"+env_id+"/stats.csv", index=False)  # make path naming more robust
        else:
            df.to_csv(log_path+"/stats.csv", index=False)


    if args.verbose > 0 and len(episode_lengths) > 0:
        print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

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
