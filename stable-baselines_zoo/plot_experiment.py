import pandas as pd
from pathlib import Path
import yaml
# import matplotlib.pyplot as plt

# added by Pierre
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os
import argparse

from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy


def moving_average(values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')


def plot_results(log_folder, type_str, leg_label):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param type: (str) either 'timesteps', 'episodes' or 'walltime_hrs'
    """

    x, y = ts2xy(load_results(log_folder), type_str)

    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    # plt.figure()
    # plt.plot(x, y, label=leg_label)
    # plt.xlabel(type_str)
    plt.ylabel('Rewards')
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('-e', '--env', help='env name', type=str)
    parser.add_argument('-ns', '--nb-seeds', help='number of seeds', type=int)
    parser.add_argument('-n', help='number of eval steps', type=int)
    args = parser.parse_args()

    nb_eval_timesteps = args.n
    nb_seeds = args.nb_seeds
    env_id = args.env
    log_dir = args.folder
    print(log_dir)


    ###############
    # METRICS 
    ###############

    # Get the mean of the reward and wall train time of all the seed runs in the experiment

    res_file_list = []

    for path in Path(log_dir).rglob('stats.csv'):
        # print(path)
        res_file_list.append(path)

    res_file_list = sorted(res_file_list)
    # print(res_file_list)

    li = []
    count = 0

    for filename in res_file_list:
        df = pd.read_csv(filename, index_col=None, header=0)
        df['seed'] = count
        df['log_dir'] = filename
        li.append(df)
        count += 1

    # print(li)

    df = pd.concat(li, axis=0, ignore_index=True)

    print(df)

    # print(df['Eval mean reward'].mean())
    # print(df['Eval mean reward'].std())
    # print(df['Eval std'].mean())
    # print(df['Train walltime (s)'].mean())
    # print(df['Train walltime (s)'].std())

    d = {
        'mean reward': df['Eval mean reward'].mean(),
        'std reward (seed)': df['Eval mean reward'].std(),
        'std reward (eval)': df['Eval std'].mean(),
        'mean train walltime (s)': df['Train walltime (s)'].mean(),
        'std train walltime (s)': df['Train walltime (s)'].std(),
        'mean success ratio 50mm': df['success ratio 50mm'].mean(),
        'std success ratio 50mm': df['success ratio 50mm'].std(),
        'mean reach time 50mm': df['Average reach time 50mm'].mean(),
        'std reach time 50mm': df['Average reach time 50mm'].std(),
        'mean success ratio 20mm': df['success ratio 20mm'].mean(),
        'std success ratio 20mm': df['success ratio 20mm'].std(),
        'mean reach time 20mm': df['Average reach time 20mm'].mean(),
        'std reach time 20mm': df['Average reach time 20mm'].std(),
        'mean success ratio 10mm': df['success ratio 10mm'].mean(),
        'std success ratio 10mm': df['success ratio 10mm'].std(),
        'mean reach time 10mm': df['Average reach time 10mm'].mean(),
        'std reach time 10mm': df['Average reach time 10mm'].std(),
        'mean success ratio 5mm': df['success ratio 5mm'].mean(),
        'std success ratio 5mm': df['success ratio 5mm'].std(),
        'mean reach time 5mm': df['Average reach time 5mm'].mean(),
        'std reach time 5mm': df['Average reach time 5mm'].std(),
    }

    df_res = pd.DataFrame(d, index=[0])
    df_res.to_csv(log_dir+"results_seed_exp.csv", index=False)


    ############### Prepare dataframe for compiling benchmark results

    if env_id == "widowx_reacher-v5":
        nb_joints = 6
        action = "[Rel_A1, Rel_A2, Rel_A3, Rel_A4, Rel_A5, Rel_A6]"
        obs = "[target_x, target_y, target_z, A1, A2, A3, A4, A5, A6]"
        reward = "[-distance_to_target**2]"
        random_goal = 0
        goal_env = 0
        ep_len = 100

    elif env_id == "widowx_reacher-v6":
        nb_joints = 6
        action = "[Rel_A1, Rel_A2, Rel_A3, Rel_A4, Rel_A5, Rel_A6]"
        obs = "[target_x, target_y, target_z, A1, A2, A3, A4, A5, A6]"
        reward = "[-distance_to_target**2]"
        random_goal = 0
        goal_env = 1
        ep_len = 100

    # I need to add forward slashes because reacher contains "her"...
    if "/a2c/" in log_dir:
        algo = "a2c"
    elif "/acktr/" in log_dir:
        algo = "acktr"
    elif "/ddpg/" in log_dir:
        algo = "ddpg"
    elif "/ppo2/" in log_dir:
        algo = "ppo2"
    elif "/sac/" in log_dir:
        algo = "sac"
    elif "/td3/" in log_dir:
        algo = "td3"
    elif "/trpo/" in log_dir:
        algo = "trpo"
    elif "/her_sac/" in log_dir:
        algo = "her_sac"
    elif "/her_td3/" in log_dir:
        algo = "her_td3"

    # # Load hyperparameters from yaml file (this is not robuts, as the hyperparams/{}.yml may have been edited manually after the training)
    # with open('hyperparams/{}.yml'.format(algo), 'r') as f:
    #     hyperparams_dict = yaml.safe_load(f)
    #     if env_id in list(hyperparams_dict.keys()):
    #         hyperparams = hyperparams_dict[env_id]
    #     else:
    #         raise ValueError("Hyperparameters not found for {}-{}".format(algo, env_id))
    

    # # LOAD HYPERPARAMS FROM config.yml (it is the same for all the seeds so selecting any config.yml is fine)
    # I specified the defaults hyperparameters in the hyperparameters/{algo}.yml
    # So the config.yml should contain all the hyperparameters used during training
    # Without omiting the default ones (so I don't need to load from tuned_hyperparams.yml)
    for config_path in Path(log_dir).rglob('config.yml'):
        print(config_path)

    # # load hyperparams
    with open(config_path, 'r') as f:
        hyperparams_ordered = yaml.load(f, Loader=yaml.UnsafeLoader)
        hyperparams = dict(hyperparams_ordered)

    benchmark_dict = {
        'env_id': env_id,
        'nb_joints': nb_joints,
        'action': action,
        'obs': obs,
        'reward': reward,
        'random_goal': random_goal,
        'goal_env': goal_env,
        'algo': algo,
        'nb_seeds': nb_seeds,
        'mean_train_time(s)': df['Train walltime (s)'].mean(),
        'std_train_time(s)': df['Train walltime (s)'].std(),
        'nb_eval_timesteps': nb_eval_timesteps,
        'mean_return': df['Eval mean reward'].mean(),
        'std_return': df['Eval mean reward'].std(),
        'mean_SR_50': df['success ratio 50mm'].mean(),
        'std_SR_50': df['success ratio 50mm'].std(),
        'mean_RT_50': df['Average reach time 50mm'].mean(),
        'std_RT_50': df['Average reach time 50mm'].std(),
        'mean_SR_20': df['success ratio 20mm'].mean(),
        'std_SR_20': df['success ratio 20mm'].std(),
        'mean_RT_20': df['Average reach time 20mm'].mean(),
        'std_RT_20': df['Average reach time 20mm'].std(),
        'mean_SR_10': df['success ratio 10mm'].mean(),
        'std_SR_10': df['success ratio 10mm'].std(),
        'mean_RT_10': df['Average reach time 10mm'].mean(),
        'std_RT_10': df['Average reach time 10mm'].std(),
        'mean_SR_0': df['success ratio 5mm'].mean(),
        'std_SR_5': df['success ratio 5mm'].std(),
        'mean_RT_5': df['Average reach time 5mm'].mean(),
        'std_RT_5': df['Average reach time 5mm'].std(),
    }

    # append hyperparameters
    benchmark_dict = {**benchmark_dict, **hyperparams}

    # transform into a dataframe
    df_bench = pd.DataFrame(benchmark_dict, index=[0])

    # add to existing results and save
    backedup_df = pd.read_csv("results/benchmark_results.csv")
    appended_df = backedup_df.append(df_bench, ignore_index=True)
    appended_df.to_csv("results/benchmark_results.csv", index=False)

    ###############
    # LEARNING CURVES
    ###############

    # Plot the learning curve of all the seed runs in the experiment

    res_file_list = []

    for path in Path(log_dir).rglob(env_id+'_*'):
        res_file_list.append(path)

    res_file_list = sorted(res_file_list)
    # print(res_file_list)

    df_list = []
    col_list = []
    count = 1

    for filename in res_file_list:
        # print(filename)
        filename = str(filename) # convert from Posixpath to string
        
        W = load_results(filename)
        print(W['r'])

        df_list.append(W['r'])
        col_list.append("seed "+str(count))
        count += 1

    #     plot_results(filename, 'timesteps', "seed nb "+str(count))
    # #     plot_results(filename, 'episodes')
    # #     plot_results(filename, 'walltime_hrs')


    all_rewards = pd.concat(df_list, axis=1)
    all_rewards.columns = col_list
    
    all_rewards_copy = all_rewards.copy()
    all_rewards["mean_reward"] = all_rewards_copy.mean(axis=1)
    all_rewards["std_reward"] = all_rewards_copy.std(axis=1)
    all_rewards["upper"] = all_rewards["mean_reward"] + all_rewards["std_reward"]
    all_rewards["lower"] = all_rewards["mean_reward"] - all_rewards["std_reward"]
    all_rewards['timesteps'] = W['l'].cumsum()
    all_rewards.to_csv(log_dir+"all_rewards.csv", index=False)

    # plot
    plt.figure(1, figsize=(10, 5))
    ax = plt.axes()

    for seed_col in col_list:
        print(seed_col)
        all_rewards.plot(x='timesteps', y=seed_col, ax=ax)

    all_rewards.plot(x='timesteps', y='mean_reward', ax=ax, color='k')

    plt.xlabel('Time steps')
    plt.ylabel('Rewards')

    plt.legend()
    plt.savefig(log_dir+"reward_vs_timesteps.png", dpi=100)
    # plt.show()




    # apply rolling window (except on timesteps)
    for col in all_rewards.columns[:-1]:
        print(col)
        all_rewards[col] = all_rewards[col].rolling(window=50).mean()

    all_rewards.dropna(inplace=True)  # remove NaN due to rolling average
    all_rewards.to_csv(log_dir+"all_rewards_smooth.csv", index=False)
    print(all_rewards)

    # plot
    plt.figure(2, figsize=(10, 5))
    ax = plt.axes()

    for seed_col in col_list:
        print(seed_col)
        all_rewards.plot(x='timesteps', y=seed_col, ax=ax)

    all_rewards.plot(x='timesteps', y='mean_reward', ax=ax, color='k')

    plt.xlabel('Time steps')
    plt.ylabel('Rewards')

    plt.legend()
    plt.savefig(log_dir+"reward_vs_timesteps_smoothed.png", dpi=100)
    # plt.show()
