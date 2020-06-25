import pandas as pd
from pathlib import Path
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
    args = parser.parse_args()


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
