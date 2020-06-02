import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import argparse

# added by Pierre
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def plot_results(log_folder, type_str, window_size=50):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param type: (str) either 'timesteps', 'episodes' or 'walltime_hrs'
    """

    x, y = ts2xy(load_results(log_folder), type_str)
    # x, y = ts2xy(load_results(log_folder), 'episodes')
    # x, y = ts2xy(load_results(log_folder), 'walltime_hrs')

    y = moving_average(y, window=window_size)
    # Truncate x
    x = x[len(x) - len(y):]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(type_str)
    plt.ylabel('Rewards')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    args = parser.parse_args()

    log_dir = args.folder

    timesteps = 1e10
    window_size = 50

    W = load_results(log_dir)

    print("results: ", W)

    # save walltime to stats.csv
    df = pd.read_csv(log_dir+'stats.csv')  
    df["Train walltime (s)"] = W["t"].max()
    df.to_csv(log_dir+"stats.csv", index=False)
    print(df)

    # plot all training rewards

    results_plotter.plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "")
    plt.savefig(log_dir+"reward_vs_timesteps.png")
    # plt.show()

    results_plotter.plot_results([log_dir], timesteps, results_plotter.X_EPISODES, "")
    plt.savefig(log_dir+"reward_vs_episodes.png")
    # plt.show()

    results_plotter.plot_results([log_dir], timesteps, results_plotter.X_WALLTIME, "")
    plt.savefig(log_dir+"reward_vs_walltime.png")
    # plt.show()


    #### smoothed training rewards
        
    plot_results(log_dir, 'timesteps', window_size)
    plt.savefig(log_dir+"reward_vs_timesteps_smoothed.png")
    # plt.show()

    plot_results(log_dir, 'episodes', window_size)
    plt.savefig(log_dir+"reward_vs_episodes_smoothed.png")
    # plt.show()

    plot_results(log_dir, 'walltime_hrs',window_size)
    plt.savefig(log_dir+"reward_vs_walltime_smoothed.png")
    # plt.show()
