import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def plot_shaded(df, ax, lab):
    ax.plot(df['timesteps'], df['mean_reward'], label=lab)
    ax.fill_between(df['timesteps'], df['lower'], df['upper'], alpha=0.35)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Log folder random', type=str)
    args = parser.parse_args()

    log_path = args.folder
    print(log_path)
    # log_path = args.folder  #"logs/random_policy_0.2M/widowx_reacher-v5/"
    df = pd.read_csv(log_path+"all_rewards.csv")

    # remove timestep column
    df_reward_only = df[df.columns[:-1]]

    # calc mean, std and shading limits
    df_reward_only["mean_reward"] = df_reward_only.mean(axis=1)
    df_reward_only["std_reward"] = df_reward_only.std(axis=1)
    df_reward_only["upper"] = df_reward_only["mean_reward"] + df_reward_only["std_reward"]
    df_reward_only["lower"] = df_reward_only["mean_reward"] - df_reward_only["std_reward"]

    # apply rolling window (except on timesteps)
    for col in df_reward_only.columns:
        df_reward_only[col] = df_reward_only[col].rolling(window=50).mean()

    # add timesteps again
    df_reward_only['timesteps'] = df['timesteps']

    # remove NaN due to rolling average
    df_reward_only.dropna(inplace=True) 

    df_reward_only.to_csv(log_path+"/all_rewards_smooth.csv", index=False)

    # plot
    plt.figure(1, figsize=(10, 5))
    ax = plt.axes()
    plot_shaded(df_reward_only, ax, "random policy")

    plt.legend(loc="lower right")
    plt.ylabel("Mean reward")
    plt.xlabel("Timesteps")
    plt.savefig(log_path+"training_curve.pdf", dpi=100)
    # plt.show()

  