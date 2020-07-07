import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from sklearn import preprocessing



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--logFolder', help='Log folder', type=str)
    parser.add_argument('-s', '--saveFolder', help='save folder', type=str)
    parser.add_argument('-e', '--envPaper', help='envPaper', type=str)
    parser.add_argument('-r', '--randomLogFolder', help='random log folder', type=str)
    args = parser.parse_args()


    path_base = args.logFolder  # "logs/train_1M_widowx_reach-v3/"
    save_dir = args.saveFolder   #"experiment_reports/1M_widowx_reach-v3/"
    random_dir = args.randomLogFolder   #"logs/random_policy/widowx_reacher-v5/""
    appendix = args.envPaper 
    os.makedirs(save_dir, exist_ok=True)

    ### GET DATA ###


    df1 = pd.read_csv(path_base+"a2c/all_rewards_smooth.csv")
    df2 = pd.read_csv(path_base+"acktr/all_rewards_smooth.csv")
    df3 = pd.read_csv(path_base+"ddpg/all_rewards_smooth.csv")
    df4 = pd.read_csv(path_base+"ppo2/all_rewards_smooth.csv")
    df5 = pd.read_csv(path_base+"sac/all_rewards_smooth.csv")
    df6 = pd.read_csv(path_base+"td3/all_rewards_smooth.csv")
    df7 = pd.read_csv(path_base+"trpo/all_rewards_smooth.csv")
    df8 = pd.read_csv(path_base+"her_sac/all_rewards_smooth.csv")
    df9 = pd.read_csv(path_base+"her_td3/all_rewards_smooth.csv")

    df_list = [
        df1, 
        df2, 
        df3, 
        df4,
        df5, 
        df6, 
        df7,
        df8,
        df9
    ]

    df_label = [
        "A2C",
        "ACKTR",
        "DDPG",
        "PPO2",
        "SAC",
        "TD3",
        "TRPO",
        "SAC + HER",
        "TD3 + HER"
    ]

    ff1 = pd.read_csv(path_base+"/a2c/results_seed_exp.csv")
    ff2 = pd.read_csv(path_base+"/acktr/results_seed_exp.csv")
    ff3 = pd.read_csv(path_base+"/ddpg/results_seed_exp.csv")
    ff4 = pd.read_csv(path_base+"/ppo2/results_seed_exp.csv")
    ff5 = pd.read_csv(path_base+"/sac/results_seed_exp.csv")
    ff6 = pd.read_csv(path_base+"/td3/results_seed_exp.csv")
    ff7 = pd.read_csv(path_base+"/trpo/results_seed_exp.csv")
    ff8 = pd.read_csv(path_base+"/her_sac/results_seed_exp.csv")
    ff9 = pd.read_csv(path_base+"/her_td3/results_seed_exp.csv")


    ff_list = [
        ff1,
        ff2,
        ff3,
        ff4,
        ff5,
        ff6,
        ff7,
        ff8,
        ff9
    ]


    ff = pd.concat(ff_list, axis=0)
    ff['exp type'] = df_label



    ### PLOT LEARNING CURVES ###

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure(1, figsize=(8, 6))
    ax1 = plt.axes()

    for (df, lab) in zip(df_list, df_label):
        df.plot(x='timesteps', y='mean_reward', ax=ax1, label=lab)


    random_df = pd.read_csv(random_dir+"all_rewards_smooth.csv")
    random_df.plot(x='timesteps', y='mean_reward', ax=ax1, label="random")    ## add random

    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 5))
    plt.ylabel(r'Average return $R_t$ (m\textsuperscript{2})', fontsize=15)
    plt.xlabel(r'Timesteps $t$', fontsize=15)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=5, fancybox=True, shadow=True)

    plt.savefig(save_dir+"learning_curves"+appendix+".pdf", bbox_inches='tight', dpi=1000)



    def plot_shaded(df, ax, lab):
        ax.plot(df['timesteps'], df['mean_reward'], label=lab)
        ax.fill_between(df['timesteps'], df['lower'], df['upper'], alpha=0.35)



    for (df, lab) in zip(df_list, df_label):
        plt.figure()
        ax = plt.axes()
        plot_shaded(df, ax, lab)

        plt.legend(loc="lower right")
        plt.ylabel(r'Average return $R_t$', fontsize=15)
        plt.xlabel(r'Timesteps $t$', fontsize=15)
        plt.savefig(save_dir+lab+appendix+".pdf", bbox_inches='tight', dpi=500)

    

    ### PLOT TRAINING STATS ###


    # min max scaled
    # ff['mean reward scaled'] = (ff['mean reward']-ff['mean reward'].min())/(ff['mean reward'].max()-ff['mean reward'].min())
    # ff['mean train walltime scaled'] = (ff['mean train walltime (s)']-ff['mean train walltime (s)'].min())/(ff['mean train walltime (s)'].max()-ff['mean train walltime (s)'].min())

    # standard scaler
    # ff['mean reward scaled'] = (ff['mean reward']-ff['mean reward'].mean())/ff['mean reward'].std()
    # ff['mean train walltime scaled'] = (ff['mean train walltime (s)']-ff['mean train walltime (s)'].mean())/ff['mean train walltime (s)'].std()

    ff['efficiency (reward / s)'] = ff['mean reward'] / ff['mean train walltime (s)']
    # ff['efficiency scaled'] = 1 / (ff['mean reward scaled'] * ff['mean train walltime scaled'])
    # ff['efficiency scaled'] = (ff['efficiency (reward / s)']-ff['efficiency (reward / s)'].min())/(ff['efficiency (reward / s)'].max()-ff['efficiency (reward / s)'].min())


    print(ff)

    # round to 2 decimal and convert to min
    ff_round = ff
    ff_round['mean train walltime (min)'] = ff_round['mean train walltime (s)'] / 60
    ff_round['std train walltime (min)'] = ff_round['std train walltime (s)'] / 60
    ff_round = ff_round.round(2)
    ff_round.to_csv(save_dir+"res.csv", index=False)



    def plot_col(y_col, yerr_col, title, ylab):

        ax = ff.plot.bar(x='exp type', y=y_col, yerr=yerr_col, rot=45)
        ax.set_xticklabels(df_label, ha='right')
        plt.ylabel(ylab)
        plt.tight_layout()
        plt.savefig(save_dir+title+appendix+".pdf", dpi=100)
        # plt.show()

    # plot_col('mean success ratio 10mm', 'std success ratio 10mm', "success_10mm.pdf")
    # plot_col('mean reach time 10mm', 'std reach time 10mm', "reachtime_10mm.pdf")
    # plot_col('mean success ratio 2mm', 'std success ratio 2mm', "success_2mm.pdf")
    # plot_col('mean reach time 2mm', 'std reach time 2mm', "reachtime_2mm.pdf")
    # plot_col('mean success ratio 1mm', 'std success ratio 1mm', "success_1mm.pdf")
    # plot_col('mean reach time 1mm', 'std reach time 1mm', "reachtime_1mm.pdf")
    # plot_col('mean success ratio 0.5mm', 'std success ratio 0.5mm', "success_0.5mm.pdf")
    # plot_col('mean reach time 0.5mm', 'std reach time 0.5mm', "reachtime_0.5mm.pdf")

    # changed for paper
    plot_col('mean success ratio 50mm', 'std success ratio 50mm', "success_50mm", 'mean success ratio')
    plot_col('mean reach time 50mm', 'std reach time 50mm', "reachtime_50mm", 'mean reach time')
    plot_col('mean success ratio 20mm', 'std success ratio 20mm', "success_20mm", 'mean success ratio')
    plot_col('mean reach time 20mm', 'std reach time 20mm', "reachtime_20mm", 'mean reach time')
    plot_col('mean success ratio 10mm', 'std success ratio 10mm', "success_10mm", 'mean success ratio')
    plot_col('mean reach time 10mm', 'std reach time 10mm', "reachtime_10mm", 'mean reach time')
    plot_col('mean success ratio 5mm', 'std success ratio 5mm', "success_5mm", 'mean success ratio')
    plot_col('mean reach time 5mm', 'std reach time 5mm', "reachtime_5mm", 'mean reach time')


    plot_col('mean reward', 'std reward (seed)', "mean_reward", 'mean return')
    plot_col('mean train walltime (min)', 'std train walltime (min)', "mean_walltime", 'mean train time (min)')
    plot_col('efficiency (reward / s)', None, "efficiency", 'mean efficiency')


