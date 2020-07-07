import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from sklearn import preprocessing
from matplotlib.ticker import EngFormatter


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', '--logFolder1', help='Log folder 1', type=str)
    parser.add_argument('-f2', '--logFolder2', help='Log folder 2', type=str)
    parser.add_argument('-s', '--saveFolder', help='save folder', type=str)
    args = parser.parse_args()


    path_base = args.logFolder1  # "logs/train_1M_widowx_reach-v3/"
    path_base2 = args.logFolder2  # "logs/train_1M_widowx_reach-v3/"
    save_dir = args.saveFolder   #"experiment_reports/1M_widowx_reach-v3/"
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

    env2_df1 = pd.read_csv(path_base2+"a2c/all_rewards_smooth.csv")
    env2_df2 = pd.read_csv(path_base2+"acktr/all_rewards_smooth.csv")
    env2_df3 = pd.read_csv(path_base2+"ddpg/all_rewards_smooth.csv")
    env2_df4 = pd.read_csv(path_base2+"ppo2/all_rewards_smooth.csv")
    env2_df5 = pd.read_csv(path_base2+"sac/all_rewards_smooth.csv")
    env2_df6 = pd.read_csv(path_base2+"td3/all_rewards_smooth.csv")
    env2_df7 = pd.read_csv(path_base2+"trpo/all_rewards_smooth.csv")
    env2_df8 = pd.read_csv(path_base2+"her_sac/all_rewards_smooth.csv")
    env2_df9 = pd.read_csv(path_base2+"her_td3/all_rewards_smooth.csv")

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

    df_list2 = [
        env2_df1, 
        env2_df2, 
        env2_df3, 
        env2_df4,
        env2_df5, 
        env2_df6, 
        env2_df7,
        env2_df8,
        env2_df9
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

    env2_ff1 = pd.read_csv(path_base2+"/a2c/results_seed_exp.csv")
    env2_ff2 = pd.read_csv(path_base2+"/acktr/results_seed_exp.csv")
    env2_ff3 = pd.read_csv(path_base2+"/ddpg/results_seed_exp.csv")
    env2_ff4 = pd.read_csv(path_base2+"/ppo2/results_seed_exp.csv")
    env2_ff5 = pd.read_csv(path_base2+"/sac/results_seed_exp.csv")
    env2_ff6 = pd.read_csv(path_base2+"/td3/results_seed_exp.csv")
    env2_ff7 = pd.read_csv(path_base2+"/trpo/results_seed_exp.csv")
    env2_ff8 = pd.read_csv(path_base2+"/her_sac/results_seed_exp.csv")
    env2_ff9 = pd.read_csv(path_base2+"/her_td3/results_seed_exp.csv")


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


    ff_list2 = [
        env2_ff1,
        env2_ff2,
        env2_ff3,
        env2_ff4,
        env2_ff5,
        env2_ff6,
        env2_ff7,
        env2_ff8,
        env2_ff9
    ]


    ff = pd.concat(ff_list, axis=0)
    ff['exp type'] = df_label

    ff2 = pd.concat(ff_list2, axis=0)
    ff2['exp type'] = df_label

    

    ### PLOT LEARNING CURVES ###

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 


    def plot_shaded(df, ax, lab):
        # df['timesteps'] /= 1000

        ax.plot(df['timesteps'], df['mean_reward'], label=lab)
        ax.fill_between(df['timesteps'], df['lower'], df['upper'], alpha=0.35)



    for (df,  df2, lab) in zip(df_list, df_list2, df_label):
        plt.figure(figsize=(8, 6))
        ax = plt.axes()
        plot_shaded(df, ax, "Env1")
        plot_shaded(df2, ax, "Env2")

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 5))

        plt.legend(loc="lower right", fontsize=25)
        plt.ylabel(r'Average return $R_t$ (m\textsuperscript{2})', fontsize=25)
        plt.xlabel(r'Timesteps $t$', fontsize=25)
        plt.savefig(save_dir+lab+".pdf", bbox_inches='tight', dpi=1000)


    ### PLOT TRAINING STATS ###
    
    
    ff['mean train walltime (min)'] = ff['mean train walltime (s)'] / 60
    ff['std train walltime (min)'] = ff['std train walltime (s)'] / 60
    ff2['mean train walltime (min)'] = ff2['mean train walltime (s)'] / 60
    ff2['std train walltime (min)'] = ff2['std train walltime (s)'] / 60

    ff['efficiency (reward / min)'] = ff['mean reward'] / ff['mean train walltime (min)']
    ff2['efficiency (reward / min)'] = ff2['mean reward'] / ff2['mean train walltime (min)']

    # set column algo as index for bar plotting
    ff.set_index('exp type', inplace=True)
    ff2.set_index('exp type', inplace=True)
    
    print(ff)
    print(ff2)

    plt.clf()
    plt.cla()

    plt.figure()
    reward_df = pd.concat([ff['mean reward'], ff2['mean reward'], ff['std reward (seed)'], ff2['std reward (seed)']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean reward")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"reward_comp.pdf", dpi=100)


    plt.figure()
    reward_df = pd.concat([ff['mean train walltime (min)'], ff2['mean train walltime (min)'], ff['std train walltime (min)'], ff2['std train walltime (min)']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean train walltime (min)")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"walltime_comp.pdf", dpi=100)


    plt.figure()
    reward_df = pd.concat([ff['efficiency (reward / min)'], ff2['efficiency (reward / min)']], axis=1)
    reward_df.columns = ['mean1', 'mean2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', alpha = 0.7, rot=45)
    plt.ylabel("Efficiency (reward / min)")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"efficiency_comp.pdf", dpi=100)

    plt.clf()
    plt.cla()


    ###
    plt.figure()
    reward_df = pd.concat([ff['mean success ratio 50mm'], ff2['mean success ratio 50mm'], ff['std success ratio 50mm'], ff2['std success ratio 50mm']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean success ratio 50mm")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"success_50_comp.pdf", dpi=100)

    plt.clf()
    plt.cla()

    # plt.figure()
    reward_df = pd.concat([ff['mean reach time 50mm'], ff2['mean reach time 50mm'], ff['std reach time 50mm'], ff2['std reach time 50mm']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean reach time 50mm")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"reachtime_50_comp.pdf", dpi=100)


    plt.clf()
    plt.cla()

    ###

    # plt.figure()
    reward_df = pd.concat([ff['mean success ratio 20mm'], ff2['mean success ratio 20mm'], ff['std success ratio 20mm'], ff2['std success ratio 20mm']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean success ratio 20mm")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"success_20_comp.pdf", dpi=100)

    plt.clf()
    plt.cla()

    # plt.figure()
    reward_df = pd.concat([ff['mean reach time 20mm'], ff2['mean reach time 20mm'], ff['std reach time 20mm'], ff2['std reach time 20mm']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean reach time 20mm")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"reachtime_20_comp.pdf", dpi=100)

    plt.clf()
    plt.cla()

    ###

    # plt.figure()
    reward_df = pd.concat([ff['mean success ratio 10mm'], ff2['mean success ratio 10mm'], ff['std success ratio 10mm'], ff2['std success ratio 10mm']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean success ratio 10mm")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"success_10_comp.pdf", dpi=100)

    plt.clf()
    plt.cla()

    # plt.figure()
    reward_df = pd.concat([ff['mean reach time 10mm'], ff2['mean reach time 10mm'], ff['std reach time 10mm'], ff2['std reach time 10mm']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean reach time 10mm")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"reachtime_10_comp.pdf", dpi=100)

    plt.clf()
    plt.cla()

    ###

    # plt.figure()
    reward_df = pd.concat([ff['mean success ratio 5mm'], ff2['mean success ratio 5mm'], ff['std success ratio 5mm'], ff2['std success ratio 5mm']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean success ratio 5mm")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"success_5_comp.pdf", dpi=100)

    plt.clf()
    plt.cla()


    # plt.figure()
    reward_df = pd.concat([ff['mean reach time 5mm'], ff2['mean reach time 5mm'], ff['std reach time 5mm'], ff2['std reach time 5mm']], axis=1)
    reward_df.columns = ['mean1', 'mean2', 'std1', 'std2']
    reward_df[['mean1', 'mean2']].plot(kind='bar', yerr=reward_df[['std1', 'std2']].values.T, alpha = 0.7, error_kw=dict(ecolor='k', lw=0.8, capsize=3), rot=45)
    plt.ylabel("Mean reach time 5mm")
    plt.xlabel("Algorithm")
    plt.legend(["Env1", "Env2"])
    plt.tight_layout()
    plt.savefig(save_dir+"reachtime_5_comp.pdf", dpi=100)

    plt.clf()
    plt.cla()