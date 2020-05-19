import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from sklearn import preprocessing


### GET DATA ###

path_base = "logs/train_0.01M_widowx_reach-v3/"

save_dir = "experiment_reports/0.01M_widowx_reach-v3/"
os.makedirs(save_dir, exist_ok=True)

df1 = pd.read_csv(path_base+"a2c/all_rewards.csv")
df2 = pd.read_csv(path_base+"acktr/all_rewards.csv")
df3 = pd.read_csv(path_base+"ddpg/all_rewards.csv")
df4 = pd.read_csv(path_base+"ppo2/all_rewards.csv")
df5 = pd.read_csv(path_base+"sac/all_rewards.csv")
df6 = pd.read_csv(path_base+"td3/all_rewards.csv")
df7 = pd.read_csv(path_base+"trpo/all_rewards.csv")

df_list = [
    df1, 
    df2, 
    df3, 
    df4,
    df5, 
    df6, 
    df7
]

df_label = [
    "A2C",
    "ACKTR",
    "DDPG",
    "PPO2",
    "SAC",
    "TD3",
    "TRPO"
]

ff1 = pd.read_csv(path_base+"/a2c/results_seed_exp.csv")
ff2 = pd.read_csv(path_base+"/acktr/results_seed_exp.csv")
ff3 = pd.read_csv(path_base+"/ddpg/results_seed_exp.csv")
ff4 = pd.read_csv(path_base+"/ppo2/results_seed_exp.csv")
ff5 = pd.read_csv(path_base+"/sac/results_seed_exp.csv")
ff6 = pd.read_csv(path_base+"/td3/results_seed_exp.csv")
ff7 = pd.read_csv(path_base+"/trpo/results_seed_exp.csv")


ff_list = [
    ff1,
    ff2,
    ff3,
    ff4,
    ff5,
    ff6,
    ff7
 ]


ff = pd.concat(ff_list, axis=0)
ff['exp type'] = df_label





### PLOT LEARNING CURVES ###


# apply curve smoothing by moving average

def smooth_reward(df):
    df['mean_reward'] = df['mean_reward'].rolling(window=50).mean()


def smooth_upper_lower(df):
    df['upper'] = df['upper'].rolling(window=50).mean()
    df['lower'] = df['lower'].rolling(window=50).mean()


for df in df_list:
    smooth_reward(df)
    smooth_upper_lower(df)



plt.figure(1)
ax1 = plt.axes()

for (df, lab) in zip(df_list, df_label):
    df.plot(x='timesteps', y='mean_reward', ax=ax1, label=lab)

plt.ylabel("Episode reward")
plt.savefig(save_dir+"learning_curves.pdf", dpi=100)



def plot_shaded(df, ax, lab):
    ax.plot(df['timesteps'], df['mean_reward'], label=lab)
    ax.fill_between(df['timesteps'], df['lower'], df['upper'], alpha=0.35)



for (df, lab) in zip(df_list, df_label):
    plt.figure()
    ax = plt.axes()
    plot_shaded(df, ax, lab)

    plt.legend(loc="lower right")
    plt.ylabel("Mean reward")
    plt.xlabel("Timesteps")
    plt.savefig(save_dir+lab+".pdf", dpi=100)


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



ax = ff.plot.bar(x='exp type', y='mean success ratio', yerr='std success ratio', rot=45)
ax.set_xticklabels(df_label, ha='right')
plt.ylabel("Mean success ratio")
plt.tight_layout()
plt.savefig(save_dir+"success_by_exp_type.pdf", dpi=100)
# plt.show()


ax = ff.plot.bar(x='exp type', y='mean reach time', yerr='std reach time', rot=45)
ax.set_xticklabels(df_label, ha='right')
plt.ylabel("Mean reach time (max: 150)")
plt.tight_layout()
plt.savefig(save_dir+"reachtime_by_exp_type.pdf", dpi=100)
# plt.show()



ax = ff.plot.bar(x='exp type', y='mean reward', yerr='std reward (seed)', rot=45)
ax.set_xticklabels(df_label, ha='right')
plt.ylabel("Mean reward")
plt.tight_layout()
plt.savefig(save_dir+"reward_by_exp_type.pdf", dpi=100)
# plt.show()


ax = ff.plot.bar(x='exp type', y='mean train walltime (s)', yerr='std train walltime (s)', rot=45)
ax.set_xticklabels(df_label, ha='right')
plt.ylabel("Mean train time (s)")
plt.tight_layout()
plt.savefig(save_dir+"/walltime_by_exp_type.pdf", dpi=100)
# plt.show()


ax = ff.plot.bar(x='exp type', y='efficiency (reward / s)', rot=45)
ax.set_xticklabels(df_label, ha='right')
plt.ylabel("Eficiency (reward / s)")
plt.tight_layout()
plt.savefig(save_dir+"/efficiency_by_exp_type.pdf", dpi=100)
# plt.show()



