import pandas as pd
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("results/benchmark_results.csv")

# be careful: ent_coeff is a string. I might need to convert it to a float in the future
# df['ent_coef'] = df['ent_coef'].astype(float)


def plot_df(dff, col, filename):
    """
    dff: dataframe to plot
    col: column name to plot on x axis
    filename: name of output png file
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), dpi=300)

    dff.plot(x=col, y='mean_return', yerr='std_return', capsize=4, ax=ax1)
    dff.plot(x=col, y='mean_train_time(s)', yerr='std_train_time(s)', capsize=4, ax=ax2)
    dff.plot(x=col, y='mean_SR_50', yerr='std_SR_50', capsize=4, ax=ax3)
    dff.plot(x=col, y='mean_SR_20', yerr='std_SR_20', capsize=4, ax=ax3)
    dff.plot(x=col, y='mean_SR_10', yerr='std_SR_10', capsize=4, ax=ax3)
    dff.plot(x=col, y='mean_SR_5', yerr='std_SR_5', capsize=4, ax=ax3)
    dff.plot(x=col, y='mean_SR_2', yerr='std_SR_2', capsize=4, ax=ax3)
    dff.plot(x=col, y='mean_SR_1', yerr='std_SR_1', capsize=4, ax=ax3)
    dff.plot(x=col, y='mean_SR_05', yerr='std_SR_05', capsize=4, ax=ax3)
    dff.plot(x=col, y='mean_RT_50', yerr='std_RT_50', capsize=4, ax=ax4)
    dff.plot(x=col, y='mean_RT_20', yerr='std_RT_20', capsize=4, ax=ax4)
    dff.plot(x=col, y='mean_RT_10', yerr='std_RT_10', capsize=4, ax=ax4)
    dff.plot(x=col, y='mean_RT_5', yerr='std_RT_5', capsize=4, ax=ax4)

    ax1.set_ylabel("Mean return")
    ax2.set_ylabel("Train time (s)")
    ax3.set_ylabel("Success ratio")
    ax4.set_ylabel("Reach time")

    # uncomment if var = n_timesteps (clearer plot)
    ## ax3.set_xscale('symlog', linthreshy=1e-1)
    ## ax3.set_yscale('symlog', linthreshy=1e-1)
    # ax3.set_xscale('log')
    # ax3.set_yscale('log')

    ax3.legend(bbox_to_anchor=(1, 1.05))
    ax4.legend(bbox_to_anchor=(1.2, 1.05))

    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)


def x_variable(dfa, mask, var):
    """
    apply mask to dfa, then plot variable var vs metrics
    """
    dfb = dfa[mask]

    # ent_coef is a string and must be converted into a float
    if var == "ent_coef":
        dfb['ent_coef'] = dfb['ent_coef'].astype(float)

    dfb = dfb.sort_values(var)
    print(dfb[var])
    # print(dfb["mean_SR_5"])
    plot_df(dfb, var, "results/plots/ppo2_"+var+"_deterministic_largeBounds.png")


print("max success ratio @5mm:",  df["mean_SR_5"].max())
print("best config:", df["mean_SR_5"].argmax())

# DEFAULT MASK WITH DEFAULT HYPERPRAMS
# mask = (df['algo']=='ppo2') & \
# (df['normalize']==True)  & \
# (df['n_envs']==8) & \
# (df['n_timesteps']==500000) & \
# (df['gamma']==0.99) & \
# (df['n_steps']==128) & \
# (df['ent_coef']=='0.01')  & \
# (df['learning_rate']==0.00025) & \
# (df['vf_coef']==0.5) & \
# (df['lam']==0.95) & \
# (df['nminibatches']==4) & \
# (df['noptepochs']==4) & \
# (df['cliprange']==0.2)& \
# (df['deterministic'] == 0) & \
# (df['env_id'] == 'widowx_reacher-v5')

#
# # TIMESTEPS
# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'n_timesteps')

# NORMALISATION
#
# mask = (df['algo'] == 'ppo2') & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'normalize')

#
# ### N_ENVS
#
# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'n_envs')
#
# # GAMMA
#
# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'gamma')

# NSTEPS

# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'n_steps')

# ENT_COEFF

# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'ent_coef')
#
# LEARNING RATE
#
# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'learning_rate')


# LAM

# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'lam')
# #
#
# # CLIPRANGE
#
# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 4) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'cliprange')

#
# NMINIBATCHES
#
# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['noptepochs'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# x_variable(df, mask, 'nminibatches')
#
# # NOPTEPOCHS
#
# mask = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['bounds'] == 'large') & \
#     (df['env_id'] == 'widowx_reacher-v28')
#
# x_variable(df, mask, 'noptepochs')

#
# # ALGO
#
# dfb = df.loc[:7]
# plot_df(dfb, 'algo', "results/plots/algo_deterministic.png")


# REWARD: dist, dist**2, dist**3, dist**4

# mask1 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 20) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# mask2 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 20) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v14')
#
# mask3 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 20) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v15')
#
# mask4 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 20) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v16')
#
#
# df_reward_dist = pd.concat([df[mask1], df[mask2], df[mask3], df[mask4]])
# #
# # df_reward_dist = df_reward_dist.reindex([74, 62, 72, 73])


# print(df_reward_dist)
# #
# plot_df(df_reward_dist, 'reward', "results/plots/reward_dist_deterministic.png")

#
# actionStepCoeff: 1, 5, 10, 20, 30, 40, 50, 60 100, 1000
#
# mask1 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v5')
#
# mask2 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v17')
#
# mask3 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v18')
#
# mask4 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v19')
#
# mask5 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v20')
#
# mask6 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v21')
#
# mask7 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v22')
#
# mask8 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v23')
#
# mask9 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v24')
#
# mask10 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v25')
#
# mask11 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v26')
#
# mask12 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['noptepochs'] == 50) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v27')
#
#
# df_actionStepCoeff = pd.concat([df[mask1], df[mask2], df[mask3], df[mask4], df[mask5],
#                                 df[mask6], df[mask7], df[mask8], df[mask9], df[mask10], df[mask11], df[mask12]])
#
# df_actionStepCoeff = df_actionStepCoeff.sort_values(by=['actionStepCoeff'])[:-1]   # removed actionStepCoeff=1000
# # print(df_actionStepCoeff['actionStepCoeff'])
#
# plot_df(df_actionStepCoeff, 'actionStepCoeff', "results/plots/actionStepCoeff_deterministic.png")
#




# ---------------
# actionStepCoeff + large bounds

mask1 = (df['algo'] == 'ppo2') & \
    (df['normalize'] == True) & \
    (df['n_envs'] == 8) & \
    (df['n_timesteps'] == 500000) & \
    (df['gamma'] == 0.99) & \
    (df['n_steps'] == 128) & \
    (df['ent_coef'] == '0.01') & \
    (df['learning_rate'] == 0.00025) & \
    (df['vf_coef'] == 0.5) & \
    (df['lam'] == 0.95) & \
    (df['nminibatches'] == 4) & \
    (df['noptepochs'] == 30) & \
    (df['cliprange'] == 0.2) & \
    (df['deterministic'] == 1) & \
    (df['bounds'] == 'large') & \
    (df['env_id'] == 'widowx_reacher-v28')

mask2 = (df['algo'] == 'ppo2') & \
    (df['normalize'] == True) & \
    (df['n_envs'] == 8) & \
    (df['n_timesteps'] == 500000) & \
    (df['gamma'] == 0.99) & \
    (df['n_steps'] == 128) & \
    (df['ent_coef'] == '0.01') & \
    (df['learning_rate'] == 0.00025) & \
    (df['vf_coef'] == 0.5) & \
    (df['lam'] == 0.95) & \
    (df['nminibatches'] == 4) & \
    (df['noptepochs'] == 30) & \
    (df['cliprange'] == 0.2) & \
    (df['deterministic'] == 1) & \
    (df['bounds'] == 'large') & \
    (df['env_id'] == 'widowx_reacher-v29')

mask3 = (df['algo'] == 'ppo2') & \
    (df['normalize'] == True) & \
    (df['n_envs'] == 8) & \
    (df['n_timesteps'] == 500000) & \
    (df['gamma'] == 0.99) & \
    (df['n_steps'] == 128) & \
    (df['ent_coef'] == '0.01') & \
    (df['learning_rate'] == 0.00025) & \
    (df['vf_coef'] == 0.5) & \
    (df['lam'] == 0.95) & \
    (df['nminibatches'] == 4) & \
    (df['noptepochs'] == 30) & \
    (df['cliprange'] == 0.2) & \
    (df['deterministic'] == 1) & \
    (df['bounds'] == 'large') & \
    (df['env_id'] == 'widowx_reacher-v30')

mask4 = (df['algo'] == 'ppo2') & \
    (df['normalize'] == True) & \
    (df['n_envs'] == 8) & \
    (df['n_timesteps'] == 500000) & \
    (df['gamma'] == 0.99) & \
    (df['n_steps'] == 128) & \
    (df['ent_coef'] == '0.01') & \
    (df['learning_rate'] == 0.00025) & \
    (df['vf_coef'] == 0.5) & \
    (df['lam'] == 0.95) & \
    (df['nminibatches'] == 4) & \
    (df['noptepochs'] == 30) & \
    (df['cliprange'] == 0.2) & \
    (df['deterministic'] == 1) & \
    (df['bounds'] == 'large') & \
    (df['env_id'] == 'widowx_reacher-v31')

mask5 = (df['algo'] == 'ppo2') & \
    (df['normalize'] == True) & \
    (df['n_envs'] == 8) & \
    (df['n_timesteps'] == 500000) & \
    (df['gamma'] == 0.99) & \
    (df['n_steps'] == 128) & \
    (df['ent_coef'] == '0.01') & \
    (df['learning_rate'] == 0.00025) & \
    (df['vf_coef'] == 0.5) & \
    (df['lam'] == 0.95) & \
    (df['nminibatches'] == 4) & \
    (df['noptepochs'] == 30) & \
    (df['cliprange'] == 0.2) & \
    (df['deterministic'] == 1) & \
    (df['bounds'] == 'large') & \
    (df['env_id'] == 'widowx_reacher-v32')

mask6 = (df['algo'] == 'ppo2') & \
    (df['normalize'] == True) & \
    (df['n_envs'] == 8) & \
    (df['n_timesteps'] == 500000) & \
    (df['gamma'] == 0.99) & \
    (df['n_steps'] == 128) & \
    (df['ent_coef'] == '0.01') & \
    (df['learning_rate'] == 0.00025) & \
    (df['vf_coef'] == 0.5) & \
    (df['lam'] == 0.95) & \
    (df['nminibatches'] == 4) & \
    (df['noptepochs'] == 30) & \
    (df['cliprange'] == 0.2) & \
    (df['deterministic'] == 1) & \
    (df['bounds'] == 'large') & \
    (df['env_id'] == 'widowx_reacher-v33')


df_actionStepCoeff = pd.concat([df[mask1], df[mask2], df[mask3], df[mask4], df[mask5], df[mask6]])

df_actionStepCoeff = df_actionStepCoeff.sort_values(by=['actionStepCoeff'])
print(df_actionStepCoeff['actionStepCoeff'])

plot_df(df_actionStepCoeff, 'actionStepCoeff', "results/plots/actionStepCoeff_deterministic_largeBounds.png")
#




# ------------- noptepoch50 at 0.5M and 20M timeteps + v22 (actionStepCoeff = 30)
#
# mask1 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 500000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v22')
#
# mask2 = (df['algo'] == 'ppo2') & \
#     (df['normalize'] == True) & \
#     (df['n_envs'] == 8) & \
#     (df['n_timesteps'] == 20000000) & \
#     (df['gamma'] == 0.99) & \
#     (df['n_steps'] == 128) & \
#     (df['ent_coef'] == '0.01') & \
#     (df['learning_rate'] == 0.00025) & \
#     (df['vf_coef'] == 0.5) & \
#     (df['lam'] == 0.95) & \
#     (df['nminibatches'] == 4) & \
#     (df['cliprange'] == 0.2) & \
#     (df['deterministic'] == 1) & \
#     (df['env_id'] == 'widowx_reacher-v22')
#
#
# df_env22 = pd.concat([df[mask1], df[mask2]])
#
# print(df_env22)
#
# plot_df(df_env22, 'n_timesteps', "results/plots/n_timesteps_env22_deterministic.png")
