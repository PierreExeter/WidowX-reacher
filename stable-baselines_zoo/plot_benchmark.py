import pandas as pd
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("results/benchmark_results.csv")

# Filter and sort
df1 = df[df['algo']=='ppo2']
df1 = df1.sort_values('n_timesteps')

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), dpi=300)

df1.plot(x='n_timesteps', y='mean_return', yerr='std_return', capsize=4, ax=ax1)
df1.plot(x='n_timesteps', y='mean_train_time(s)', yerr='std_train_time(s)', capsize=4, ax=ax2)
df1.plot(x='n_timesteps', y='mean_SR_50', yerr='std_SR_50', capsize=4, ax=ax3)
df1.plot(x='n_timesteps', y='mean_SR_20', yerr='std_SR_20', capsize=4, ax=ax3)
df1.plot(x='n_timesteps', y='mean_SR_10', yerr='std_SR_10', capsize=4, ax=ax3)
df1.plot(x='n_timesteps', y='mean_SR_5', yerr='std_SR_5', capsize=4, ax=ax3)
df1.plot(x='n_timesteps', y='mean_RT_50', yerr='std_RT_50', capsize=4, ax=ax4)
df1.plot(x='n_timesteps', y='mean_RT_20', yerr='std_RT_20', capsize=4, ax=ax4)
df1.plot(x='n_timesteps', y='mean_RT_10', yerr='std_RT_10', capsize=4, ax=ax4)
df1.plot(x='n_timesteps', y='mean_RT_5', yerr='std_RT_5', capsize=4, ax=ax4)

ax1.set_ylabel("Mean return")
ax2.set_ylabel("Train time (s)")
ax3.set_ylabel("Success ratio")
ax4.set_ylabel("Reach time")

plt.tight_layout()
# plt.show()
plt.savefig("results/ppo2_timesteps.png")
