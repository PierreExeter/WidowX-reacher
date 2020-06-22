import gym, widowx_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import time


env_id = "widowx_reacher-v7"
env = gym.make(env_id)

log_path = "logs/random_policy/"+env_id
os.makedirs(log_path, exist_ok=True)


# print(env)

# normalise action space, observation space and reward
# env.action_space.low *= 10
# env.action_space.high *= 10
# env = NormalizedBoxEnv(env)
# env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

# # save env
# env.save("vec_normalize.pkl")

# # load env
# env = DummyVecEnv([lambda: env])
# env = VecNormalize.load("vec_normalize.pkl", env)



# # comment this when using widowx_reacher-v3 and widowx_reacher-v6 (goal oriented env, observation is a dict)
# print("Action space: ", env.action_space)
# print(env.action_space.high)
# print(env.action_space.low)
# print("Observation space: ", env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)


start_time = time.time()

## LEARNING CURVE
         
nb_timesteps = 500000
nb_seeds = 10
walltime_seed = []
all_rewards = []

for seed in range(nb_seeds):

    obs = env.reset()    
    rewards = []
    timesteps = []

    for t in range(nb_timesteps):
        action = env.action_space.sample()  
        obs, reward, done, info = env.step(action) 

        rewards.append(reward)
        timesteps.append(t)

    # walltime
    end_time = time.time()
    walltime = end_time - start_time
    walltime_seed.append(walltime)
    
    # reward
    df_rewards = pd.Series(rewards, name="seed_"+str(seed))
    all_rewards.append(df_rewards)


env.close()

## walltime
print(walltime_seed)
mean_walltime = np.mean(walltime_seed)
std_walltime = np.std(walltime_seed)

# convert to min
mean_walltime /= 60
std_walltime /= 60

d_walltime = {"mean_walltime": mean_walltime, "std_walltime": std_walltime}
df_walltime = pd.DataFrame(d_walltime, index=[0])
df_walltime.to_csv(log_path+"/walltime.csv", index=False)

## reward
# print(all_rewards)
all_rewards_df = pd.concat(all_rewards, axis=1)
all_rewards_df.to_csv(log_path+"/all_rewards.csv", index=False)
print(all_rewards_df)

all_rewards_smooth_df = all_rewards_df
all_rewards_smooth_df["mean_reward"] = all_rewards_df.mean(axis=1)
all_rewards_smooth_df["std_reward"] = all_rewards_df.std(axis=1)
all_rewards_smooth_df["upper"] = all_rewards_df["mean_reward"] + all_rewards_df["std_reward"]
all_rewards_smooth_df["lower"] = all_rewards_df["mean_reward"] - all_rewards_df["std_reward"]

all_rewards_smooth_df['timesteps'] = pd.Series(timesteps)
print(all_rewards_smooth_df)

# apply rolling window (except on timesteps)
for col in all_rewards_smooth_df.columns[:-1]:
    all_rewards_smooth_df[col] = all_rewards_smooth_df[col].rolling(window=50).mean()

all_rewards_smooth_df.dropna(inplace=True)  # remove NaN due to rolling average
all_rewards_smooth_df.to_csv(log_path+"/all_rewards_smooth.csv", index=False)
print(all_rewards_smooth_df)




# plot

def plot_shaded(df, ax, lab):
    ax.plot(df['timesteps'], df['mean_reward'], label=lab)
    ax.fill_between(df['timesteps'], df['lower'], df['upper'], alpha=0.35)


plt.figure(2, figsize=(10, 5))
ax = plt.axes()
plot_shaded(all_rewards_smooth_df, ax, "random policy")

plt.legend(loc="lower right")
plt.ylabel("Mean reward")
plt.xlabel("Timesteps")
plt.savefig(log_path+"/training_curve.pdf", dpi=100)
# plt.show()
