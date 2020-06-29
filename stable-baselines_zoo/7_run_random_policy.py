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

log_path = "logs/random_policy_0.2M/"+env_id
os.makedirs(log_path, exist_ok=True)


## LEARNING CURVE
         
nb_timesteps = 200000
nb_seeds = 10
walltime_seed = []
all_rewards = []
ep_reward = 0

start_time = time.time()

for seed in range(nb_seeds):

    obs = env.reset()    
    rewards = []
    timesteps = []

    for t in range(nb_timesteps):
        action = env.action_space.sample()  
        obs, reward, done, info = env.step(action) 
        ep_reward += reward

        if done:
            rewards.append(ep_reward)
            timesteps.append(t+1)
            ep_reward = 0
            obs = env.reset()

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
all_rewards_df['timesteps'] = pd.Series(timesteps)
print(all_rewards_df)

all_rewards_df.to_csv(log_path+"/all_rewards.csv", index=False)


