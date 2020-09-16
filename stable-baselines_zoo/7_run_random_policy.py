import gym, widowx_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import time
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,)
    parser.add_argument('--nb-seeds', help='Number of seeds to evaluate', type=int, default=0)
    
    # get arguments
    args = parser.parse_args()
    env_id = args.env
    log_path = args.folder
    nb_timesteps = int(args.n_timesteps)
    nb_seeds = int(args.nb_seeds)

    os.makedirs(log_path, exist_ok=True)

    env = gym.make(env_id)

    ## LEARNING CURVE
    
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

    # # convert to min
    # mean_walltime /= 60
    # std_walltime /= 60

    d_walltime = {"mean_walltime": mean_walltime, "std_walltime": std_walltime}
    df_walltime = pd.DataFrame(d_walltime, index=[0])
    df_walltime.to_csv(log_path+"/walltime.csv", index=False)

    ## reward
    # print(all_rewards)
    all_rewards_df = pd.concat(all_rewards, axis=1)
    all_rewards_df['timesteps'] = pd.Series(timesteps)
    print(all_rewards_df)

    all_rewards_df.to_csv(log_path+"/all_rewards.csv", index=False)


if __name__ == '__main__':
    main()