import gym
import time
import widowx_original
import widowx_pybullet
import widowx_physical
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

# env = gym.make('widowx_reach-v0')._start_sim(goal_oriented=False, render_bool=True)
env = gym.make('widowx_reach-v1')._start_sim(goal_oriented=False, render_bool=True)
# env = gym.make('widowx_reach-v2')._start_rospy(goal_oriented=False)   # requires a roscore to be running

print("Action space: ", env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


plot_bool = True
log_bool = False


if plot_bool:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

if log_bool:
    output = pd.DataFrame()


for episode in range(3):
    obs = env.reset()             
    rewards = []
    
    for t in range(20):
        action = env.action_space.sample()  
        obs, reward, done, info = env.step(action) 

        if plot_bool:
            goal = info['goal position']
            tip = info['tip position']

            ax1.cla()
            ax1.plot(goal[0], goal[2], marker='x', color='b')
            ax1.plot(tip[0], tip[2], marker='o', color='r')
            ax1.set_xlim([-0.2, 0.2])
            ax1.set_ylim([0, 0.5])
            ax1.set_xlabel("x")
            ax1.set_ylabel("z")

            ax2.cla()
            ax2.plot(goal[1], goal[2], marker='x', color='b')
            ax2.plot(tip[1], tip[2], marker='o', color='r')
            ax2.set_xlim([-0.2, 0.2])
            ax2.set_ylim([0, 0.5])
            ax2.set_xlabel("y")

            fig.suptitle("episode "+str(episode)+" | timestep "+str(t))
            plt.pause(0.01)
            # plt.show()

        if log_bool:
            d = info
            d['action'] = action
            d['obs'] = obs
            d['reward'] = reward
            d['done'] = done
            d['timestep'] = t
            d['episode'] = episode
            output = output.append(d, ignore_index=True)
        
        rewards.append(reward)
        # time.sleep(1./30.) 


    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))  

    if log_bool:
        # output.to_csv("logs/res_episode_"+str(episode)+".csv", index=False)  # slow
        output.to_pickle("logs/res_episode_"+str(episode)+".pkl")
        

env.close()


