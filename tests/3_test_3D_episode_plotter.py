import gym
import time
import widowx_original
import widowx_pybullet
import widowx_physical
import pandas as pd
import numpy as np

# env = gym.make('widowx_reach-v0')._start_sim(goal_oriented=False, render_bool=True)
env = gym.make('widowx_reach-v1')._start_sim(goal_oriented=False, render_bool=False)
# env = gym.make('widowx_reach-v2')._start_rospy(goal_oriented=False)   # requires a roscore to be running

print("Action space: ", env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


d = {}
output = pd.DataFrame()

count = 0

for episode in range(3):
    obs = env.reset()             
    rewards = []
    
    for t in range(2000):
        action = env.action_space.sample()  
        obs, reward, done, info = env.step(action) 

        # d = info['']
        d['goal position'] = np.array(info['goal position'])
        d['joint position'] = np.array(info['joint position'])
        d['tip position'] = np.array(info['tip position'])
        d['total_distance'] = np.array(info['total_distance'])
        d['action'] = np.array(action)
        d['obs'] = np.array(obs)
        d['reward'] = np.array(reward)
        d['done'] = np.array(done)
        d['timestep'] = np.array(t)
        d['episode'] = np.array(episode)
        # print("reward type", d['reward'].dtype)
        # print("obs type", d['obs'].dtype)
        # print("obs", obs)


        output = output.append(d, ignore_index=True)
        # output.to_csv("res_episode_"+str(episode)+".csv", index=False)
        output.to_pickle("res_episode_"+str(episode)+".pkl")
        print(count)
        count += 1

        rewards.append(reward)
        # time.sleep(1./30.) 

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))  
    
env.close()


