from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.envs.wrappers import NormalizedBoxEnv
import joblib
import gym
import gym_replab
import time
import numpy as np


# need to run save clean pickle before... (to remove env saved)

# # LOAD DATA
# path = 'rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_15_29_53_0000--s-0/params.pkl'
# data = joblib.load(path)
# print(data)

# # LOAD POLICY
# policy = data['policy']
# print(policy)

# DEFINE ENVIRONMENT
# env = data['env']._start_sim(goal_oriented=False, render_bool=True)
env = gym.make('replab-v0')._start_sim(goal_oriented=False, render_bool=True)
env = NormalizedBoxEnv(env)

# SIMULATE POLICY
for episode in range(10):
    state = env.reset()
    rewards = []

    for t in range(100):

        action = env.action_space.sample()  
        # action, agent_info = policy.get_action(state)
        state, reward, done, info = env.step(action) 

        # print("action: ", action)
        # print("state: ", state)
        # print("reward: ", reward)
        # print("done: ", done)
        # print("info: ", info)
        # print("timestep: ", t)
        # print('episode: ', episode)

        rewards.append(reward)
        time.sleep(1./30.) 

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))  

env.close()


