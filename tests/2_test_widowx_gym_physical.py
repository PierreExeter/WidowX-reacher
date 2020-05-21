import gym
import time
import gym_replab
from rlkit.envs.wrappers import NormalizedBoxEnv

env = gym.make('replab-v0')._start_rospy(goal_oriented=False)
#env.action_space.low *= 10
#env.action_space.high *= 10
env = NormalizedBoxEnv(env)

print("Action space: ", env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


for episode in range(3):
    obs = env.reset()
    rewards = []
    time.sleep(1) # wait for the action to complete 

    for t in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        print("timestep: ", t)

        rewards.append(reward)
        time.sleep(1) # wait for the action to complete 

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))

env.close()


