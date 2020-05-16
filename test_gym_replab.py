import gym
import time
import gym_replab

env = gym.make('replab-v0')._start_sim(goal_oriented=False, render=True)

print("Action space: ", env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


for episode in range(3):
    obs = env.reset()             
    rewards = []
    
    for t in range(200):
        action = env.action_space.sample()  
        obs, reward, done, info = env.step(action) 

        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        print("timestep: ", t)

        rewards.append(reward)
        # time.sleep(1./30.) 

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))  
    
env.close()


