import gym
import time
import widowx_original
import widowx_pybullet
import widowx_physical
import widowx_pybullet_no_start_sim

# env = gym.make('widowx_reach-v0')._start_sim(goal_oriented=False, render_bool=True)
# env = gym.make('widowx_reach-v1')._start_sim(goal_oriented=False, render_bool=True)
# env = gym.make('widowx_reach-v2')._start_rospy(goal_oriented=False)   # requires a roscore to be running
env = gym.make('widowx_reach-v3')   

print("Action space: ", env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


for episode in range(3):
    obs = env.reset()             
    rewards = []
    
    for t in range(2000):
        action = env.action_space.sample()  
        obs, reward, done, info = env.step(action) 

        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        print("timestep: ", t)

        rewards.append(reward)
        time.sleep(1./30.) 

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))  
    
env.close()


