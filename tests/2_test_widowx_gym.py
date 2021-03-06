import gym
import time
import widowx_env
from rlkit.envs.wrappers import NormalizedBoxEnv
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.env_checker import check_env


# env = gym.make('widowx_reacher-v0').start_sim(goal_oriented=False, render_bool=True)   # or .start_rospy(goal_oriented=False)
# env = gym.make('widowx_reacher-v1').start_rospy(goal_oriented=False)                   # requires a roscore to be running
# env = gym.make('widowx_reacher-v2')                                              # requires a roscore to be running
# env = gym.make('widowx_reacher-v3')                                              # requires a roscore to be running
# env = gym.make('widowx_reacher-v4').start_sim(goal_oriented=False, render_bool=True)
env = gym.make('widowx_reacher-v5')
# env = gym.make('widowx_reacher-v6')
# env = gym.make('widowx_reacher-v7')
# env = gym.make('widowx_reacher-v8')
# env = gym.make('widowx_reacher-v12')
# env = gym.make('widowx_reacher-v13')
# env = gym.make('widowx_reacher-v14')
# env = gym.make('widowx_reacher-v15')

print("isinstance(env, gym.GoalEnv)", isinstance(env, gym.Env))

# It will check your custom environment and output additional warnings if needed
print("any warnings?", check_env(env))


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

# print(env)

# # comment this when using widowx_reacher-v3 and widowx_reacher-v6 (goal oriented env, observation is a dict)
# print("Action space: ", env.action_space)
# print(env.action_space.high)
# print(env.action_space.low)
# print("Observation space: ", env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)


for episode in range(1):
    obs = env.reset()
    rewards = []

    for t in range(100):
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
