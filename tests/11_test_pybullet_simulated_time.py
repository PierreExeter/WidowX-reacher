import gym
import widowx_env
import time

env = gym.make('widowx_reacher-v5')
env.reset()
nb_timesteps = 100000

start = time.time()

for t in range(nb_timesteps):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # print(t)

end = time.time()
tot = end - start
print("Execution time (s): ", tot)
print("Pybullet simulation time (s): ", nb_timesteps / 240)
print("Pybullet step time (s): ", 1/240)
print("Measured step time (s): ", tot / nb_timesteps)


env.close()
