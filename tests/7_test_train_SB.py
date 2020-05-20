from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
import gym
# import widowx_pybullet
import widowx_pybullet_no_start_sim
import os


log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# env = gym.make('widowx_reach-v1')._start_sim(goal_oriented=False, render_bool=False)
env = gym.make('widowx_reach-v3') 
# env = gym.make('Pendulum-v0')
env = Monitor(env, log_dir)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000, log_interval=10)
model.save("widowx_reach-v1")
