from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import SAC, PPO2, A2C
import gym
import widowx_pybullet


env = gym.make('widowx_reach-v1')._start_sim(goal_oriented=False, render_bool=True)
model = PPO2(MlpPolicy, env, verbose=1)
model = PPO2.load("widowx_reach-v1")
# env.render(mode="human")  
 
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)