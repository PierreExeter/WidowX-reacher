import gym
import time
import widowx_env
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from plot_lib import plot_df

env = gym.make('widowx_reacher-v5')
filename = "13_action5_+0.05"

# print(env)

print("Action space: ", env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

log_df = pd.DataFrame()
log_dict = OrderedDict()

for episode in range(1):
    obs = env.reset()
    rewards = []

    for t in range(100):
        # action = env.action_space.sample()
        action = [0, 0, 0, 0, 0.05, 0]
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        cumulative_reward = sum(rewards)

        # print("action: ", action)
        # print("obs: ", obs)
        # print("reward: ", cumulative_reward)
        # print("done: ", done)
        # print("info: ", info)
        print("timestep: ", t)

        log_dict['episode'] = episode
        log_dict['timestep'] = t
        log_dict['action_1'] = action[0]
        log_dict['action_2'] = action[1]
        log_dict['action_3'] = action[2]
        log_dict['action_4'] = action[3]
        log_dict['action_5'] = action[4]
        log_dict['action_6'] = action[5]
        log_dict['current_joint_pos_1'] = info['current_joint_pos'][0]
        log_dict['current_joint_pos_2'] = info['current_joint_pos'][1]
        log_dict['current_joint_pos_3'] = info['current_joint_pos'][2]
        log_dict['current_joint_pos_4'] = info['current_joint_pos'][3]
        log_dict['current_joint_pos_5'] = info['current_joint_pos'][4]
        log_dict['current_joint_pos_6'] = info['current_joint_pos'][5]
        log_dict['new_joint_pos_1'] = info['new_joint_pos'][0]
        log_dict['new_joint_pos_2'] = info['new_joint_pos'][1]
        log_dict['new_joint_pos_3'] = info['new_joint_pos'][2]
        log_dict['new_joint_pos_4'] = info['new_joint_pos'][3]
        log_dict['new_joint_pos_5'] = info['new_joint_pos'][4]
        log_dict['new_joint_pos_6'] = info['new_joint_pos'][5]
        log_dict['joint_vel_1'] = info['joint_vel'][0]
        log_dict['joint_vel_2'] = info['joint_vel'][1]
        log_dict['joint_vel_3'] = info['joint_vel'][2]
        log_dict['joint_vel_4'] = info['joint_vel'][3]
        log_dict['joint_vel_5'] = info['joint_vel'][4]
        log_dict['joint_vel_6'] = info['joint_vel'][5]
        log_dict['joint1_min'] = -3.1
        log_dict['joint1_max'] = 3.1
        log_dict['joint2_min'] = -1.571
        log_dict['joint2_max'] = 1.571
        log_dict['joint3_min'] = -1.571
        log_dict['joint3_max'] = 1.571
        log_dict['joint4_min'] = -1.745
        log_dict['joint4_max'] = 1.745
        log_dict['joint5_min'] = -2.617
        log_dict['joint5_max'] = 2.617
        log_dict['joint6_min'] = 0.003
        log_dict['joint6_max'] = 0.03
        log_dict['action_low1'] = env.action_space.low[0]
        log_dict['action_low2'] = env.action_space.low[1]
        log_dict['action_low3'] = env.action_space.low[2]
        log_dict['action_low4'] = env.action_space.low[3]
        log_dict['action_low5'] = env.action_space.low[4]
        log_dict['action_low6'] = env.action_space.low[5]
        log_dict['action_high1'] = env.action_space.high[0]
        log_dict['action_high2'] = env.action_space.high[1]
        log_dict['action_high3'] = env.action_space.high[2]
        log_dict['action_high4'] = env.action_space.high[3]
        log_dict['action_high5'] = env.action_space.high[4]
        log_dict['action_high6'] = env.action_space.high[5]
        log_dict['reward'] = reward
        log_dict['return'] = cumulative_reward
        log_dict['dist'] = info['total_distance']
        log_dict['target_x'] = info['goal position'][0]
        log_dict['target_y'] = info['goal position'][1]
        log_dict['target_z'] = info['goal position'][2]
        log_dict['tip_x'] = info['tip position'][0]
        log_dict['tip_y'] = info['tip position'][1]
        log_dict['tip_z'] = info['tip position'][2]
        log_dict['done'] = done
        # log_dict['obs'] = obs
        # log_dict['obs_space_low'] = env.observation_space.low
        # log_dict['obs_space_high'] = env.observation_space.high


        log_df = log_df.append(log_dict, ignore_index=True)

        # time.sleep(1./30.)


log_df = log_df[log_dict.keys()]  # sort columns

# add estimated tip velocity (according to the documentation, 1 timestep = 240 Hz)
log_df['est_vel'] = log_df['dist'].diff()*240

log_df.to_csv("logs/"+filename+".csv", index=False)

env.close()

plot_df(log_df, "plots/"+filename+".png")
