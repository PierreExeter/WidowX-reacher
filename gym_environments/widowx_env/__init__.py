from gym.envs.registration import register


# environment for both the physical arm and the Pybullet simulation
register(id='widowx_reacher-v0',
         entry_point='widowx_env.envs.v0_widowx_original:WidowxEnv',
         max_episode_steps=100
         )

# environment for the physical arm only         
register(id='widowx_reacher-v1',
         entry_point='widowx_env.envs.v1_widowx_physical:WidowxEnv',
         max_episode_steps=100
         )

# environment for the physical arm only + no start_rospy method 
register(id='widowx_reacher-v2',
         entry_point='widowx_env.envs.v2_widowx_physical_no_start_rospy:WidowxEnv',
         max_episode_steps=100
         )

# FOR SOME STRANGE REASON, I CAN'T SPECIFY max_episode_steps for GoalEnv
# environment for the physical arm only + no start_rospy method + goal oriented
register(id='widowx_reacher-v3',
         entry_point='widowx_env.envs.v3_widowx_physical_no_start_rospy_goal_oriented:WidowxEnv',
         max_episode_steps=100
         )

# environment for the Pybullet simulation only. ROS install not required
register(id='widowx_reacher-v4',
         entry_point='widowx_env.envs.v4_widowx_pybullet:WidowxEnv',
         max_episode_steps=100
         )

# environment for the Pybullet simulation + no start_sim required
register(id='widowx_reacher-v5',
         entry_point='widowx_env.envs.v5_widowx_pybullet_no_start_sim:WidowxEnv',
         max_episode_steps=100
         )

# FOR SOME STRANGE REASON, I CAN'T SPECIFY max_episode_steps for GoalEnv
# environment for the Pybullet simulation + no start_sim required + goal_oriented
register(id='widowx_reacher-v6',
         entry_point='widowx_env.envs.v6_widowx_pybullet_no_start_sim_goal_oriented:WidowxEnv',
         max_episode_steps=100
         )

# test gym.GoalEnv
register(id='my_goal_env-v0',
         entry_point='widowx_env.envs.my_goal_env:MyGoalEnv',
         max_episode_steps=100
         )