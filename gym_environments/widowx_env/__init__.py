from gym.envs.registration import register


# environment for both the physical arm and the Pybullet simulation
register(id='widowx_reacher-v0',
         entry_point='widowx_env.envs.v0_widowx_original:WidowxEnv',
         max_episode_steps=100
         )

# environment for the physical arm only         
register(id='widowx_reacher-v1',
         entry_point='widowx_env.envs.v1_widowx_physical_only:WidowxEnv',
         max_episode_steps=100
         )

# environment for the physical arm only + no start_rospy method 
register(id='widowx_reacher-v2',
         entry_point='widowx_env.envs.v2_widowx_physical_env_fixedGoal:WidowxEnv',
         max_episode_steps=100
         )

# FOR SOME STRANGE REASON, I CAN'T SPECIFY max_episode_steps for GoalEnv (solved)
# environment for the physical arm only + no start_rospy method + goal oriented
register(id='widowx_reacher-v3',
         entry_point='widowx_env.envs.v3_widowx_physical_goalEnv_fixedGoal:WidowxEnv',
         max_episode_steps=100
         )

# environment for the Pybullet simulation only. ROS install not required
register(id='widowx_reacher-v4',
         entry_point='widowx_env.envs.v4_widowx_pybullet_only:WidowxEnv',
         max_episode_steps=100
         )

# environment for the Pybullet simulation + no start_sim required + fixed goal
register(id='widowx_reacher-v5',
         entry_point='widowx_env.envs.v5_widowx_pybullet_env_fixedGoal:WidowxEnv',
         max_episode_steps=100
         )

# FOR SOME STRANGE REASON, I CAN'T SPECIFY max_episode_steps for GoalEnv
# environment for the Pybullet simulation + no start_sim required + goal_oriented + fixed goal
register(id='widowx_reacher-v6',
         entry_point='widowx_env.envs.v6_widowx_pybullet_goalEnv_fixedGoal:WidowxEnv',
         max_episode_steps=100
         )

# environment for the Pybullet simulation + no start_sim required + random goal
register(id='widowx_reacher-v7',
         entry_point='widowx_env.envs.v7_widowx_pybullet_env_randomGoal:WidowxEnv',
         max_episode_steps=100
         )

# FOR SOME STRANGE REASON, I CAN'T SPECIFY max_episode_steps for GoalEnv (solved)
# environment for the Pybullet simulation + no start_sim required + goal_oriented + random goal
register(id='widowx_reacher-v8',
         entry_point='widowx_env.envs.v8_widowx_pybullet_goalEnv_randomGoal:WidowxEnv',
         max_episode_steps=100
         )

# environment for the physical arm only + no start_rospy method + random goal
register(id='widowx_reacher-v12',
         entry_point='widowx_env.envs.v12_widowx_physical_env_randomGoal:WidowxEnv',
         max_episode_steps=100
         )
         
# environment for the physical arm only + no start_rospy method + goal environment + random goal
register(id='widowx_reacher-v13',
         entry_point='widowx_env.envs.v13_widowx_physical_goalEnv_randomGoal:WidowxEnv',
         max_episode_steps=100
         )

# test gym.GoalEnv
register(id='my_goal_env-v0',
         entry_point='widowx_env.envs.my_goal_env:MyGoalEnv',
         max_episode_steps=100
         )