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

# environment for the pybullet simulation + fixed goal + reward = -dist**3
register(id='widowx_reacher-v14',
         entry_point='widowx_env.envs.v14_widowx_pybullet_env_fixedGoal_dist3:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + reward = -dist**4
register(id='widowx_reacher-v15',
         entry_point='widowx_env.envs.v15_widowx_pybullet_env_fixedGoal_dist4:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + reward = -dist
register(id='widowx_reacher-v16',
         entry_point='widowx_env.envs.v16_widowx_pybullet_env_fixedGoal_dist:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 100
register(id='widowx_reacher-v17',
         entry_point='widowx_env.envs.v17_widowx_pybullet_env_fixedGoal_actionStepCoeff100:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 1
register(id='widowx_reacher-v18',
         entry_point='widowx_env.envs.v18_widowx_pybullet_env_fixedGoal_actionStepCoeff1:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 1000
register(id='widowx_reacher-v19',
         entry_point='widowx_env.envs.v19_widowx_pybullet_env_fixedGoal_actionStepCoeff1000:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 20
register(id='widowx_reacher-v20',
         entry_point='widowx_env.envs.v20_widowx_pybullet_env_fixedGoal_actionStepCoeff20:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 5
register(id='widowx_reacher-v21',
         entry_point='widowx_env.envs.v21_widowx_pybullet_env_fixedGoal_actionStepCoeff5:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 30
register(id='widowx_reacher-v22',
         entry_point='widowx_env.envs.v22_widowx_pybullet_env_fixedGoal_actionStepCoeff30:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 40
register(id='widowx_reacher-v23',
         entry_point='widowx_env.envs.v23_widowx_pybullet_env_fixedGoal_actionStepCoeff40:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 50
register(id='widowx_reacher-v24',
         entry_point='widowx_env.envs.v24_widowx_pybullet_env_fixedGoal_actionStepCoeff50:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 60
register(id='widowx_reacher-v25',
         entry_point='widowx_env.envs.v25_widowx_pybullet_env_fixedGoal_actionStepCoeff60:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25
register(id='widowx_reacher-v26',
         entry_point='widowx_env.envs.v26_widowx_pybullet_env_fixedGoal_actionStepCoeff25:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 35
register(id='widowx_reacher-v27',
         entry_point='widowx_env.envs.v27_widowx_pybullet_env_fixedGoal_actionStepCoeff35:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 10 + no bounds
register(id='widowx_reacher-v28',
         entry_point='widowx_env.envs.v28_widowx_pybullet_env_fixedGoal_actionStepCoeff10_nobounds:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 20 + no bounds
register(id='widowx_reacher-v29',
         entry_point='widowx_env.envs.v29_widowx_pybullet_env_fixedGoal_actionStepCoeff20_nobounds:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds
register(id='widowx_reacher-v30',
         entry_point='widowx_env.envs.v30_widowx_pybullet_env_fixedGoal_actionStepCoeff25_nobounds:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 30 + no bounds
register(id='widowx_reacher-v31',
         entry_point='widowx_env.envs.v31_widowx_pybullet_env_fixedGoal_actionStepCoeff30_nobounds:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 35 + no bounds
register(id='widowx_reacher-v32',
         entry_point='widowx_env.envs.v32_widowx_pybullet_env_fixedGoal_actionStepCoeff35_nobounds:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 40 + no bounds
register(id='widowx_reacher-v33',
         entry_point='widowx_env.envs.v33_widowx_pybullet_env_fixedGoal_actionStepCoeff40_nobounds:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + dense and sparse reward
register(id='widowx_reacher-v34',
         entry_point='widowx_env.envs.v34_widowx_pybullet_env_fixedGoal_actionStepCoeff25_nobounds_sparseDense:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only
register(id='widowx_reacher-v35',
         entry_point='widowx_env.envs.v35_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + reward shape: equation 5
register(id='widowx_reacher-v36',
         entry_point='widowx_env.envs.v36_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_reward5:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + reward shape: equation 6
register(id='widowx_reacher-v37',
         entry_point='widowx_env.envs.v37_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_reward6:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + reward shape: equation 10
register(id='widowx_reacher-v38',
         entry_point='widowx_env.envs.v38_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_reward10:WidowxEnv',
         max_episode_steps=100
         )
         
# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + target initialised closed to tip
register(id='widowx_reacher-v39',
         entry_point='widowx_env.envs.v39_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_closeTarget:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + target initialised on the tip
register(id='widowx_reacher-v40',
         entry_point='widowx_env.envs.v40_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_onTarget:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + reward shape: equation 16
register(id='widowx_reacher-v41',
         entry_point='widowx_env.envs.v41_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_reward16:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + reward shape: equation 16 + beta = zero
register(id='widowx_reacher-v42',
         entry_point='widowx_env.envs.v42_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_reward16_beta0:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + reward shape: equation 1 (from env 42: don't work!)
register(id='widowx_reacher-v43',
         entry_point='widowx_env.envs.v43_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_reward1:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + actionStepCoeff = 25 + no bounds + 1 joint moving only + reward shape: equation 17 (from env 35)
register(id='widowx_reacher-v44',
         entry_point='widowx_env.envs.v44_widowx_pybullet_env_fixedGoal_actionStepCoeff25_noBounds_1joint_reward17:WidowxEnv',
         max_episode_steps=100
         )

# environment for the pybullet simulation + fixed goal + cleaned software Impact env
register(id='widowx_reacher-v45',
         entry_point='widowx_env.envs.v45_widowx_pybullet_fixed_gymEnv_softwareImpact:WidowxEnv',
         max_episode_steps=100
         )



# test gym.GoalEnv
register(id='my_goal_env-v0',
         entry_point='widowx_env.envs.my_goal_env:MyGoalEnv',
         max_episode_steps=100
         )
