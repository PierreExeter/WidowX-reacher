from gym.envs.registration import register

register(id='widowx_reach-v4',
         entry_point='widowx_pybullet_no_start_sim_goal_oriented.envs.widowx_env:WidowxEnv',
         max_episode_steps=100
         )
         
