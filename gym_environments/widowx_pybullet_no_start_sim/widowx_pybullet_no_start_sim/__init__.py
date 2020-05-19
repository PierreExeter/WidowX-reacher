from gym.envs.registration import register

register(id='widowx_reach-v3',
         entry_point='widowx_pybullet_no_start_sim.envs.widowx_env:WidowxEnv',
         max_episode_steps=100
         )
         
