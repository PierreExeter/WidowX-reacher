from gym.envs.registration import register

register(id='widowx_reach-v1',
         entry_point='widowx_pybullet.envs.widowx_env:WidowxEnv',
         max_episode_steps=100,
         )
         
