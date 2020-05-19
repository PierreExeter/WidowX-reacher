from gym.envs.registration import register

register(id='widowx_reach-v2',
         entry_point='widowx_physical.envs.widowx_env:WidowxEnv',
         max_episode_steps=100
         )
         
