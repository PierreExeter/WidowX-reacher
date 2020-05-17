from gym.envs.registration import register

register(id='widowx_reach-v0',
         entry_point='widowx_original.envs.widowx_env:WidowxEnv',
         )
         
