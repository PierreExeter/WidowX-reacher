from gym.envs.registration import register

register(id='widowx_reach-v2',
         entry_point='widowx_physical.envs.widowx_env:WidowxEnv',
         )
         
