from gym.envs.registration import register

register(id='widowx_reach-v5',
         entry_point='widowx_physical_no_start_ros.envs.widowx_env:WidowxEnv',
         max_episode_steps=100
         )
         
