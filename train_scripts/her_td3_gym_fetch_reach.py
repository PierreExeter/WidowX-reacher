"""
This is an example training script with the HerTD3 algorithm for random goal reaching.
By default, the gym environment will randomly sample a goal upon each rollout.

All modifiable parameters are in this script, including the sizes of the Q-networks, number
of epochs, discount factor, etc. 
"""
import gym
import widowx_pybullet
from rlkit.envs.wrappers import NormalizedBoxEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import (
    GaussianAndEpislonStrategy
)
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.her.her import HerTd3
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy


def experiment(variant):
    # env = gym.make('replab-v0')._start_rospy(goal_oriented=True)
    #SIM
    # env = gym.make('replab-v0')._start_sim(goal_oriented=True, render=False)
    env = gym.make('widowx_reach-v1')._start_sim(goal_oriented=True, render_bool=False)
    env = NormalizedBoxEnv(env)
    es = GaussianAndEpislonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerTd3(
        her_kwargs=dict(
            observation_key='observation',
            desired_goal_key='desired_goal'
        ),
        td3_kwargs = dict(
            env=env,
            qf1=qf1,
            qf2=qf2,
            policy=policy,
            exploration_policy=exploration_policy
        ),
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=3000, #100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=50,
            batch_size=128,
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.25,
        ),
    )
    ptu.set_gpu_mode(False)
    setup_logger('Her_TD3_Fetch_Experiment', variant=variant)
    experiment(variant)
