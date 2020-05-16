"""
This is an example training script with the DDPG algorithm for fixed goal reaching.
By default, the gym environment will randomly sample a goal to use for the rest of training.

All modifiable parameters are in this script, including the sizes of the Q-networks, number
of epochs, discount factor, etc. 
"""
import gym
import gym_replab
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.ddpg.ddpg import DDPG
import rlkit.torch.pytorch_util as ptu


def experiment(variant):
    env = gym.make('replab-v0')._start_rospy(goal_oriented=False)
    #SIM
    #env = gym.make('replab-v0')._start_sim(goal_oriented=False, render=False)
    env = NormalizedBoxEnv(env)
    es = GaussianStrategy(action_space=env.action_space)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=50,
            num_steps_per_epoch=5000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=100,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
    )
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    setup_logger('DDPG_Experiment', variant=variant)
    experiment(variant)
