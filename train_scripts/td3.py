"""
This is an example training script with the TD3 algorithm for fixed goal reaching.
By default, the gym environment will randomly sample a goal to use for the rest of training.

All modifiable parameters are in this script, including the sizes of the Q-networks, number
of epochs, discount factor, etc. 
"""
import gym
import gym_replab
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3


def experiment(variant):
    #Robot 
    # env = gym.make('replab-v0')._start_rospy(goal_oriented=False)
    #SIM
    env = gym.make('replab-v0')._start_sim(goal_oriented=False, render_bool=False)
    env.action_space.low *= 10
    env.action_space.high *= 10
    env = NormalizedBoxEnv(env)
    es = GaussianStrategy(
        action_space=env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
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
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=50,
            num_steps_per_epoch=5000,
            num_steps_per_eval=1000,
            max_path_length=1000,
            batch_size=100,
            discount=0.99,
            replay_buffer_size=int(1E6),
        ),
    )
    ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)
    setup_logger('TD3_Experiment', variant=variant)
    experiment(variant)
