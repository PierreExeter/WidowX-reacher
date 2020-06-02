from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.core.eval_util import get_generic_path_information     # added by Pierre
import argparse
import joblib
import uuid
from rlkit.core import logger

# pierre
import gym, widowx_env
from rlkit.envs.wrappers import NormalizedBoxEnv

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['policy']

    # pierre
    # env = data['env']
    # env = gym.make('widowx_reach-v1')._start_sim(goal_oriented=False, render_bool=True)
    env = gym.make('widowx_reacher-v5')
    env.action_space.low *= 10
    env.action_space.high *= 10
    env = NormalizedBoxEnv(env)

    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        print("dump to logger: don't work")
        
        for k, v in get_generic_path_information("log_eval").items():
            logger.record_tabular(k, v)

        logger.dump_tabular()
        print("mean reward: ", path['rewards'].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
