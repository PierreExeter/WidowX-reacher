import gym
import widowx_pybullet
import pybullet as p
import numpy as np

# not working well yet

env = gym.make('widowx_reach-v1')._start_sim(goal_oriented=False, render_bool=True)
obs = env.reset()

keys_actions = {
    p.B3G_LEFT_ARROW: np.array([-0.01, 0, 0]), 
    p.B3G_RIGHT_ARROW: np.array([0.01, 0, 0]), 
    p.B3G_UP_ARROW: np.array([0, 0, 0.01]), 
    p.B3G_DOWN_ARROW: np.array([0, 0, -0.01]),
    p.B3G_ALT: np.array([0, 0.01, 0]), 
    p.B3G_SHIFT: np.array([0, -0.01, 0])
    }

# Get the position and orientation of the end effector
position, orientation = p.getLinkState(env.arm, 5, computeForwardKinematics=True)[4:6]


while True:

    keys = p.getKeyboardEvents()
    for key, action in keys_actions.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            position += action

    # IK to get new joint positions (angles) for the robot
    target_joint_positions = p.calculateInverseKinematics(env.arm, 5, position, orientation)
    target_joint_positions = target_joint_positions[:6]

    # Get the joint positions (angles) of the robot arm
    joint_positions = env._get_current_joint_positions()

    # Set joint action to be the error between current and target joint positions
    joint_action = (target_joint_positions - joint_positions) / 10000

    # print("YOPO")
    # print(target_joint_positions)
    # print(joint_positions)
    # print(joint_action)
    obs, reward, done, info = env.step(joint_action)

