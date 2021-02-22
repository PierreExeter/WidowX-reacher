import gym
import pybullet as p
import pybullet_data
import os
import numpy as np
from gym import spaces


# Initial joint angles
RESET_VALUES = [
    0.015339807878856412, 
    -1.2931458041875956,
    1.0109710760673565, 
    -1.3537670644267164, 
    -0.07158577010132992, 
    .027]

# End-effector boundaries
BOUNDS_XMIN = -100
BOUNDS_XMAX = 100
BOUNDS_YMIN = -100
BOUNDS_YMAX = 100
BOUNDS_ZMIN = -100
BOUNDS_ZMAX = 100

# Joint boundaries
JOINT_MIN = np.array([
    -3.1,
    -1.571,
    -1.571,
    -1.745,
    -2.617,
    0.003
])

JOINT_MAX = np.array([
    3.1,
    1.571,
    1.571,
    1.745,
    2.617,
    0.03
])


class WidowxEnv(gym.Env):


    def __init__(self):
        """
        Initialise the environment
        """

        self.goal_oriented = False

        # Define action space
        self.action_space = spaces.Box(
            low=np.float32(np.array([-0.5, -0.25, -0.25, -0.25, -0.5, -0.005]) / 30),
            high=np.float32(np.array([0.5, 0.25, 0.25, 0.25, 0.5, 0.005]) / 30), 
            dtype=np.float32)

        # Define observation space
        self.obs_space_low = np.float32(np.array([-.16, -.15, 0.14, -3.1, -1.6, -1.6, -1.8, -3.1, 0]))
        self.obs_space_high = np.float32(np.array([.16, .15, .41, 3.1, 1.6, 1.6, 1.8, 3.1, 0.05]))

        self.observation_space = spaces.Box(
            low=np.float32(self.obs_space_low), 
            high=np.float32(self.obs_space_high), 
            dtype=np.float32)

        if self.goal_oriented:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=np.float32(np.array([-.16, -.15, 0.25])), high=np.float32(np.array([.16, .15, 0.41])), dtype=np.float32),
                achieved_goal=spaces.Box(low=np.float32(self.obs_space_low[:3]), high=np.float32(self.obs_space_high[:3]), dtype=np.float32),
                observation=self.observation_space
            ))

        self.current_pos = None
        
        # Initialise the goal position
        self.goal = np.array([.14, .0, 0.26])  # Fixed goal
        # self.set_goal(self.sample_goal_for_rollout())  # Random goal
        
        # Connect to physics client. By default, do not render
        self.physics_client = p.connect(p.DIRECT)

        # Load URDFs
        self.create_world()


    def create_world(self):

        # Initialise camera angle
        p.resetDebugVisualizerCamera(
            cameraDistance=0.6, 
            cameraYaw=0, 
            cameraPitch=-30, 
            cameraTargetPosition=[0.2, 0, 0.1], 
            physicsClientId=self.physics_client)

        # Load robot, sphere and plane urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        path = os.path.abspath(os.path.dirname(__file__))
        self.arm = p.loadURDF(os.path.join(path, "URDFs/widowx/widowx.urdf"), useFixedBase=True)
        self.sphere = p.loadURDF(os.path.join(path, "URDFs/sphere.urdf"), useFixedBase=True)
        self.plane = p.loadURDF('plane.urdf')

        # reset environment
        self.reset()

        
    def sample_goal_for_rollout(self):
        """ Sample random goal coordinates """
        return np.random.uniform(low=np.array([-.14, -.13, 0.26]), high=np.array([.14, .13, .39]))


    def set_goal(self, goal):
        self.goal = goal


    def step(self, action):
        """
        Execute the action.

        Parameters
        ----------
        action : array holding the angles changes from the previous time step [δ1, δ2, δ3, δ4, δ5, δ6]

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
                Either  [xe, ye, ze, θ1, θ2, θ3, θ4, θ5, θ6] for a Gym env
                or an observation dict for a goal env
            reward (float) :
                Negative, squared, l2 distance between current position and goal position
            episode_over (bool) :
                Whether or not we have reached the goal
            info (dict) :
                Additional information
        """
        self.action = np.array(action, dtype=np.float32)

        # Retrive current joint position and velocities 
        # (note that velocities are always 0 due to the force joint reset)
        self.joint_positions, self.joint_velocities = self._get_current_joint_positions()

        # Update the new joint position with the action
        self.new_joint_positions = self.joint_positions + self.action

        # Clip the joint position to fit the joint's allowed boundaries
        self.new_joint_positions = np.clip(np.array(self.new_joint_positions), JOINT_MIN, JOINT_MAX)

        # Instantaneously reset the joint position (no torque applied)
        self._force_joint_positions(self.new_joint_positions)

        # Retrieve the end effector position. 
        # If it's outside the boundaries defined, don't update the joint position
        end_effector_pos = self._get_current_end_effector_position()
        x, y, z = end_effector_pos[0], end_effector_pos[1], end_effector_pos[2]
        conditions = [
            x <= BOUNDS_XMAX,
            x >= BOUNDS_XMIN,
            y <= BOUNDS_YMAX,
            y >= BOUNDS_YMIN,
            z <= BOUNDS_ZMAX,
            z >= BOUNDS_ZMIN
        ]

        violated_boundary = False
        for condition in conditions:
            if not condition:
                violated_boundary = True
                break
        if violated_boundary:
            self._force_joint_positions(self.joint_positions)

        # Backup old position and get current joint position and current end effector position
        self.old_pos = self.current_pos  
        self.current_pos = self._get_current_state()

        return self._generate_step_tuple()


    def _generate_step_tuple(self):
        """ return (obs, reward, episode_over, info) tuple """

        # Reward
        reward = self._get_reward(self.goal)

        # Info
        self.old_distance = np.linalg.norm(self.old_pos[:3] - self.goal) 
        self.new_distance = np.linalg.norm(self.current_pos[:3] - self.goal) 

        info = {}
        info['total_distance'] = self.new_distance
        info['old_distance'] = self.old_distance
        info['goal_position'] = self.goal
        info['tip_position'] = self.current_pos[:3]
        info['old_joint_pos'] = self.joint_positions
        info['new_joint_pos'] = self.new_joint_positions
        info['joint_vel'] = self.joint_velocities

        # Never end episode prematurily
        episode_over = False
        # if self.new_distance < 0.0005:
        #     episode_over = True

        if self.goal_oriented:
            obs = self._get_obs()
            return obs, reward, episode_over, info

        return self.current_pos, reward, episode_over, info


    def reset(self):
        """ 
        Reset robot and goal at the beginning of an episode 
        Return observation
        """

        # Reset robot at the origin and move sphere to the goal position
        p.resetBasePositionAndOrientation(self.arm, [0, 0, 0], p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        p.resetBasePositionAndOrientation(self.sphere, self.goal, p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))

        # Reset joint at initial angles and get current state
        self._force_joint_positions(RESET_VALUES)
        self.current_pos = self._get_current_state()

        if self.goal_oriented:
            return self._get_obs()

        return self.current_pos


    def _get_obs(self):
        """ return goal_oriented observation """
        obs = {}
        obs['observation'] = self.current_pos
        obs['desired_goal'] = self.goal
        obs['achieved_goal'] = self.current_pos[:3]
        return obs


    def _get_reward(self, goal):
        """ Calculate the reward as - distance **2 """
        return - (np.linalg.norm(self.current_pos[:3] - goal) ** 2)


    def render(self, mode='human'):
        """ Render Pybullet simulation """
        p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI)
        self.create_world()


    def compute_reward(self, achieved_goal, goal, info):
        """ Function necessary for goal Env"""
        return - (np.linalg.norm(achieved_goal - goal)**2)


    def _get_current_joint_positions(self):
        """ Return current joint position and velocities """
        joint_positions = []
        joint_velocities = []

        for i in range(6):
            joint_positions.append(p.getJointState(self.arm, i)[0])
            joint_velocities.append(p.getJointState(self.arm, i)[1])

        return np.array(joint_positions, dtype=np.float32), np.array(joint_velocities, dtype=np.float32)


    def _get_current_end_effector_position(self):
        """ Get end effector coordinates """
        return np.array(list(p.getLinkState(self.arm, 5, computeForwardKinematics=1)[4]))


    def _set_joint_positions(self, joint_positions):
        """ Position control (not reset) """
        # In Pybullet, gripper halves are controlled separately
        joint_positions = list(joint_positions) + [joint_positions[-1]]
        p.setJointMotorControlArray(
            self.arm,
            [0, 1, 2, 3, 4, 7, 8],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions
        )


    def _force_joint_positions(self, joint_positions):
        """ Instantaneous reset of the joint angles (not position control) """
        for i in range(5):
            p.resetJointState(
                self.arm,
                i,
                joint_positions[i]
            )
        # In Pybullet, gripper halves are controlled separately
        for i in range(7, 9):
            p.resetJointState(
                self.arm,
                i,
                joint_positions[-1]
            )


    def _get_current_state(self):
        """ Return observation: end effector position + current joint position """
        return np.concatenate(
            [self._get_current_end_effector_position(),
             self._get_current_joint_positions()[0]],
            axis=0)
