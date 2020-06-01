import gym
from gym import error, spaces, utils
from gym.utils import seeding
from numbers import Number
from collections import OrderedDict
import pybullet as p
import pybullet_data
import os
import numpy as np
import random


# ENVIRONMENT CONFIGURATION
NEUTRAL_VALUES = [0.015339807878856412, -1.4839419194602816,
                  1.4971652489763858, -0.008369006790373335, -0.08692557798018634, .027]
RESET_VALUES = [0.015339807878856412, -1.2931458041875956,
                1.0109710760673565, -1.3537670644267164, -0.07158577010132992, .027]

# RL BOUNDS
BOUNDS_FLOOR = .41
BOUNDS_LEFTWALL = .14
BOUNDS_RIGHTWALL = -.14
BOUNDS_FRONTWALL = -.13
BOUNDS_BACKWALL = .13

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
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3',
               'joint_4', 'joint_5', 'gripper_joint']
SIM_START_POSITION = np.array([-0.185033226409, 0.00128528, 0.46227163])



class WidowxEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        How to initialize this environment:
        env = gym.make('replab-v0').start_rospy(goal_oriented=[GOAL_ORIENTED])
        If goal_oriented is true, then the environment's observations become a dict
        and the goal is randomly resampled upon every reset

        params:
        goal_oriented: Changes some aspects of the environment for goal-oriented tasks

        rospy.init_node is set with random number so we can have multiple
        nodes running at the same time.

        self.goal is set to a fixed, randomly drawn goal if goal_oriented = False
        """
        self.obs_space_low = np.array(
            [-.16, -.15, 0.14, -3.1, -1.6, -1.6, -1.8, -3.1, 0])
        self.obs_space_high = np.array(
            [.16, .15, .41, 3.1, 1.6, 1.6, 1.8, 3.1, 0.05])
        observation_space = spaces.Box(
            low=self.obs_space_low, high=self.obs_space_high, dtype=np.float32)
        self.observation_space = observation_space
        self.action_space = spaces.Box(low=np.array([-0.5, -0.25, -0.25, -0.25, -0.5, -0.005]) / 10,
                                       high=np.array([0.5, 0.25, 0.25, 0.25, 0.5, 0.005]) / 10, dtype=np.float32)
        self.current_pos = None
        #self.goal = np.array([-.14, -.13, 0.26])
        self.set_goal(self.sample_goal_for_rollout())
        print("********goal is : ***********", self.goal)

    #shared functions between both sim and robot mode   
    
    def sample_goal_for_rollout(self):
        return np.random.uniform(low=np.array([-.14, -.13, 0.26]), high=np.array([.14, .13, .39]))

    def set_goal(self, goal):
        self.goal = goal

    def step(self, action):
        """

        Parameters
        ----------
        action : [change in x, change in y, change in z]

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                either current position or an observation object, depending on
                the type of environment this is representing
            reward (float) :
                negative, squared, l2 distance between current position and 
                goal position
            episode_over (bool) :
                Whether or not we have reached the goal
            info (dict) :
                 For now, all this does is keep track of the total distance from goal.
                 This is used for rlkit to get the final total distance after evaluation.
                 See function get_diagnostics for more info.
        """
        action = np.array(action, dtype=np.float32)

        joint_positions = self._get_current_joint_positions()
        new_joint_positions = joint_positions + action
        new_joint_positions = np.clip(np.array(new_joint_positions), JOINT_MIN, JOINT_MAX)
        self._force_joint_positions(new_joint_positions)
        
        end_effector_pos = self._get_current_end_effector_position()
        x, y, z = end_effector_pos[0], end_effector_pos[1], end_effector_pos[2]
        conditions = [
            x <= BOUNDS_LEFTWALL,
            x >= BOUNDS_RIGHTWALL,
            y <= BOUNDS_BACKWALL,
            y >= BOUNDS_FRONTWALL,
            z <= BOUNDS_FLOOR,
            z >= 0.15
        ]
        violated_boundary = False
        for condition in conditions:
            if not condition:
                violated_boundary = True
                break
        if violated_boundary:
            self._force_joint_positions(joint_positions)
        self.current_pos = self._get_current_state()

        return self._generate_step_tuple()

    def _generate_step_tuple(self):
        reward = self._get_reward(self.goal)

        episode_over = False
        total_distance_from_goal = np.sqrt(-reward)

        info = {}
        info['total_distance'] = total_distance_from_goal
        info['goal position'] = self.goal              # added by Pierre
        info['tip position'] = self.current_pos[:3]    # added by Pierre
        info['joint position'] = self.current_pos[3:]  # added by Pierre

        if reward > -0.0001:
            episode_over = True

        if self.goal_oriented:
            obs = self._get_obs()
            return obs, reward, episode_over, info

        return self.current_pos, reward, episode_over, info

    def reset(self):

        p.resetBasePositionAndOrientation(self.arm, [0, 0, 0], p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        p.resetBasePositionAndOrientation(self.sphere, self.goal, p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))         # added by Pierre: move sphere to self.goal position
        self._force_joint_positions(RESET_VALUES)
        self.current_pos = self._get_current_state()

        if self.goal_oriented:
            self.set_goal(self.sample_goal_for_rollout())
            return self._get_obs()

        return self.current_pos

    def _get_obs(self):
        obs = {}
        obs['observation'] = self.current_pos
        obs['desired_goal'] = self.goal
        obs['achieved_goal'] = self.current_pos[:3]
        return obs

    def sample_goals(self, num_goals):
        sampled_goals = np.array(
            [self.sample_goal_for_rollout() for i in range(num_goals)])
        goals = {}
        goals['desired_goal'] = sampled_goals
        return goals

    def _get_reward(self, goal):
        return - (np.linalg.norm(self.current_pos[:3] - goal) ** 2)

    def render(self, mode='human', close=False):
        pass

    def compute_reward(self, achieved_goal, goal, info):
        return - (np.linalg.norm(achieved_goal - goal)**2)

    def get_diagnostics(self, paths):
        """
        This adds the diagnostic "Final Total Distance" for RLkit
        """
        def get_stat_in_paths(paths, dict_name, scalar_name):
            if len(paths) == 0:
                return np.array([[]])

            if type(paths[0][dict_name]) == dict:
                return [path[dict_name][scalar_name] for path in paths]
            return [[info[scalar_name] for info in path[dict_name]] for path in paths]

        def create_stats_ordered_dict(
                name,
                data,
                stat_prefix=None,
                always_show_all_stats=True,
                exclude_max_min=False,
        ):
            if stat_prefix is not None:
                name = "{} {}".format(stat_prefix, name)
            if isinstance(data, Number):
                return OrderedDict({name: data})

            if len(data) == 0:
                return OrderedDict()

            if isinstance(data, tuple):
                ordered_dict = OrderedDict()
                for number, d in enumerate(data):
                    sub_dict = create_stats_ordered_dict(
                        "{0}_{1}".format(name, number),
                        d,
                    )
                    ordered_dict.update(sub_dict)
                return ordered_dict

            if isinstance(data, list):
                try:
                    iter(data[0])
                except TypeError:
                    pass
                else:
                    data = np.concatenate(data)

            if (isinstance(data, np.ndarray) and data.size == 1
                    and not always_show_all_stats):
                return OrderedDict({name: float(data)})

            stats = OrderedDict([
                (name + ' Mean', np.mean(data)),
                (name + ' Std', np.std(data)),
            ])
            if not exclude_max_min:
                stats[name + ' Max'] = np.max(data)
                stats[name + ' Min'] = np.min(data)
            return stats
        statistics = OrderedDict()
        stat_name = 'total_distance'
        stat = get_stat_in_paths(paths, 'env_infos', stat_name)
        statistics.update(create_stats_ordered_dict('Final %s' % (stat_name), [
                          s[-1] for s in stat], always_show_all_stats=True,))
        return statistics

    #Functions only for sim mode
    def _get_current_joint_positions(self):
        joint_positions = []
        for i in range(6):
            joint_positions.append(p.getJointState(self.arm, i)[0])
        return np.array(joint_positions, dtype=np.float32)
        
    def _get_current_end_effector_position(self):
        real_position = np.array(list(p.getLinkState(self.arm, 5, computeForwardKinematics=1)[4]))
        #real_position[2] = -real_position[2] #SIM z coordinates are reversed
        #adjusted_position = real_position + SIM_START_POSITION
        return real_position

    def _set_joint_positions(self, joint_positions):
        # In SIM, gripper halves are controlled separately
        joint_positions = list(joint_positions) + [joint_positions[-1]]
        p.setJointMotorControlArray(
            self.arm,
            [0, 1, 2, 3, 4, 7, 8],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions
        )

    def _force_joint_positions(self, joint_positions):
        for i in range(5):
            p.resetJointState(
                self.arm,
                i,
                joint_positions[i]
            )
        for i in range(7, 9):
            p.resetJointState(
                self.arm,
                i,
                joint_positions[-1]
            )

    def _get_current_state(self):
        return np.concatenate(
                [self._get_current_end_effector_position(),
                self._get_current_joint_positions()],
                axis = 0)

    def start_sim(self, goal_oriented=False, render_bool=False):

        if render_bool:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        self.render_bool = render_bool
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.goal_oriented = goal_oriented
        if self.goal_oriented:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=np.array(
                    [-.16, -.15, 0.25]), high=np.array([.16, .15, 0.41]), dtype=np.float32),
                achieved_goal=spaces.Box(low=self.obs_space_low[
                    :3], high=self.obs_space_high[:3], dtype=np.float32),
                observation=self.observation_space
            ))
        #p.resetSimulation()
        #p.setTimeStep(0.01)
        path = os.path.abspath(os.path.dirname(__file__))
        self.arm = p.loadURDF(os.path.join(path, "URDFs/widowx/widowx.urdf"), useFixedBase=True)
        self.sphere = p.loadURDF(os.path.join(path, "URDFs/sphere.urdf"), useFixedBase=True)      # added by Pierre
        self.plane = p.loadURDF('plane.urdf')   # added by Pierre
        self.reset()
        return self

    # Functions for pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if state['render_bool']:
            self.start_sim(goal_oriented=state['goal_oriented'], render_bool=False)
        else:
            self.start_sim(goal_oriented=state['goal_oriented'], render_bool=state['render_bool'])
        self.reset()
