# WidowX reacher
Training the WidowX robotic arm to reach a target position with reinforcement learning.
The Gym environments are adapted from [Replab](https://github.com/bhyang/replab).

![Alt text](/docs/images/widowx_pybullet.gif?raw=true "The Widowx Gym environment in Pybullet")

## Installation

1. Install [ROS](http://wiki.ros.org/ROS/Installation) (optional)

This is only required if training the physical arm.

2. Install and activate the Conda environment

```bash
# train with Rlkit
conda env create -f conda_envs/environment_rlkit.yml
conda activate rlkit
```
OR
```bash
# train with Stable Baselines
conda env create -f conda_envs/environment_sb_light.yml   # or conda_envs/environment_sb_kay.yml
conda activate SB_widowx
```

3. Install the custom Gym environments

```bash
# install widowx_reach-v0 (environment for both the physical arm and the Pybullet simulation)
cd gym_environments/widowx_original/
pip install -e .
cd ..
# install widowx_reach-v1 (environment for the Pybullet simulation only. ROS install not required) 
cd gym_environments/widowx_pybullet/
pip install -e .
cd ..
# install widowx_reach-v2 (environment for the physical arm only)
cd gym_environments/widowx_physical/
pip install -e .
cd ..
# install widowx_reach-v3 (environment for the Pybullet simulation + no start_sim required)
cd gym_environments/widowx_pybullet_no_start_sim/
pip install -e .
cd ..
# install widowx_reach-v4 (environment for the Pybullet simulation + no start_sim required + goal_oriented = True)
cd gym_environments/widowx_pybullet_no_start_sim_goal_oriented/
pip install -e .
cd ..
```

4. Install the local [Rlkit](https://github.com/vitchyr/rlkit) repository
```bash
cd rlkit
pip install -e .
cd ..
```

5. Install the local Viskit repository
```bash
cd viskit
pip install -e .
cd ..
```

## Test the Gym environment

```bash
python tests/0_test_widowx_pybullet.py
python tests/1_test_widowx_gym.py
```

## Train

```bash
python train_scripts/td3.py
```

## Plot training stats

```bash
python viskit/viskit/frontend.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_10_35_20000--s-0/
```

python viskit/viskit/frontend.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_17_22_42_12_0000--s-0/


## Evaluate a trained policy

```bash
python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_10_35_26_0000--s-0/params.pkl
```

python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_17_01_07_10_0000--s-0/params.pkl



## Visualise a trained policy

```bash
python enjoy_scripts/save_clean_pickle.py 
python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_15_29_53_0000--s-0/cleaned_params.pkl
python enjoy_scripts/simple_sim_policy.py
```

To log each timestep stats and plot the goal and tip position while evaluating the policy (slow):
```bash
python tests/3_test_episode_plotter_logger.py
```

![Alt text](/docs/images/widowx_plot2d.gif?raw=true "plot 2D")
![Alt text](/docs/images/widowx_plot3d.gif?raw=true "plot 3D")

# Tested on

- ROS Melodic
- Ubuntu 18.04
- python 3.5.2
- conda 4.8.0