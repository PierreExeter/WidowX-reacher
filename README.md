# WidowX reacher
Training the WidowX robotic arm to reach a target position with reinforcement learning.
The Gym environments are adapted from [Replab](https://github.com/bhyang/replab).

![Alt text](/docs/images/widowx_pybullet.gif?raw=true "The Widowx Gym environment in Pybullet")

## Installation

1. Install [ROS](http://wiki.ros.org/ROS/Installation) (optional)


1.1. Install the ROS packages (optional)

```bash
mv widowx_ros_packages ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
```

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
cd gym_environments/
pip install -e .
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

6. Install the local Stable Baselines repository
```bash
cd stable-baselines
pip install -e .
cd ..
```

## Test the Gym environment

```bash
python tests/0_test_widowx_pybullet.py
python tests/1_test_widowx_gym.py
```

To log each timestep stats and plot the goal and tip position while evaluating the policy (slow):
```bash
python tests/3_test_episode_plotter_logger.py
```

![Alt text](/docs/images/widowx_plot2d.gif?raw=true "plot 2D")
![Alt text](/docs/images/widowx_plot3d.gif?raw=true "plot 3D")

# STABLE BASELINES


## Train

```bash
cd stable-baselines
./5_run_experiments.sh
```

## Evaluate policy and plot training stats

```bash
./6_get_results.sh
```

# RLKIT (deprecated)

## Train

```bash
python train_scripts/td3.py
```

## Plot training stats

```bash
python viskit/viskit/frontend.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_10_35_20000--s-0/
```

## Evaluate a trained policy

```bash
python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_10_35_26_0000--s-0/params.pkl
```

## Visualise a trained policy

```bash
python enjoy_scripts/save_clean_pickle.py 
python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_15_29_53_0000--s-0/cleaned_params.pkl
python enjoy_scripts/simple_sim_policy.py
```

# Control the physical arm

Add Docker pull

## Run Docker image

```bash
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --privileged pierre/widowx_rl:version2 
```


## Start roscore

In terminal 1

```bash
roslaunch widowx_arm_bringup arm_moveit.launch sim:=false sr300:=false
```

## Activate the motors

In terminal 2

```bash
docker container ls
docker exec -it [container ID] bash
rosrun replab_core controller.py
widowx.move_to_neutral()
# exit with CTRL+D
```

## Configure the servos torque (

Set the servos torque to 50% of their maximum value to prevent collision damage (to do only once, even after closing the container).

In terminal 2

```bash
rosrun replab_core compute_control_noise.py
```

## Start environment subscriber

In terminal 2

```bash
rosrun replab_rl replab_env_subscriber.py
```

## Test environment

In terminal 3

```bash
docker container ls
docker exec -it [container ID] bash
source activate rlkit  # source activate SB_widowx
cd /root/ros_ws/rl_scripts/WidowX-reacher/
python examples/test_physical_env.py
python examples/test_episode_plotted.py  
```

In case of CV2 error:
```bash
ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type
```

edit 

```bash
/miniconda/envs/rlkit/lib/python3.5/site-packages/stable_baselines/common/atari_wrappers.py
```

by adding this line:

```python
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    import cv2
    sys.path.append(ros_path)

#import cv2  # pytype:disable=import-error
cv2.ocl.setUseOpenCL(False)
```

## Train

In terminal 3

```bash
docker container ls
docker exec -it [container ID] bash
source activate rlkit
cd /root/ros_ws/rl_scripts/rlkit/
python examples/td3.py 
```

# Tested on

- ROS Melodic
- Ubuntu 18.04
- Python 3.5.2
- Conda 4.8.0