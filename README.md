# WidowX reacher
Training the WidowX robotic arm to reach a target position with reinforcement learning.
The Gym environments are adapted from [Replab](https://github.com/bhyang/replab).


![Alt text](/docs/images/widowx_pybullet.gif?raw=true "The Widowx Gym environment in Pybullet")


- widowx_reach-v0   # environment for the physical arm and the Pybullet simulation
- widowx_reach-v1   # environment only for the Pybullet simulation
- widowx_reach-v2   # environment only for the physical arm


## Installation


Install ROS
(tested on ROS Melodic and Ubuntu 18.04)

```bash
conda env create -f environment.yml
conda activate rlkit
```

```bash
cd gym_environments/widowx_original/
pip install -e .
cd ..
```

```bash
cd rlkit
pip install -e .
cd ..
```

```bash
cd viskit
pip install -e .
cd ..
```

## Test the Gym environment

```bash
python test_widowx_gym.py
```


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
python enjoy_scripts/save_clean_pickle.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_10_35_26_0000--s-0/params.pkl
python enjoy_scripts/sim_policy.py rlkit/data/TD3-Experiment/TD3_Experiment_2020_05_16_15_29_53_0000--s-0/cleaned_params.pkl
python enjoy_scripts/simple_sim_policy.py
```