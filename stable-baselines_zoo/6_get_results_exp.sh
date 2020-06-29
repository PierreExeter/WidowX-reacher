#!/bin/bash

# source env/bin/activate



nsteps=10000     # each episode last 100 timesteps, so evaluating for 2000 timeteps = 20 episodes
nb_seeds=10
opti_dir="logs/opti100t_0.1M_widowx_reacher-v7_SONIC/"
log_dir="logs/train_0.2M_widowx_reacher-v5_SONIC/"
# log_dir2="logs/train_0.5M_widowx_reacher-v7_KAY/"
save_dir="experiment_reports/train_0.2M_widowx_reacher-v5_SONIC/"
# save_dir2="experiment_reports/comp_0.5M_widowx_reacher-v5-v7_KAY/"
env="widowx_reacher-v5"
env_her="widowx_reacher-v6"
appendix="_env1"
random_log_folder="logs/random_policy_0.2M/widowx_reacher-v5/"
echo "ENV: ${env}"

# STEP 1
# for each seed experiment, evaluate and calculate mean reward (+std), train walltime, success ratio and average reach time
# + plot


# for ((i=1;i<${nb_seeds}+1;i+=1))
# do
#     echo "A2C $i"
#     python3 3_enjoy.py --algo a2c --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
#     python3 plot_1seed.py -f ${log_dir}a2c/${env}_$i/

#     echo "ACKTR $i"
#     python3 3_enjoy.py --algo acktr --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
#     python3 plot_1seed.py -f ${log_dir}acktr/${env}_$i/

#     echo "DDPG $i"
#     python3 3_enjoy.py --algo ddpg --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
#     python3 plot_1seed.py -f ${log_dir}ddpg/${env}_$i/

#     echo "PPO2 $i"
#     python3 3_enjoy.py --algo ppo2 --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
#     python3 plot_1seed.py -f ${log_dir}ppo2/${env}_$i/

#     echo "SAC $i"
#     python3 3_enjoy.py --algo sac --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
#     python3 plot_1seed.py -f ${log_dir}sac/${env}_$i/

#     echo "TD3 $i"
#     python3 3_enjoy.py --algo td3 --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
#     python3 plot_1seed.py -f ${log_dir}/td3/${env}_$i/

#     echo "TRPO $i"
#     python3 3_enjoy.py --algo trpo --env ${env} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
#     python3 plot_1seed.py -f ${log_dir}trpo/${env}_$i/

#     echo "HER $i"
#     python3 3_enjoy.py --algo her --env ${env_her} -f ${log_dir} --exp-id $i --no-render -n ${nsteps}
#     python3 plot_1seed.py -f ${log_dir}her/${env_her}_$i/

# done

# evaluate random policy
# python3 3_enjoy.py --random-pol True --env ${env} -f ${log_dir} --exp-id -1 --no-render -n ${nsteps}  # if random-pol = True, it doesn't matter to specify -f ${log_dir}
# python clean_random_training.py -f ${random_log_folder}

# record video
# python3 -m utils.record_video --algo td3 --env ${env} -n 400 -f ${log_dir}td3/${env}_1/



# STEP 2: Get the mean of the reward and wall train time of all the seed runs in the experiment

# python3 plot_experiment.py -f ${log_dir}a2c/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}acktr/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}ddpg/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}ppo2/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}sac/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}td3/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}trpo/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}her/ --env ${env_her}


# # STEP 3: Plot learning curves and training stats
# python3 plot_experiment_comparison.py -f ${log_dir} -s ${save_dir} -e ${appendix} -r ${random_log_folder}
 
## STEP 4: compare learning curves between 2 envs
#python3 plot_comp_envs_learning_curves.py -f1 ${log_dir} -f2 ${log_dir2} -s ${save_dir2}

# # IF OPTIMISATION
# # python3 plot_opti_report.py
# python3 plot_opti_best.py
python clean_opti_params.py -f ${opti_dir}a2c/
python clean_opti_params.py -f ${opti_dir}acktr/
python clean_opti_params.py -f ${opti_dir}ddpg/
python clean_opti_params.py -f ${opti_dir}ppo2/
python clean_opti_params.py -f ${opti_dir}sac/
python clean_opti_params.py -f ${opti_dir}td3/
python clean_opti_params.py -f ${opti_dir}trpo/
python clean_opti_params.py -f ${opti_dir}her/


# STEP 4: view trained agent
# python3 3_enjoy.py --algo a2c --env ${env} -f ${log_dir} --exp-id 1 -n ${nsteps} --render-pybullet True
# python3 3_enjoy.py --algo her --env ${env_her} -f ${log_dir} --exp-id 1 -n ${nsteps} --render-pybullet True
