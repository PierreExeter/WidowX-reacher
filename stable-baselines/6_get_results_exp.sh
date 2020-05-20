#!/bin/bash

# source env/bin/activate


# STEP 1
# for each seed experiment, evaluate and calculate mean reward (+std), train walltime, success ratio and average reach time
# + plot



nsteps=2000     # each episode last 100 timesteps, so evaluating for 2000 timeteps = 20 episodes
nb_seeds=2
log_dir="logs/train_1M_widowx_reach-v3/"
env="widowx_reach-v3"
echo "ENV: ${env}"



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

# done

# record video
# python3 -m utils.record_video --algo td3 --env ${env} -n 400 -f ${log_dir}td3/${env}_1/



# STEP 2
# Get the mean of the reward and wall train time of all the seed runs in the experiment
# plot all seeds runs

# python3 plot_experiment.py -f ${log_dir}a2c/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}acktr/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}ddpg/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}ppo2/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}sac/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}td3/ --env ${env}
# python3 plot_experiment.py -f ${log_dir}trpo/ --env ${env}


# # STEP 3
# # Plot learning curves and training stats
# python3 plot_experiment_comparison.py


# # IF OPTIMISATION
# # python3 plot_opti_report.py
# python3 plot_opti_best.py


# STEP 4: view trained agent

python3 3_enjoy.py --algo trpo --env ${env} -f ${log_dir} --exp-id 1 -n ${nsteps} --render-pybullet True