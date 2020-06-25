#!/bin/bash

# experiments over 10 initialisation seeds

nsteps=500000    
log_dir="logs/train_0.5M_widowx_reacher-v7_KAY"
env="widowx_reacher-v7"
env_her="widowx_reacher-v8"


for ((i=0;i<2;i+=1))
do
    echo "A2C $i"
    python3 2_train.py --algo a2c --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_log/log_a2c_0$i.run

    echo "ACKTR $i"
    python3 2_train.py --algo acktr --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_log/log_acktr_0$i.run

    echo "DDPG $i"
    python3 2_train.py --algo ddpg --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_log/log_ddpg_0$i.run

    echo "PPO2 $i"
    python3 2_train.py --algo ppo2 --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_log/log_ppo2_0$i.run

    echo "SAC $i"
    python3 2_train.py --algo sac --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_log/log_sac_0$i.run

    echo "TD3 $i"
    python3 2_train.py --algo td3 --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_log/log_td3_0$i.run

    echo "TRPO $i"
    python3 2_train.py --algo trpo --env ${env} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_log/log_trpo_0$i.run

    echo "HER $i"
    python3 2_train.py --algo her --env ${env_her} -n ${nsteps} --seed $i --log-folder ${log_dir} &> submission_log/log_her_td3_0$i.run
done
