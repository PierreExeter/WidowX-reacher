#!/bin/bash

ntrials=10                            
nsteps=10000           
njobs=-1                 
sampler="tpe"
pruner="median"
log_dir="logs/opti10t_0.01M_widowx_reacher-v5_KAY/"
env="widowx_reacher-v5"
env_her="widowx_reacher-v6"


echo "A2C OPTI"
python3 2_train.py --algo a2c --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_a2c_opti.run

echo "ACKTR OPTI"
python3 2_train.py --algo acktr --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_acktr_opti.run

echo "DDPG OPTI"
python3 2_train.py --algo ddpg --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_ddpg_opti.run

echo "PPO2 OPTI"
python3 2_train.py --algo ppo2 --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_ppo2_opti.run

echo "SAC OPTI"
python3 2_train.py --algo sac --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_sac_opti.run

echo "TD3 OPTI"
python3 2_train.py --algo td3 --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_td3_opti.run

echo "TRPO OPTI"
python3 2_train.py --algo trpo --env ${env} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs 1 --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_trpo_opti.run

echo "HER OPTI"
# python3 2_train.py --algo her --env ${env_her} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_her_td3_opti.run
python3 2_train.py --algo her --env ${env_her} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_her_sac_opti.run
# python3 2_train.py --algo her --env ${env_her} -n ${nsteps} --log-folder ${log_dir} -optimize --n-trials ${ntrials} --n-jobs ${njobs} --sampler ${sampler} --pruner ${pruner}  &> submission_log/log_her_ddpg_opti.run


# python clean_opti_params.py --folder ${log_dir}
# python plot_opti_best.py
# python plot_opti_report.py
