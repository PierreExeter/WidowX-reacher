#!/bin/sh
# Slurm flags                                                                                                                                                                              
#SBATCH -p ProdQ                                                                                                                                                                                
#SBATCH -N 1                                                                                                                                                                                     
#SBATCH -t 20:00:00                                                                                                                                                                                 
#SBATCH -A ucd01                                                                                                                                                                                        
#SBATCH -o submission_log/output.txt                                                                                                                                                                                                          
# Mail me on job start & end                                                                                                                                                                            
#SBATCH --mail-user=pierre.aumjaud@ucd.ie                                                                                                                                    
#SBATCH --mail-type=BEGIN,END  

cd $SLURM_SUBMIT_DIR

# Load software                                                                                                                                                                                                                
module load conda/2
source activate rlkit
module load cuda/10.0

# Run code                                                                                                                                                                                         
date
time python train_scripts/td3.py &> submission_log/log_td3.run
time python train_scripts/her_td3_gym_fetch_reach.py &> submission_log/log_her_td3.run
time python train_scripts/ddpg.py &> submission_log/log_ddpg.run
#time python tests/test_pytorch.py &> submission_log/test_pytorch.run
date
