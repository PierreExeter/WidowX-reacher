#!/bin/bash -l

# Slurm flags
#SBATCH -N 1

# GPU QUEUE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH -t 72:00:00
#SBATCH -o submission_log/output.txt
#SBATCH --mail-user=pierre.aumjaud@ucd.ie
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

# Load software
module load anaconda/3.2020.2 
module load cuda/10.0 
module load openmpi/4.0.1
source activate SB_widowx


# Run code
date
#time ./4_optimise_hyperparameters.sh
time ./5_run_experiments.sh
# time ./6_get_results.sh
date

