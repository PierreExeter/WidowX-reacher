#!/bin/sh
# Slurm flags
#SBATCH -p GpuQ
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -A ucd01
#SBATCH -o submission_log/output.txt
# Mail me on job start & end
#SBATCH --mail-user=pierre.aumjaud@ucd.ie
#SBATCH --mail-type=BEGIN,END  

cd $SLURM_SUBMIT_DIR

# Cuda support for MPI
OMPI_MCA_opal_cuda_support=true  

# Load software
module load openmpi/gcc/4.0.2rc2-ucx-cuda
module load conda/2
source activate rlkit
module load cuda/10.0

# Run code
date
#time python train_scripts/td3.py &> submission_log/log_td3.run
#time python train_scripts/her_td3_gym_fetch_reach.py &> submission_log/log_her_td3.run
#time python train_scripts/ddpg.py &> submission_log/log_ddpg.run
# time python tests/6_test_tensorflow_gpu.py &> submission_log/test_tensorflow.run
# time python tests/4_test_pytorch.py &> submission_log/test_pytorch.run
date
