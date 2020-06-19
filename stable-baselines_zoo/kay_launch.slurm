#!/bin/sh
# Slurm flags
#SBATCH -p ProdQ
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH -A ucd01
#SBATCH -o submission_log/output.txt
# Mail me on job start & end
#SBATCH --mail-user=pierre.aumjaud@ucd.ie
#SBATCH --mail-type=BEGIN,END  

cd $SLURM_SUBMIT_DIR

# Cuda support for MPI
# OMPI_MCA_opal_cuda_support=true  

# Load software           
# module load openmpi/gcc/4.0.2rc2-ucx-cuda
# module load cuda/10.0
module load conda/2
source activate SB_widowx

# Run code
date
#time ./4_optimise_hyperparameters.sh
time ./5_run_experiments.sh
# time ./6_get_results.sh
date
