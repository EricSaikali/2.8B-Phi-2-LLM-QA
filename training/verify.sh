#!/bin/bash -l
#SBATCH --chdir /scratch/izar/saikali/<proj-dir>/training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-gpu 32G
#SBATCH --time 24:00:00
#SBATCH --gres gpu:1
#SBATCH --account <specific-account>
#SBATCH --qos <specific-account>
#SBATCH --reservation <specific-account>

# `SBATCH --something` is how you tell SLURM what resources you need
# The --reservation line only works during the 2-week period
# where 80 GPUs are available. Remove otherwise

module load gcc
module load python

source ../../shaikespear_train/bin/activate

python verify_config.py
