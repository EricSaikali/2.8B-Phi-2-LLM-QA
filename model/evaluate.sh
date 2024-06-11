#!/bin/bash -l
#SBATCH --chdir /scratch/izar/balykov/ShAIkespear_m2/model
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-gpu 32G
#SBATCH --time 10:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552

# `SBATCH --something` is how you tell SLURM what resources you need
# The --reservation line only works during the 2-week period
# where 80 GPUs are available. Remove otherwise

module load gcc
module load python

source ../../shaikespear/bin/activate

python depth_evaluator.py --eval_method reward --file "../evaluation_data/EPFL/EPFL_DPO_val.jsonl" --model "../training/EPFL_PLUS_10_06/models/EPFL_DPO/final"
python depth_evaluator.py --eval_method reward --file "../evaluation_data/helpSteer/helpsteer_val.jsonl" --model "../training/EPFL_PLUS_10_06/models/EPFL_DPO/final"
python depth_evaluator.py --eval_method mcqa --file "../evaluation_data/MathQA/mathQA_val.jsonl" --model "../training/EPFL_PLUS_10_06/models/EPFL_DPO/final"
python depth_evaluator.py --eval_method mcqa --file "../evaluation_data/openQA/openQA_val.jsonl" --model "../training/EPFL_PLUS_10_06/models/EPFL_DPO/final"