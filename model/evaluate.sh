#!/bin/bash -l
#SBATCH --chdir /scratch/izar/saikali/mnlpproj4/model
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

python depth_evaluator.py --eval_method reward --file "../datasets/EPFL/processed/EPFL_DPO_val.jsonl" --model "../training/training_31_05_Anthony/models/EPFL_DPO/final"
python depth_evaluator.py --eval_method reward --file "../datasets/helpSteer/processed/helpsteer_val.jsonl" --model "../training/training_31_05_Anthony/models/EPFL_DPO/final"
python depth_evaluator.py --eval_method mcqa --file "../datasets/MathQA/processed/mathQA_val.jsonl" --model "../training/training_31_05_Anthony/models/EPFL_DPO/final"
python depth_evaluator.py --eval_method mcqa --file "../datasets/openQA/processed/openQA_val.jsonl" --model "../training/training_31_05_Anthony/models/EPFL_DPO/final"
python depth_evaluator.py --eval_method mcqa --file "../datasets/MMLU/processed/MMLU_val.jsonl" --model "../training/training_31_05_Anthony/models/EPFL_DPO/final"