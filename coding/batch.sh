#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=a3
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodelist=linse18
 
# Activate everything you need
module load cuda/11.2
pyenv activate venv
# Run your python code
python3 a4_label_matching.py
