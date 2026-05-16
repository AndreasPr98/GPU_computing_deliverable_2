#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --time=00:4:30
#SBATCH --gres=gpu:0
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --job-name=matrix_analysis
#SBATCH --output=matrix_analysis/R-%x.%j.out
#SBATCH --error=matrix_analysis/R-%x.%j.err

module load python/3.10

python3 matrix_analysis.py
