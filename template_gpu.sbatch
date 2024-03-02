#!/bin/env bash
#SBATCH -A naiss2024-5-84
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 0-24:00:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
source my_python/bin/activate
python contrastive_learning.py