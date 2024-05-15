#!/bin/env bash
#SBATCH -A naiss2024-5-84
#SBATCH --gpus-per-node=0
#SBATCH -C NOGPU -n 32
#SBATCH -t 0-01:00:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load PyTorch-Lightning/2.2.1-foss-2023a-CUDA-12.1.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source my_venv/bin/activate
python test.py