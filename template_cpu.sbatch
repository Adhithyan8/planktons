#!/bin/env bash
#SBATCH -A naiss2024-5-84
#SBATCH --gpus-per-node=0
#SBATCH -C NOGPU -n 8
#SBATCH -t 0-03:00:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load matplotlib/3.5.2-foss-2022a
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
source my_python/bin/activate
python test.py