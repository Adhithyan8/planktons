#!/bin/env bash
#SBATCH -A naiss2023-5-75
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-00:30:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-learn/1.1.2-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source my_python/bin/activate
python pretrained_inference.py