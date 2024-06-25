#!/bin/env bash
#SBATCH -A naiss2024-5-84
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 0-01:00:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load PyTorch-Lightning/2.2.1-foss-2023a-CUDA-12.1.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source myvenv/bin/activate
srun python mu_gcd_inference.py --name semicosineCUBvit --data cub --out-dim 230