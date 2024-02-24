#!/bin/env bash
#SBATCH -A naiss2023-5-75
#SBATCH --gpus-per-node=0
#SBATCH -C NOGPU -n 32
#SBATCH -t 0-03:00:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load scikit-learn/1.1.2-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source my_python/bin/activate
for M in resnet18 resnet50 vitb14-dinov2; do python tsne_knn.py -m $M; done