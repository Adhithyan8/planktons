#!/bin/env bash
#SBATCH -A naiss2024-5-84
#SBATCH --gpus-per-node=0
#SBATCH -C NOGPU -n 8
#SBATCH -t 0-01:00:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load scikit-learn/1.1.2-foss-2022a
module load matplotlib/3.5.2-foss-2022a
source my_python/bin/activate
python tsne_knn.py -m selfcauchy_resnet18 -v