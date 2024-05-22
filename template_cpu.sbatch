#!/bin/env bash
#SBATCH -A naiss2024-5-84
#SBATCH --gpus-per-node=0
#SBATCH -C NOGPU -n 8
#SBATCH -t 0-01:00:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load scikit-learn/1.3.1-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
source myvenv/bin/activate
for k in 1 3 5 7 9
do
    python knn.py --name tsim_selfcauchyrnd_ph --metric euclidean --knn-neighbors $k
done