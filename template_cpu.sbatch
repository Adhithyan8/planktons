#!/bin/env bash
#SBATCH -A naiss2024-5-84
#SBATCH --gpus-per-node=0
#SBATCH -C NOGPU -n 8
#SBATCH -t 0-03:00:00
#SBATCH -o %J_slurm.out
#SBATCH -e %J_slurm.err

module purge
module load matplotlib/3.5.2-foss-2022a
module load scikit-learn/1.1.2-foss-2022a
module load SciPy-bundle/2022.05-foss-2022a
source my_python/bin/activate
python classifier_analysis.py