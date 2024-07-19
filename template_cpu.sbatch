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
srun python prototypes_save.py --name semicosineWHOIvit --out-dim 110
srun python prototypes_check.py --name semicosineWHOIvit --data whoi_plankton
srun python prototypes_save.py --name semicosineCUBvit --out-dim 230
srun python prototypes_check.py --name semicosineCUBvit --data cub