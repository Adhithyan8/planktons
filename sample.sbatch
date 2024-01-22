#!/usr/bin/env bash
#SBATCH --mem 128GB
#SBATCH --gres gpu:0
#SBATCH --cpus-per-task 10
#SBATCH --time 24:00:00
#SBATCH --constrain "smaug"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user adhkal@kth.se
#SBATCH --output /Midgard/home/%u/planktons/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/planktons/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python pretrained.py