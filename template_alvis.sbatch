#!/bin/bash
#SBATCH --account=adhkal
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=01:00:00

module load pytorch
~/my_ml_pytorch_script.py --epochs=12 --mnist_src /mimer/NOBACKUP/Datasets/MNIST/raw