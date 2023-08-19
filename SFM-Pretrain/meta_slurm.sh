#!/bin/bash
#SBATCH -J test
#SBATCH -p GPU-8A100
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -t 15:00

export PYTHONPATH=$PYTHONPATH:/gpfs/home/ess/hlsheng/.local/lib/python3.9/site-packages
nvidia-smi
./trainmeta.sh
