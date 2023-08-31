#!/bin/bash
#SBATCH -J SFM
#SBATCH -p GPU-8A100
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH -t 30:00
#SBATCH --qos=gpu_8a100
./train.sh
