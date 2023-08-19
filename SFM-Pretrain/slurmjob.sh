#!/bin/bash
#SBATCH -J test
#SBATCH -p GPU-8A100
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -t 15:00
#SBATCH --qos=gpu_8a100
./train.sh
