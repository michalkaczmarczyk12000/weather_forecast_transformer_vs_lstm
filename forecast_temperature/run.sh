#!/bin/bash

# SLURM directives
#SBATCH -A plgzzsn2024-gpu-a100
#SBATCH -o slurm2_%a.log
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 360
#SBATCH --array 0-1
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 40G
#SBATCH --nodes 1
source activate /net/tscratch/people/plgmaciejstranz/cache/test-env
export HF_HOME=/net/tscratch/people/plgmaciejstranz/cache
srun --mpi=openmpi python3 main.py
