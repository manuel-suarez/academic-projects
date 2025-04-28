#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=CHN6_CUG-Figures
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm_figures-%A.out


srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python main-figures.py --model_name=$1 --encoder_name=$2 --epochs=$3
