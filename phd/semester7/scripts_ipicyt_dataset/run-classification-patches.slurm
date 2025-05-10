#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=MakeClassificationPatches
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm-classification_patches-%A.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python build_classification_patches.py
