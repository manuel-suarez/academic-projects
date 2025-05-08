#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=MakeAugmentationPatches
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-19%1
#SBATCH --ntasks=64
#SBATCH --output=outputs/slurm-augmentation_patches-%A_%a.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python build_augmentation_patches.py
