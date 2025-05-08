#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=MakeSegmentationPatches
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-19%1
#SBATCH --ntasks=64
#SBATCH --output=outputs/slurm-segmentation_patches-%A_%a.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python build_segmentation_patches.py
