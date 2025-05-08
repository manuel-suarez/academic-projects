#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=CountSegmentationPixels
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-19
#SBATCH --ntasks=500
#SBATCH --output=outputs/slurm-count_pixels-%A_%a.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python count_patches_pixels.py
