#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=SumSegmentationPixels
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm-sum_pixels-%A.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python sum_patches_pixels.py
