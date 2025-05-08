#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=SplitSegmentationPatches
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm-split_patches-%A.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python split_segmentation_patches.py
