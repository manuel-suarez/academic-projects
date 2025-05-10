#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=MakeOverlays
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --ntasks=24
#SBATCH --output=outputs/slurm-%A.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python make_mask_overlays.py
