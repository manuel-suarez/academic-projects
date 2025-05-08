#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=MakeBinaryMasks
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-19

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python convert_png_to_binary.py
