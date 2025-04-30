#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=Unet-Models-Lightning-2GPU-10EPOCHS
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-8%1

/home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main.py results 10
