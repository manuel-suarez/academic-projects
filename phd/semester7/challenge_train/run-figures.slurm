#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=ChallengeFigures
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-5
#SBATCH --output=outputs/slurm_figures-%A_%a.out
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=manuel.suarez@cimat.mx

srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python figures.py results figures /home/est_posgrado_manuel.suarez/data/challenge/dataset/training_patches /home/est_posgrado_manuel.suarez/data/challenge/dataset/training_noisy_labels /home/est_posgrado_manuel.suarez/data/challenge-features/dataset
