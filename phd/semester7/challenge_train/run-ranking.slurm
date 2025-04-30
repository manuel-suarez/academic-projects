#!/bin/bash

#SBATCH --partition=C1Mitad1
#SBATCH --job-name=ChallengeRankingJD
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-5
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=manuel.suarez@cimat.mx

srun /home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python ranking.py results output /home/est_posgrado_manuel.suarez/data/challenge/dataset/training_patches /home/est_posgrado_manuel.suarez/data/challenge/dataset/training_noisy_labels /home/est_posgrado_manuel.suarez/data/challenge-features/dataset
