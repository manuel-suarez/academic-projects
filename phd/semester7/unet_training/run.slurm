#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=UNet-Logs
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --array=1-5%1
#SBATCH --nodes=1

export SM_FRAMEWORK=tf.keras
/home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-tf/bin/python main.py results 5
