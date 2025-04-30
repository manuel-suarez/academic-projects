#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=TextureMatrices
#SBATCH --time=0
#SBATCH --mem=0

srun bash make_texture_matrices.sh ~/data/cimat/dataset-cimat products texture temp2 $1
