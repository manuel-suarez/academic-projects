#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=PolarimetricDecompositions
#SBATCH --time=0
#SBATCH --mem=0

srun bash make_polarimetric_decompositions.sh ~/data/cimat/dataset-sentinel SLC POL temp1 $1
