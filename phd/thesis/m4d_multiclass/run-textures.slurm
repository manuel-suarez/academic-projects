#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=SOS-Textures
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/texture/slurm_textures-%A_%a.out

echo "Distances: $1"
echo "Angles: $2"
echo "Levels: $3"
echo "Patch size: $4"
echo "Dataset: $5"
echo "Version: $6"

srun /home/$(whoami)/tools/anaconda3/envs/py3.10-pt/bin/python main-textures.py --distances=$1 --angles=$2 --levels=$3 --patch_size=$4 --dataset=$5 --version=$6 --images_per_task=$7 --missing_images=$8
