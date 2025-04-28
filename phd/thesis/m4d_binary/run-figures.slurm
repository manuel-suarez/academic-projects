#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=Krestenitis-Figures
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm_figures-%A_%a.out

echo "Model name: $1"
echo "Encoder name: $2"
echo "Epochs: $3"
echo "Dataset: $4"
echo "Images per task: $5"
echo "Missing images: $6"
echo "Figures path: $7"

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python main-figures.py --model_name=$1 --encoder_name=$2 --epochs=$3 --dataset=$4 --images_per_task=$5 --missing_images=$6 --figures_path=$7
