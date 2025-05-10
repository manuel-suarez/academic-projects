#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=MakeBinaryMasks
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-19
#SBATCH --ntasks=2

SOURCE_DIR=~/data/cimat/dataset-cimat/mask_xcf
OUTPUT_DIR=~/data/cimat/dataset-cimat/mask_png

srun /bin/bash convert_xcf_to_png.sh $SOURCE_DIR $OUTPUT_DIR
