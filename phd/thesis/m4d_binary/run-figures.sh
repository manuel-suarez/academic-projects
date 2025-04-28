#!/bin/bash

echo "Cleaning previous outputs"
# rm figures.zip
# rm -rf figures
# rm outputs/slurm_figures*.out

echo "Sendind sbatch job"
# Submit the job and capture the output
job_output=$(python prepare-figures.py --model_name=unet --encoder_name=resnet18 --datasets=test --ntasks=500)
# Extract the job ID from the output 
job_id=$(echo $job_output | awk '{print $4}')

# Start the watch output
sleep 5s
watch tail outputs/slurm_figures-${job_id}_1.out
zip -r figures.zip figures/
