#!/bin/bash

echo "Cleaning previous outputs"
rm outputs/slurm_results*.out

echo "Sendind sbatch job"
# Submit the job and capture the output
job_output=$(python prepare-results.py --model_name=unet --encoder_name=resnet18 --datasets=train --ntasks=500)
# Extract the job ID from the output 
job_id=$(echo $job_output | awk '{print $4}')

# Start the watch output
sleep 5s
watch cat outputs/slurm_results-${job_id}_1.out
