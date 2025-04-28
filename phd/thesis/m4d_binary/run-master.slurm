#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=SOS-UNet-MasterTraining-Pretrained
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/master/slurm-master-%A_%a.out

# Reimplementation of master training process using SLURM job arrays
# We are using a master array with the combinations of models-encoders to train and using
# the job array id to select which model to run. We are using the same file indicators to
# know if a model-encoder combination has already trained to avoid duplicates and proceed
# with next model-encoder.
#
# In case we need to add new combinations of models and encoders we will simply add to the
# lists and the file indicator will tell us which combination not to run

# Initial configuration
models=$1
encoders=$2
echo "Models $models"
echo "Encoders $encoders"

epochs=$3
pretrained=$4
dataset=SOS
# We are running the combination of models and encoders
# 1.- Running training process
# 2.- Running figures
# 3.- Running gradcam
# 4.- TODO Implement SHAP
# 5.- TODO Implement comparative tables (export to latex)

# Build main model-encoder array
# CBDNet with base encoder will be first
#result+=("base,cbdnet")
# Using all other models
IFS="," read -ra models_list <<< "$models"
IFS="," read -ra encoders_list <<< "$encoders"
echo "Models list: ${models_list[@]}"
echo "Encoders list: ${encoders_list[@]}"
for model_name in ${models_list[@]}; do 
  for encoder_name in ${encoders_list[@]}; do 
    result+=("${encoder_name},${model_name}")
  done
done
echo "Result: $result"
# Selecting model-encoder using task array id
# Get the SLURM array index
task_id=${SLURM_ARRAY_TASK_ID}
job_name=${SLURM_JOB_NAME}
node_list=${SLURM_JOB_NODELIST}

# Check if the task ID is within bounds
if [[ $task_id -ge ${#result[@]} ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($task_id) out of bounds. Max index: $((${#result[@]} - 1))"
    exit 1
fi
IFS="," read -r encoder_name model_name <<< "${result[$task_id]}"
echo "Selecting ${model_name}-${encoder_name}"
echo "Running ${job_name} on ${node_list}"
mkdir -p outputs/$encoder_name/$model_name
# Verify if the encoder has been processed previously
if [ ! -f outputs/$encoder_name/$model_name/training.txt ]; then
  # Run training process
  echo "run training process for $model_name, encoder=$encoder_name"
  initial_seed=$RANDOM
  echo "using $initial_seed for reproducible results"
  # We are running directly the python process to wait for the training to finish
  srun --job-name=${dataset}-${model_name}-${encoder_name}-Training --output=outputs/$encoder_name/$model_name/slurm-training-%A.out /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python main-train.py --model_name=$model_name --encoder_name=$encoder_name --epochs=$epochs --random_seed=$initial_seed --nodes=$node_list --pretrained_encoder=$pretrained

  # Run result process after the training has finished
  
  # Run figures
  #if [ ! -f outputs/$encoder_name/$model_name/figures.txt ]; then
  #  echo "run figures generation for $model_name, encoder=$encoder_name"
  #  sbatch --job-name=${dataset}-${model_name}-${encoder_name}-Figures --output=outputs/$encoder_name/$model_name/slurm-figures-%A.out run-figures.slurm $model_name $encoder_name 20,40,60,80,100
  #fi
  #
  # Run gradcam
  #if [ ! -f outputs/$encoder_name/$model_name/gradcam.txt ]; then
  #  echo "run gradcam generation for $model_name, encoder=$encoder_name"
  #  sbatch --job-name=${dataset}-${model_name}-${encoder_name}-GradCAM --output=outputs/$encoder_name/${model_name}/slurm-gradcam-%A.out run-gradcam.slurm $model_name $encoder_name 20,40,60,80,100
  #fi
  # TODO run SHAP
  # TODO run latex tables generation
fi
