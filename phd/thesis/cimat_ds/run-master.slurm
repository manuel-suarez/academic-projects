#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=CIMAT-MasterTraining
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
epochs=80
dataset=Full_CIMAT
# We are running the combination of models and encoders
# 1.- Running training process
# 2.- Running figures
# 3.- Running gradcam
# 4.- TODO Implement SHAP
# 5.- TODO Implement comparative tables (export to latex)

# Build main model-encoder array
#for size in 18 34; do
#  for model_name in unet linknet pspnet fpn deeplabv3p manet unet2p unet3p; do
#    for encoder in resnet senet cbamnet mrnet mrnetv2_ mrnetv3_ resnetmr senetmr cbamnetmr resnetmrv2_ senetmrv2_ cbamnetmrv2_; do
#      for feat_channels in o ta tc td tn te tr tm tv th tx; do
for size in 34; do
  for model_name in unet; do # linknet pspnet fpn deeplabv3p manet unet2p unet3p; do
    for encoder in resnet resnetmrv3_ resnetmdv2_; do # resnetmr resnetmrv2_ resnetmrv3_; do
      for feat_channels in o; do # ta tc td tn te tr tm tv th tx; do
        encoder_name=$encoder$size
        result+=("${encoder_name},${model_name},${feat_channels}")
      done
    done
  done
done
# Selecting model-encoder using task array id
# Get the SLURM array index
#
# Apparently we can not run array taks on clúster so we are changing the strategy to use command line arguments
task_id=${SLURM_ARRAY_TASK_ID}
# task_id=$1
job_name=${SLURM_JOB_NAME}
node_list=${SLURM_JOB_NODELIST}

# Check if the task ID is within bounds
if [[ $task_id -ge ${#result[@]} ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($task_id) out of bounds. Max index: $((${#result[@]} - 1))"
    exit 1
fi
IFS="," read -r encoder_name model_name feat_channels <<< "${result[$task_id]}"
echo "Selecting ${model_name}-${encoder_name}, feat_channels ${feat_channels}"
echo "Running ${job_name} on ${node_list}"
mkdir -p outputs/$encoder_name/$model_name/$feat_channels
# Verify if the encoder has been processed previously
if [ ! -f outputs/$encoder_name/$model_name/$feat_channels/training.txt ]; then
  # Run training process
  echo "run training process for $model_name, encoder=$encoder_name, feat_channels=$feat_channels"
  initial_seed=$RANDOM
  echo "using $initial_seed for reproducible results"
  # We are running directly the python process to wait for the training to finish
  srun --job-name=${dataset}-${model_name}-${encoder_name}-Training --output=outputs/$encoder_name/$model_name/$feat_channels/slurm-training-%A.out /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python main-train.py --model_name=$model_name --encoder_name=$encoder_name --epochs=$epochs --feat_channels=$feat_channels --random_seed=$initial_seed --nodes=$node_list
  # Unless bash master script we don't need to sleep and check to verify if training has finished because SRUN will wait until the training script has finished, however we need to be 
  # sure that the training has finished so we nest the figures and gradcam inside the if of the training process
  # Run figures
  #if [ ! -f outputs/$encoder_name/$model_name/$feat_channels/figures.txt ]; then
  #  echo "run figures generation for $model_name, encoder=$encoder_name, feat_channels=$feat_channels"
  #  sbatch --job-name=${dataset}-${model_name}-${encoder_name}-${feat_channels}-Figures --output=outputs/$encoder_name/$model_name/$feat_channels/slurm-figures-%A.out run-figures.slurm $model_name $encoder_name 20,40,60,80,100 $feat_channels
  #fi
  #
  # Run gradcam
  #if [ ! -f outputs/$encoder_name/$model_name/$feat_channels/gradcam.txt ]; then
  #  echo "run gradcam generation for $model_name, encoder=$encoder_name, feat_channels=$feat_channels"
  #  sbatch --job-name=${dataset}-${model_name}-${encoder_name}-${feat_channels}-GradCAM --output=outputs/$encoder_name/$model_name/$feat_channels/slurm-gradcam-%A.out run-gradcam.slurm $model_name $encoder_name 20,40,60,80,100 $feat_channels
  #fi
fi

# For the next processes we will use the same configuration as the master bash script because we need to let the process run separatedly and this script continue with the next training process so we only validate if the process has been runned and send to his own job in case of not 

# TODO run SHAP
# TODO run latex tables generation
