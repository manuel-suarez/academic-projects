#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=Krestenitis-Master
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=0-120%1
#SBATCH --output=outputs/master/slurm-master-%A_%a.out

# Reimplementation of master training process using SLURM job arrays
# We are using a master array with the combinations of models-encoders to train and using
# the job array id to select which model to run. We are using the same file indicators to
# know if a model-encoder combination has already trained to avoid duplicates and proceed
# with next model-encoder.
#
# In case we need to add new combinations of models and encoders we will simply add to the
# lists and the file indicator will tell us which combination not to run
#
# Initial configuration
epochs=$1
num_classes=5
dataset=Krestenitis
# We are running the combination of models and encoders
# 1.- Running training process
# 2.- Running figures
# 3.- Running gradcam
# 4.- TODO Implement SHAP
# 5.- TODO Implement comparative tables (export to latex)
#
# Build main model-encoder array
for size in 18 34; do
  for model_name in unet linknet pspnet fpn deeplabv3p unet2p unet3p; do
    for encoder in resnet resnetmr resnetmrv2_ resnetmrv3_ resnetmrv4_ resnetmrv5_ resnetmrv6_ resnetmrv7_ resnetmrv8_; do
      encoder_name=$encoder$size
      result+=("${encoder_name},${model_name}")
    done
  done
done

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
echo "Running on ${node_list}"
mkdir -p outputs/$encoder_name/$model_name
# Verify if the encoder has been processed previously
if [ ! -f outputs/$encoder_name/$model_name/training.txt ]; then
  # Run training process
  echo "run training process for $model_name, encoder=$encoder_name, num_classes=$num_classes"
  initial_seed=$RANDOM
  echo "using $initial_seed for reproducible results"
  # We are running directly the python process to wait for the training to finish
  srun --job-name=${dataset}-${model_name}-${encoder_name}-Training --output=outputs/$encoder_name/$model_name/slurm-training-%A.out /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python main-train.py --model_name=$model_name --encoder_name=$encoder_name --epochs=$epochs --num_classes=$num_classes --random_seed=$initial_seed --nodes=$node_list


  # For the next processes we will use the same configuration as the master bash script because we need to let the process run separatedly and this script continue with the next training process so we only validate if the process has been runned and send to his own job in case of not 

  # For Krestenitis dataset we need to modify the figures and gradcam processes cause we need to consider that
  # is a multiclass problem.
  
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
# Unless bash master script we don't need to sleep and check to verify if training has finished because SRUN will wait until the training script has finished

