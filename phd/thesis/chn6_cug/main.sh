#!/bin/bash

# Initial configuration
dataset=CHN6_CUG
epochs=100
# We are running the combination of models and encoders
# 1.- Running training process
# 2.- Running figures
# 3.- Running gradcam
# 4.- TODO Implement SHAP
# 5.- TODO Implement comparative tables (export to latex)
for size in 18 34 50 101 152; do
  for name in resnet senet cbamnet; do
    for model_name in unet linknet pspnet fpn deeplabv3p manet unet2p unet3p; do
      encoder_name=$name$size
      echo "Training ${model_name}-${encoder_name}"
      mkdir -p outputs/$encoder_name/$model_name
      # Verify if the encoder has been processed previously
      if [ ! -f outputs/$encoder_name/$model_name/training.txt ]; then
        # Run training process
        echo "run training process for $model_name, encoder=$encoder_name"
        sbatch --job-name=${dataset}-${model_name}-${encoder_name}-Training --output=outputs/$encoder_name/$model_name/slurm_training-%A.out run-train.slurm $model_name $encoder_name $epochs
        #
        # Wait until last weight of last model has been generated (100 epochs)
        while [ ! -f outputs/$encoder_name/$model_name/training.txt ]; do
          # Sleep
          echo "outputs/$encoder_name/$model_name/training.txt not generated, waiting 60m..."
          sleep 60m
        done
        echo "outputs/$encoder_name/$model_name/training.txt has been generated, training has finished..."
      fi
      #
      # Run figures
      if [ ! -f outputs/$encoder_name/$model_name/figures.txt ]; then
        echo "run figures generation for $model_name, encoder=$encoder_name"
        sbatch --job-name=${dataset}-${model_name}-${encoder_name}-Figures --output=outputs/$encoder_name/$model_name/slurm_figures-%A.out run-figures.slurm $model_name $encoder_name 20,40,60,80,100
      fi
      #
      # Run gradcam
      if [ ! -f outputs/$encoder_name/$model_name/gradcam.txt ]; then
        echo "run gradcam generation for $model_name, encoder=$encoder_name"
        sbatch --job-name=${dataset}-${model_name}-${encoder_name}-GradCAM --output=outputs/$encoder_name/$model_name/slurm_gradcam-%A.out run-gradcam.slurm $model_name $encoder_name 20,40,60,80,100
      fi
    done
  done
done

