import os
import torch
import logging
import argparse
import pandas as pd

from itertools import product

# In this scripts we are generating the latex code for different comparative tables
# of the metrics obtained with the models

# Because we have different variables of metrics we can combine them in different ways
# Variables:
# - Model: semantic segmentation models (defined by the decoder part of the model)
#   - UNet, UNet2p, UNet3p, LinkNet, PSPNet, FPNet, DeepLabV3p, MA-Net
# - Encoder: architecture used in the encoder part of the model
#   - ResNet, SENet, CBAMNet, MRNet
# - Layers: size of the encoder in terms of the layers used
#   - 18, 34, 50, 101, 152
# - Epochs: num of epochs of the training process
#   - 1-100
# - Metrics: metrics calculated in the training and validation process
#   - Accuracy, Precision, Recall, Sensitivity, F1-Score, IoU, Dice

# Model, Encoder and Layers metrics are saved per file according to the training process
# - Directory structure: encoder-layer/model/metrics.csv
# Epochs and metrics are saved in each row of the metrics.csv file

# So, there are different ways of organize the tables:

# Per model, comparing encoders (rows) and layers (columns) with a fixed num of epochs
# Per model, comparing encoder-layer (rows) with variable num of epochs (cols)

# We need a code base to dynamically select the structure needed for the table however
# in this first version we are using the next combination for the informs needed:

# Per model, compare metrics for encoders (rows) with num of layers (cols)

# Define variables (we are defining locally the variables to not depend on the modules)
models = ["unet", "unet2p", "unet3p", "linknet", "pspnet", "fpn", "deeplabv3p", "manet"]
encoders = [
    "resnet",
    "senet",
    "cbamnet",
    "mrnet",
    "mrnetv2_",
    "resnetmr",
    "senetmr",
    "cbamnetmr",
    "resnetmrv2_",
    "senetmrv2_",
    "cbamnetmrv2_",
]
layers = ["18", "34", "50", "101", "152"]
metrics = ["accuracy", "precision", "recall", "sensitivity", "iou", "dice"]

table_results = {
    "models": [],
}
encoders_dicts = {
    encoder + layer: [] for encoder, layer in product(encoders, layers[:1])
}
table_results.update(encoders_dicts)
print(table_results)
for model, encoder, layer in product(models, encoders, layers[:1]):
    if model not in table_results["models"]:
        table_results["models"].append(model)
    encoder_layer = f"{encoder}{layer}"
    metrics_path = os.path.join("metrics", encoder_layer, model)
    # Open metrics file and read results
    metrics_file = os.path.join(metrics_path, "metrics.csv")
    if not os.path.exists(metrics_file):
        table_results[encoder_layer].append("NA")
        continue
    # Open CSV
    metrics_data = pd.read_csv(os.path.join(metrics_path, "metrics.csv"))
    # Read loss
    loss_data = metrics_data["train_loss"]
    loss_encoder = loss_data.iloc[-1]
    # Append to model-encoder data
    table_results[encoder_layer].append(loss_encoder)


df_results = pd.DataFrame(table_results)
print(df_results.to_latex(index=False, float_format="{:.6f}".format))
