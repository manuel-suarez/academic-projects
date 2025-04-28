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
# In CIMAT dataset we need to specify the features channels however for this variable
# we are using command line arguments
def list_of_strings(arg):
    return arg.split(",")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    type=list_of_strings,
    required=True,
    default="unet,unet2p,unet3p",
    help="comma separated list of models to evaluate",
)
parser.add_argument(
    "--encoders",
    type=list_of_strings,
    required=True,
    default="resnet,senet,cbamnet",
    help="comma separated list of encoders to evaluate",
)
parser.add_argument(
    "--layers",
    type=list_of_strings,
    required=True,
    default="18,34,50",
    help="comma separated list of encoders num of layers",
)
parser.add_argument(
    "--epochs", type=int, required=True, default=60, help="epochs to analyze"
)
parser.add_argument(
    "--metrics",
    type=list_of_strings,
    required=True,
    default="accuracy,precision,recall,specificity,iou,dice",
    help="comma separated list of metrics to evaluate",
)
parser.add_argument(
    "--steps",
    type=list_of_strings,
    required=True,
    default="train,valid,test",
    help="comma separated list of ",
)
parser.add_argument(
    "--feat_channels",
    type=list_of_strings,
    required=True,
    default="o,w,v",
    help="comma separated list of features to use",
)
args = parser.parse_args()
print("Args: ", args)
epochs = args.epochs

# Model, Encoder and Layers metrics are saved per file according to the training process
# - Directory structure: encoder-layer/model/metrics.csv
# Epochs and metrics are saved in each row of the metrics.csv file

# So, there are different ways of organize the tables:

# Per model, comparing encoders (rows) and layers (columns) with a fixed num of epochs
# Per model, comparing encoder-layer (rows) with variable num of epochs (cols)

# We need a code base to dynamically select the structure needed for the table however
# in this first version we are using the next combination for the informs needed:


# Function to bold the max value per row
def bold_max_row(row):
    max_val = row.max()
    return [f"\\textbf{{{val}}}" if val == max_val else val for val in row]


# Per model, compare metrics for encoders (rows) with num of layers (cols)

# Define variables (we are defining locally the variables to not depend on the modules)
models = args.models
model_names = {
    "unet": "UNet",
    "unet2p": "UNet++",
    "unet3p": "UNet3p",
    "linknet": "LinkNet",
    "pspnet": "PSPNet",
    "fpn": "FPN",
    "deeplabv3p": "DeepLabV3+",
    "manet": "MA-Net",
}
encoders = args.encoders
layers = args.layers
metrics = args.metrics
steps = args.steps
feat_channels = args.feat_channels
feat_channels_names = {
    "o": "origin",
    "ta": "texture/asm",
    "tc": "texture/contrast",
    "td": "texture/dissimilarity",
    "tn": "texture/energy",
    "te": "texture/entropy",
    "tr": "texture/glcmcorrelation",
    "tm": "texture/glcmmean",
    "tv": "texture/glcmvariance",
    "th": "texture/homogeneity",
    "tx": "texture/max",
}


def encoder_name(encoder, layer):
    name = "ResNet" + str(layer)
    if "senet" in encoder:
        name = name + "+SE"
    if "cbamnet" in encoder:
        name = name + "+CBAM"
    if "mr" in encoder:
        if "v2_" in encoder:
            name = name + "+MR (cat)"
        else:
            name = name + "+MR (add)"
    return name


def print_metric_results(metric, step):
    print("\\subsubsection{{Metric: {0}, step: {1}}}".format(metric, step))
    table_results = {
        "Model": [],
    }
    metrics_dicts = {
        feat_channels_names[feat_channel]: [] for feat_channel in feat_channels
    }
    table_results.update(metrics_dicts)
    for model, encoder, layer, feat_channel in product(
        models, encoders, layers, feat_channels
    ):
        if model_names[model] not in table_results["Model"]:
            table_results["Model"].append(model_names[model])
        encoder_layer = f"{encoder}{layer}"
        metrics_path = os.path.join("metrics", encoder_layer, model, feat_channel)
        # Open metrics file and read results
        metrics_file = os.path.join(metrics_path, f"epoch_metrics.csv")
        if not os.path.exists(metrics_file):
            table_results[feat_channels_names[feat_channel]].append("NA")
            continue
        # Open CSV
        metrics_data = pd.read_csv(metrics_file)
        # Check if training epochs was register on CSV (if num of rows in CSV is greater than epochs required)
        rows, _ = metrics_data.shape
        if rows < epochs:
            table_results[feat_channels_names[feat_channel]].append("NA")
            continue
        # Read loss
        metric_data = metrics_data[f"{step}_{metric}"]
        metric_encoder = float(metric_data.iloc[epochs - 1])
        # Append to model-encoder data
        table_results[feat_channels_names[feat_channel]].append(metric_encoder)

    df_results = pd.DataFrame(table_results)
    # Apply the function row-wise
    # df_results = df_results.apply(bold_max_row, axis=1)
    print(df_results.to_latex(index=False, escape=False))


if __name__ == "__main__":
    for metric, step in product(metrics, steps):
        print_metric_results(metric, step)
