import os
import torch
import logging
import argparse
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from utils import default
from models import build_model
from slurm import slurm_vars
from skimage.io import imread
from matplotlib import pyplot as plt

from calculate_outcomes import outcomes
from calculate_metrics import metrics


# Define a custom argument type for a list
def list_of_strings(arg):
    return arg.split(",")


# Configure command line arguments
parser = argparse.ArgumentParser()
# We are passing the epochs argument as a list of integers
parser.add_argument(
    "--epochs",
    type=list_of_strings,
    required=True,
    default="20,40,60,80,100",
    help="comma separated list of epoch weigths to load",
)
parser.add_argument(
    "--model_name", type=str, required=True, default="unet", help="model to use"
)
parser.add_argument(
    "--encoder_name", type=str, required=True, default="base", help="encoder to use"
)
parser.add_argument(
    "--dataset", type=str, required=False, default="train", help="dataset to load"
)
parser.add_argument(
    "--metrics_path",
    type=str,
    required=False,
    default="metrics",
    help="path to load metrics",
)
parser.add_argument(
    "--figures_path",
    type=str,
    required=False,
    default="figures",
    help="path to store input figures",
)
parser.add_argument(
    "--outputs_path",
    type=str,
    required=False,
    default="figures",
    help="path to store output figures",
)
parser.add_argument(
    "--replace_dst_files",
    type=bool,
    required=False,
    default=False,
    help="flag to confirm output file replacement",
)
parser.add_argument(
    "--logging_path",
    type=str,
    required=False,
    default="outputs",
    help="path to store logging",
)
parser.add_argument(
    "--done_message",
    type=str,
    required=False,
    default="Done!",
    help="message to show on training end",
)
args = parser.parse_args()
logging.info("Args: ", args)
print("Args: ", args)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(
        args.logging_path,
        f"slurm-training_{slurm_vars['array_job_id']}_{slurm_vars['array_task_id']}.out",
    ),
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

# Initial configuration
base_path = os.path.expanduser("~")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = [int(epoch) for epoch in args.epochs]
model_name = default(args.model_name, "unet")
encoder_name = default(args.encoder_name, "base")
dataset = default(args.dataset, "train")
print("Device: ", device)
print("Model: ", model_name)
print("Encoder: ", encoder_name)
print("Epochs: ", epochs)
print("Dataset: ", dataset)

# Read images metrics and sort figures descending by metric

# Initial configuration
base_path = os.path.expanduser("~")

if __name__ == "__main__":
    # We are working directly with the dataset cause we are accesing by index
    data_path = os.path.join(base_path, "data", "krestenitis_v1", args.dataset)
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels_1D")
    # Figures path
    figures_path = os.path.join(
        args.figures_path, "results", encoder_name, model_name, args.dataset
    )
    # Output path
    output_path = os.path.join(
        args.outputs_path, "results_per_metrics", encoder_name, model_name, args.dataset
    )
    os.makedirs(output_path, exist_ok=True)
    # Reading image list
    images_list = [fname.split(".")[0] for fname in os.listdir(images_path)]
    # Reading images metrics (open image metrics and read metrics per epoch)
    metrics_path = os.path.join(
        args.metrics_path, "images", encoder_name, model_name, args.dataset
    )
    images_data: dict = {"image": []}
    images_outcomes: dict = {
        f"epoch{epoch}_{outcome}": [] for epoch, outcome in product(epochs, outcomes)
    }
    images_data.update(images_outcomes)
    images_metrics: dict = {
        f"epoch{epoch}_{metric}": [] for epoch, metric in product(epochs, metrics)
    }
    images_data.update(images_metrics)
    print(images_data)
    for image_name in images_list:
        # Open image metrics
        image_data = pd.read_csv(os.path.join(metrics_path, image_name + ".csv"))
        images_data["image"].append(image_name)
        # Search and locate epoch metrics
        for epoch in epochs:
            image_epoch_data = image_data[image_data["epoch"] == epoch]
            # Add outcomes
            for outcome in outcomes:
                images_data[f"epoch{epoch}_{outcome}"].append(
                    image_epoch_data[outcome].iloc[0]
                )
            for metric in metrics:
                images_data[f"epoch{epoch}_{metric}"].append(
                    image_epoch_data[metric].iloc[0]
                )
    # Save data to a DataFrame pandas to sorting
    images_df = pd.DataFrame.from_dict(images_data)
    print(images_df)

    # Sorting data by outcome-epoch and save figures per outcomes-epochs
    for outcome, epoch in product(outcomes, epochs):
        # Sort dataframe per outcome-epoch
        print(outcome, epoch)
        images_df_sorted = images_df.sort_values(
            by=f"epoch{epoch}_{outcome}", ascending=False
        )
        # Output path
        images_output_path = os.path.join(output_path, f"epoch{epoch}_{outcome}")
        os.makedirs(images_output_path, exist_ok=True)
        # Get images names sorted by epoch+outcome and traverse to copy to output path renaming by index
        images_names = images_df_sorted["image"].values.tolist()
        for index, image_name in tqdm(enumerate(images_names[:40])):
            # Check if source image exists (we don't let the figures script to full run so in this moment we are missing some figures)
            src_path = os.path.join(figures_path, f"{image_name}.png")
            dst_path = os.path.join(images_output_path, f"{index:04d}_{image_name}.png")
            if os.path.exists(src_path) and (
                not os.path.exists(dst_path) or args.replace_dst_files
            ):
                shutil.copy(
                    src=src_path,
                    dst=dst_path,
                )
    # Sorting data by metric-epoch and save figures per metrics-epochs
    for metric, epoch in product(metrics, epochs):
        # Sort dataframe per metric-epoch
        print(metric, epoch)
        images_df_sorted = images_df.sort_values(
            by=f"epoch{epoch}_{metric}", ascending=False
        )
        # Output path
        images_output_path = os.path.join(output_path, f"epoch{epoch}_{metric}")
        os.makedirs(images_output_path, exist_ok=True)
        # Get images names sorted by epoch+metric and traverse to copy to output path renaming by index
        images_names = images_df_sorted["image"].values.tolist()
        for index, image_name in tqdm(enumerate(images_names[:40])):
            # Check if source image exists (we don't let the figures script to full run so in this moment we are missing some figures)
            src_path = os.path.join(figures_path, f"{image_name}.png")
            dst_path = os.path.join(images_output_path, f"{index:04d}_{image_name}.png")
            if os.path.exists(src_path) and (
                not os.path.exists(dst_path) or args.replace_dst_files
            ):
                shutil.copy(
                    src=src_path,
                    dst=dst_path,
                )
    print(args.done_message)
