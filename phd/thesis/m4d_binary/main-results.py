import os
import torch
import logging
import argparse

import numpy as np
import pandas as pd

from utils import default
from models import build_model
from slurm import slurm_vars
from skimage.io import imread
from matplotlib import pyplot as plt

from calculate_outcomes import outcomes, calculate_outcomes
from calculate_metrics import metrics, calculate_metrics as calculate_model_metrics


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
    "--images_per_task",
    type=int,
    required=True,
    help="num of images to process per task",
)
parser.add_argument(
    "--missing_images",
    type=int,
    required=True,
    help="num of missing images per the integer division of images_per_task",
)
parser.add_argument(
    "--weights_path",
    type=str,
    required=False,
    default="weights",
    help="path to load training weighs",
)
parser.add_argument(
    "--metrics_path",
    type=str,
    required=False,
    default="figures",
    help="path to store figures",
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
# Use SLURM variables to get the index to process
slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
slurm_ntasks = os.getenv("SLURM_NTASKS")
slurm_procid = os.getenv("SLURM_PROCID")
slurm_task_pid = os.getenv("SLURM_TASK_PID")
print("SLURM_ARRAY_TASK_ID: ", slurm_array_task_id)
print("SLURM_NTASKS: ", slurm_ntasks)
print("SLURM_PROCID: ", slurm_procid)
print("SLURM_TASK_PID: ", slurm_task_pid)

# Open image according to array task id
images_per_task = args.images_per_task
images_indexes = [
    int(slurm_procid) * images_per_task + index for index in range(images_per_task)
]
if int(slurm_procid) < args.missing_images:
    additional_index = int(slurm_ntasks) * images_per_task + int(slurm_procid)
    images_indexes.append(additional_index)
print("Dataindex: ", images_indexes)


def calculate_metrics(
    model, image, targets, p_threshold=0.5, outcomes=outcomes, metrics=metrics
):
    with torch.no_grad():
        outputs = model(image)
    # Apply sigmoid
    outputs = torch.sigmoid(outputs)
    outcomes = calculate_outcomes(
        outputs, targets, use_threshold=True, use_sigmoid=False, p_threshold=p_threshold
    )
    metrics = calculate_model_metrics(outcomes)
    return outcomes, metrics


def generate_metrics(model, dataitem, weights_path, metrics_path):
    image, label, image_name = dataitem
    image, label = image.to(device), label.to(device)
    # image_name = image_name[0]
    print(f"Image name: {image_name}")

    calculated_metrics = {"epoch": []}
    model_outcomes = {outcome: [] for outcome in outcomes}
    calculated_metrics.update(model_outcomes)
    model_metrics = {metric: [] for metric in metrics}
    calculated_metrics.update(model_metrics)
    for epoch in epochs:
        calculated_metrics["epoch"].append(epoch)
        weights_fname = os.path.join(weights_path, f"weights_{epoch}_epochs.pth")
        # Load model weights
        model.load_state_dict(
            torch.load(weights_fname, map_location=device, weights_only=True)
        )
        # Generate predictions
        image_outcomes, image_metrics = calculate_metrics(model, image, label)
        # Save outcomes and metrics for this epoch
        for outcome in outcomes:
            calculated_metrics[outcome].append(image_outcomes[outcome])
        for metric in metrics:
            calculated_metrics[metric].append(image_metrics[metric])

    # Save metric on CSV per image
    output_filename = os.path.join(metrics_path, image_name + ".csv")
    calculated_metrics_df = pd.DataFrame.from_dict(calculated_metrics)
    calculated_metrics_df.to_csv(output_filename)


# Initial configuration
base_path = os.path.expanduser("~")

if __name__ == "__main__":
    # We are working directly with the dataset cause we are accesing by index
    data_path = os.path.join(base_path, "data", "krestenitis_v1", args.dataset)
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels_1D")
    images_list = [fname.split(".")[0] for fname in os.listdir(images_path)]

    # Prepare model according to SLURM array task id
    model = build_model(
        model_name=model_name,
        encoder_name=encoder_name,
        in_channels=1,
    ).to(device)

    # Configure paths
    weights_path = os.path.join(args.weights_path, encoder_name, model_name)
    if not os.path.exists(weights_path):
        raise Exception("Weights path doesn't exists.")
    metrics_path = os.path.join(args.metrics_path, "images", encoder_name, model_name)
    os.makedirs(metrics_path, exist_ok=True)

    # For each instance of each dataloader we generate one plot
    metrics_name_path = os.path.join(metrics_path, args.dataset)
    os.makedirs(metrics_name_path, exist_ok=True)
    for image_index in images_indexes:
        print(f"Processing dataindex: {image_index}")
        # Get data (based on index)
        if image_index >= len(images_list):
            print(f"Image index {image_index} out of images list bounds")
            exit(0)
        image_name = images_list[image_index]
        image = torch.unsqueeze(
            torch.from_numpy(
                np.expand_dims(
                    imread(
                        os.path.join(images_path, image_name + ".jpg"), as_gray=True
                    ),
                    0,
                )
            ).type(torch.float),
            0,
        )
        label = torch.unsqueeze(
            torch.from_numpy(
                np.expand_dims(
                    imread(
                        os.path.join(labels_path, image_name + ".png"), as_gray=True
                    ),
                    0,
                )
            ).type(torch.float),
            0,
        )
        # Set only binary label
        label[label > 1] = 0
        generate_metrics(
            model, [image, label, image_name], weights_path, metrics_name_path
        )

    print(args.done_message)
