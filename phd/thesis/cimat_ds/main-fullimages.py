import os
import torch
import logging
import argparse

import numpy as np

from utils import default
from models import build_model
from slurm import slurm_vars
from tqdm import tqdm
from itertools import product
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

from calculate_outcomes import calculate_outcomes
from calculate_metrics import calculate_metrics as calculate_model_metrics


# Define a custom argument type for a list
def list_of_strings(arg):
    return arg.split(",")


# Configure command line arguments
parser = argparse.ArgumentParser()
# We are passing the epochs argument as a list of integers
parser.add_argument(
    "--epochs",
    type=int,
    required=True,
    default=60,
    help="epoch weigths to load",
)
parser.add_argument(
    "--model_name", type=str, required=True, default="unet", help="model to use"
)
parser.add_argument(
    "--encoder_name", type=str, required=True, default="base", help="encoder to use"
)
parser.add_argument(
    "--image_name", type=str, required=True, help="image name to segment"
)
parser.add_argument(
    "--image_patch_size", type=int, required=False, default=256, help="image patch size"
)
parser.add_argument(
    "--mask_threshold",
    type=float,
    required=False,
    default=0.5,
    help="threshold value for segmentation mask",
)
parser.add_argument(
    "--images_path",
    type=str,
    required=False,
    default="full_images",
    help="path to full images data",
)
parser.add_argument(
    "--weights_path",
    type=str,
    required=False,
    default="weights",
    help="path to load training weighs",
)
parser.add_argument(
    "--results_path",
    type=str,
    required=False,
    default="figures",
    help="path to store full segmentation result",
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


# Initial configuration
base_path = os.path.expanduser("~")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
model_name = default(args.model_name, "unet")
encoder_name = default(args.encoder_name, "base")
print("Device: ", device)
print("Model: ", model_name)
print("Encoder: ", encoder_name)
print("Epochs: ", epochs)

mask_threshold = args.mask_threshold
patch_size = args.image_patch_size
if __name__ == "__main__":
    # We are working directly with the dataset cause we are accesing by index
    data_path = os.path.join(base_path, "data")
    images_path = os.path.join(data_path, args.images_path)

    # Prepare model according to SLURM array task id
    model = build_model(
        model_name=model_name,
        encoder_name=encoder_name,
        in_channels=1,
    ).to(device)

    # Configure paths
    weights_path = os.path.join(args.weights_path, encoder_name, model_name, "o")
    if not os.path.exists(weights_path):
        raise Exception("Weights path doesn't exists.")
    # Load weights corresponding to epochs
    weights_fname = os.path.join(weights_path, f"weights_{epochs}_epochs.pth")
    # Load model weights
    model.load_state_dict(
        torch.load(weights_fname, map_location=device, weights_only=True)
    )

    results_path = os.path.join(
        args.results_path, "segmentation", encoder_name, model_name
    )
    os.makedirs(results_path, exist_ok=True)

    # Load full image
    image_path = os.path.join(images_path, args.image_name)
    if not os.path.exists(image_path):
        raise Exception(
            f"Image {args.image_name} doesn't exists on images path: {images_path}"
        )
    image = imread(image_path, as_gray=True)
    print("Image shape: ", image.shape)
    mask = np.zeros(image.shape)
    # Num patches
    patches_ii = image.shape[0] // patch_size
    patches_jj = image.shape[1] // patch_size
    print("Num of patches: ", patches_ii, patches_jj)
    # Traverse patches and apply segmentation
    for index_i, index_j in tqdm(product(range(patches_ii), range(patches_jj))):
        # Get image patch position according patch sizes
        pos_i = index_i * patch_size
        pos_j = index_j * patch_size
        if pos_i + patch_size > image.shape[0]:
            pos_i = pos_i - (pos_i + patch_size - image.shape[0])
        if pos_j + patch_size > image.shape[1]:
            pos_j = pos_j - (pos_i + patch_size - image.shape[1])

        # Get image patch
        image_patch = torch.from_numpy(
            # Add channel dimension
            np.expand_dims(
                image[pos_i : pos_i + patch_size, pos_j : pos_j + patch_size], 0
            )
        ).type(torch.float)
        # print("Image patch: ", image_patch.shape)
        # Add batch dimension
        image_patch = image_patch.unsqueeze(0)
        # print(image_patch.shape)
        # Generate patch mask
        with torch.no_grad():
            image_patch = image_patch.to(device)
            outputs = model(image_patch)
        # Apply sigmoid
        outputs = torch.sigmoid(outputs)
        # Binarize
        mask_patch = (outputs > mask_threshold).float().squeeze().cpu().numpy()
        # Add patch to full mask
        mask[pos_i : pos_i + patch_size, pos_j : pos_j + patch_size] = mask_patch

    # Save results (convert to 255 int values to save as PNG)
    print("Mask values: ", mask.min(), mask.max(), np.unique(mask))
    mask = (mask * 255).astype(np.int16)
    print("Mask values: ", mask.min(), mask.max(), np.unique(mask))
    os.makedirs(os.path.join(args.results_path, "masks"), exist_ok=True)
    imsave(os.path.join(args.results_path, "masks", args.image_name), mask)
    print(args.done_message)
