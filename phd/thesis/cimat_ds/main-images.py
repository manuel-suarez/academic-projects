import os
import torch
import logging
import argparse
import itertools
import numpy as np

from skimage.io import imread
from utils import default
from models import build_model
from slurm import slurm_vars
from dataset import prepare_dataloaders
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


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
    default="20,40,60",
    help="comma separated list of epoch weigths to load",
)
parser.add_argument(
    "--model_name", type=str, required=True, default="unet", help="model to use"
)
parser.add_argument(
    "--encoder_name", type=str, required=True, default="base", help="encoder to use"
)
parser.add_argument(
    "--feat_channels", type=str, required=True, default="oov", help="features to use"
)
parser.add_argument(
    "--patch_size", type=int, required=True, default=224, help="patch size of images"
)
parser.add_argument(
    "--images_path", type=str, required=True, default="images", help="images to load"
)
parser.add_argument(
    "--weights_path",
    type=str,
    required=False,
    default="weights",
    help="path to load training weighs",
)
parser.add_argument(
    "--figures_path",
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
    "--max_images",
    type=int,
    required=False,
    default=None,
    help="max images to load from training set",
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
feat_channels = default(args.feat_channels, "oov")
patch_size = default(args.patch_size, 224)
logging.info(f"Device: {device}")
logging.info(f"Model name: {model_name}")
logging.info(f"Encoder name: {encoder_name}")
logging.info(f"Epochs: {epochs}")
logging.info(f"Features channels: {feat_channels}")
logging.info(f"Patch size: {patch_size}")
print("Device: ", device)
print("Model: ", model_name)
print("Encoder: ", encoder_name)
print("Epochs: ", epochs)
print("Features channels: ", feat_channels)
print("Patch size: ", patch_size)


def generate_mask(model, image):
    with torch.no_grad():
        output = model(image)
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    return mask


def save_figure(figure_name, image, label, generated_masks):
    fig, axes = plt.subplots(1, len(generated_masks) + 2, figsize=(15, 5))
    image = image.squeeze(0).squeeze(0)
    label = label.squeeze(0)

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    for i, (epoch, mask) in enumerate(generated_masks.items()):
        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(f"Label epoch {epoch}")
        axes[i + 1].axis("off")

    axes[-1].imshow(label, cmap="gray")
    axes[-1].set_title("True label")
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()


def generate_images(model, images_path, weights_path, figures_path):
    for epoch in epochs:
        os.makedirs(
            os.path.join(figures_path, "images", f"{feat_channels}_{epoch}_epochs"),
            exist_ok=True,
        )
        print(f"Using {feat_channels} features channels, {epoch} epochs")
        weights_fname = os.path.join(
            weights_path, f"weights_{feat_channels}_{epoch}_epochs.pth"
        )
        # Load model weights
        model.load_state_dict(
            torch.load(weights_fname, map_location=device, weights_only=True)
        )
        model.eval()
        # Open images and generate masks overlays
        for fname in os.listdir(images_path):
            print(f"\tProcessing {fname}")
            image = imread(os.path.join(images_path, fname), as_gray=True).astype(
                np.float32
            )
            result = np.zeros_like(image, np.uint8)
            # Process image patches to generate predictions and create image overlay
            image_height, image_width = image.shape
            count_x = int(image_width // patch_size) + 1
            count_y = int(image_height // patch_size) + 1
            # Iterate over patches
            for index, (j, i) in tqdm(
                enumerate(itertools.product(range(count_y), range(count_x)))
            ):
                # Get pixel positions for patch
                y = patch_size * j
                x = patch_size * i
                # Crop whenever patch size is outside image limits
                if x + patch_size > image_width:
                    x = image_width - patch_size - 1
                if y + patch_size > image_height:
                    y = image_height - patch_size - 1
                # Get patch
                image_patch = image[y : y + patch_size, x : x + patch_size]
                image_patch = torch.tensor(image_patch).unsqueeze(0).unsqueeze(0)
                # Process
                mask = generate_mask(model, image_patch)
                # Copy onto result
                result[y : y + patch_size, x : x + patch_size] = mask
            # Make overlay
            overlay = Image.fromarray(result)
            overlay.save(
                os.path.join(
                    figures_path, "images", f"{feat_channels}_{epoch}_epochs", fname
                )
            )


if __name__ == "__main__":
    # Prepare data loaders with one image loading at a time to generate plots per item
    # Prepare model according to SLURM array task id
    model = build_model(
        model_name=model_name,
        encoder_name=encoder_name,
        in_channels=1,
    ).to(device)

    # Configure paths
    images_path = os.path.join(os.path.expanduser("~"), args.images_path)
    weights_path = os.path.join(args.weights_path, encoder_name, model_name)
    figures_path = os.path.join(args.figures_path, encoder_name, model_name)
    if not os.path.exists(weights_path):
        raise Exception("Weights path doesn't exists.")
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(os.path.join(figures_path, "images"), exist_ok=True)

    # For each instance of each dataloader we generate one plot
    generate_images(model, images_path, weights_path, figures_path)

    # Create flag file to indicate main script that weight models has been generated
    f = open(os.path.join("outputs", encoder_name, model_name, "images.txt"), "x")
    f.close()
    logging.info(args.done_message)
    print(args.done_message)
