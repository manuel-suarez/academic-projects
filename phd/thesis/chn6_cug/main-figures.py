import os
import torch
import logging
import argparse
import numpy as np

from utils import default
from models import build_model
from slurm import slurm_vars
from dataset import prepare_dataloaders
from matplotlib import pyplot as plt


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
    "--max_train_images",
    type=int,
    required=False,
    default=None,
    help="max images to load from training set",
)
parser.add_argument(
    "--max_valid_images",
    type=int,
    required=False,
    default=None,
    help="max images to load from validation set",
)
parser.add_argument(
    "--max_test_images",
    type=int,
    required=False,
    default=None,
    help="max images to load from test set",
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
logging.info(f"Device: {device}")
logging.info(f"Model name: {model_name}")
logging.info(f"Encoder name: {encoder_name}")
logging.info(f"Epochs: {epochs}")
print("Device: ", device)
print("Model: ", model_name)
print("Encoder: ", encoder_name)
print("Epochs: ", epochs)


def generate_mask(model, image):
    with torch.no_grad():
        output = model(image)
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    return mask


def save_figure(figure_name, image, label, generated_masks):
    fig, axes = plt.subplots(1, len(generated_masks) + 2, figsize=(15, 5))
    image = np.transpose(image.squeeze(0).numpy(), (1, 2, 0))
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


def generate_figures(model, dataloader, weights_path, figures_path):
    for image, label, image_name in dataloader:
        image, label = image.to(device), label.to(device)
        image_name = image_name[0]

        generated_masks = {}
        for epoch in epochs:
            weights_fname = os.path.join(weights_path, f"weights_{epoch}_epochs.pth")
            # Load model weights
            model.load_state_dict(
                torch.load(weights_fname, map_location=device, weights_only=True)
            )
            # Generate predictions
            generated_masks[epoch] = generate_mask(model, image)

        # Generate plot
        output_filename = os.path.join(figures_path, image_name + "_figure.png")
        label = torch.squeeze(label, 0)
        save_figure(output_filename, image, label, generated_masks)


if __name__ == "__main__":
    # Prepare data loaders with one image loading at a time to generate plots per item
    dataloaders = prepare_dataloaders(
        base_path,
        train_batch_size=1,
        valid_batch_size=1,
        test_batch_size=1,
        return_train_names=True,
        return_valid_names=True,
        max_train_images=args.max_train_images,
        max_valid_images=args.max_valid_images,
    )
    # Prepare model according to SLURM array task id
    model = build_model(
        model_name=model_name,
        encoder_name=encoder_name,
        in_channels=3,
    ).to(device)

    # Configure paths
    weights_path = os.path.join(args.weights_path, encoder_name, model_name)
    figures_path = os.path.join(args.figures_path, encoder_name, model_name)
    if not os.path.exists(weights_path):
        raise Exception("Weights path doesn't exists.")
    os.makedirs(figures_path, exist_ok=True)

    # For each instance of each dataloader we generate one plot
    names = ["train", "valid", "test"]
    for name, dataloader in zip(names, dataloaders):
        figures_name_path = os.path.join(figures_path, name)
        os.makedirs(figures_name_path, exist_ok=True)
        generate_figures(model, dataloader, weights_path, figures_name_path)

    # Create flag file to indicate main script that weight models has been generated
    f = open(os.path.join("outputs", encoder_name, model_name, "figures.txt"), "x")
    f.close()
    logging.info(args.done_message)
    print(args.done_message)
