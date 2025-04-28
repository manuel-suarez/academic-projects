import os
import torch
import argparse
import pandas as pd

from itertools import product

from utils import default
from models import build_model
from slurm import slurm_vars
from metrics import calculate_metrics
from dataset import prepare_dataloaders
from matplotlib import pyplot as plt

# Configure command line arguments
parser = argparse.ArgumentParser()
# We are passing the epochs argument as a list of integers
parser.add_argument(
    "--epochs",
    type=int,
    required=True,
    default=60,
    help="epochs weigths to load",
)
parser.add_argument(
    "--model_name", type=str, required=True, default="unet", help="model to use"
)
# parser.add_argument(
#    "--encoder_name", type=str, required=True, default="base", help="encoder to use"
# )
parser.add_argument(
    "--feat_channels", type=str, required=True, default="oov", help="features to use"
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
    "--done_message",
    type=str,
    required=False,
    default="Done!",
    help="message to show on training end",
)
args = parser.parse_args()
print("Args: ", args)

# Initial configuration
base_path = os.path.expanduser("~")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
model_name = default(args.model_name, "unet")
# encoder_name = default(args.encoder_name, "base")
feat_channels = default(args.feat_channels, "oov")
print("Device: ", device)
print("Model: ", model_name)
# print("Encoder: ", encoder_name)
print("Epochs: ", epochs)
print("Features channels: ", feat_channels)


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


def generate_figures(models, names, dataloader, figures_path):
    for image, label, image_name in dataloader:
        image, label = image.to(device), label.to(device)
        image_name = image_name[0]

        # Generate metrics
        models_metrics = []
        with torch.no_grad():
            models_metrics = []
            for model in models:
                output = model(image)
                _, model_metrics = calculate_metrics(output, label)
                models_metrics.append(model_metrics["accuracy"])

        # If metric of model 0 is less than model 1 or model 2 (MR models), then
        # generate predictions and figures
        if (
            models_metrics[0] < models_metrics[1]
            or models_metrics[0] < models_metrics[2]
        ):
            generated_masks = {}
            for model, name in zip(models, names):
                generated_masks[name] = generate_mask(model, image)

            # Generate plot
            output_filename = os.path.join(figures_path, image_name + "_figure.png")
            label = torch.squeeze(label, 0)
            save_figure(output_filename, image, label, generated_masks)


def build_and_load_model(encoder_name, model_name, feat_channels, epoch, in_channels=1):
    model = build_model(
        model_name=model_name,
        encoder_name=encoder_name,
        in_channels=in_channels,
    ).to(device)
    weights_path = os.path.join(args.weights_path, encoder_name, model_name)
    if not os.path.exists(weights_path):
        raise Exception(f"Weights path {weights_path} doesn't exists.")
    weights_fname = os.path.join(
        weights_path, f"weights_{feat_channels}_{epoch}_epochs.pth"
    )
    # Load model weights
    model.load_state_dict(
        torch.load(weights_fname, map_location=device, weights_only=True)
    )
    return model


if __name__ == "__main__":
    # Prepare data loaders with one image loading at a time to generate plots per item
    dataloaders = prepare_dataloaders(
        base_path,
        args.feat_channels,
        train_batch_size=1,
        valid_batch_size=1,
        test_batch_size=1,
        return_names=True,
    )
    # Configure and load models (three versions of resnet)
    model_resnet18 = build_and_load_model(
        "resnet18", "unet", args.feat_channels, args.epochs
    ).to(device)
    model_resnetmr18 = build_and_load_model(
        "resnetmr18", "unet", args.feat_channels, args.epochs
    ).to(device)
    model_resnetmrv2_18 = build_and_load_model(
        "resnetmrv2_18", "unet", args.feat_channels, args.epochs
    ).to(device)

    figures_path = os.path.join(args.figures_path, "resnet18", "unet", "comparison")
    os.makedirs(figures_path, exist_ok=True)

    # For each instance of each dataloader we generate one plot
    models = [model_resnet18, model_resnetmr18, model_resnetmrv2_18]
    models_names = ["resnet18", "resnetmr18", "resnetmrv2_18"]
    datasets_names = ["train", "valid", "test"]
    for dataset_name, dataloader in zip(datasets_names, dataloaders):
        figures_name_path = os.path.join(figures_path, dataset_name)
        os.makedirs(figures_name_path, exist_ok=True)
        generate_figures(models, models_names, dataloader, figures_name_path)

    # Create flag file to indicate main script that weight models has been generated
    f = open(
        os.path.join("outputs", encoder_name, model_name, "figures_comparison.txt"), "x"
    )
    f.close()
    logging.info(args.done_message)
    print(args.done_message)
