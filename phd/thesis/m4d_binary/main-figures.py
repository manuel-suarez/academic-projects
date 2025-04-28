import os
import torch
import logging
import argparse

import numpy as np

from utils import default
from models import build_model
from slurm import slurm_vars
from itertools import product
from skimage.io import imread
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


def get_segmentation_outcomes(outputs, labels, p_threshold=0.5):
    predictions = torch.squeeze((outputs > p_threshold).int(), 0)
    labels = torch.squeeze(labels.int(), 0)

    TP = ((labels == 1) & (predictions == 1)).squeeze().cpu().numpy()
    FP = ((labels == 0) & (predictions == 1)).squeeze().cpu().numpy()
    TN = ((labels == 0) & (predictions == 0)).squeeze().cpu().numpy()
    FN = ((labels == 1) & (predictions == 0)).squeeze().cpu().numpy()
    return TP, FP, TN, FN


color_dict = {
    "TP": [0, 0, 255],  # Blue
    "TN": [200, 200, 200],  # Gray
    "FP": [255, 0, 0],  # Red
    "FN": [255, 255, 0],  # Yellow
}

color_names = {
    "TP": "True Positive",
    "TN": "True Negative",
    "FP": "False Positive",
    "FN": "False Negative",
}


def generate_segmentation_image(outputs, targets, p_threshold=0.5):
    TP, FP, TN, FN = get_segmentation_outcomes(outputs, targets, p_threshold)

    # Blank image
    outcome_image = np.zeros((outputs.shape[2], outputs.shape[3], 3))
    # Colors
    outcome_image[TP] = color_dict["TP"]
    outcome_image[FP] = color_dict["FP"]
    outcome_image[FN] = color_dict["FN"]
    outcome_image[TN] = color_dict["TN"]

    return outcome_image / 255.0


def generate_mask(model, image, p_threshold=0.5):
    with torch.no_grad():
        outputs = model(image)
    # Apply sigmoid
    outputs = torch.sigmoid(outputs)
    # Binarize
    preds = (outputs > p_threshold).float()

    mask = preds.squeeze().cpu().numpy()
    return outputs, mask


def calculate_metrics(outputs, targets, p_threshold=0.5):
    outcomes = calculate_outcomes(
        outputs, targets, use_threshold=True, use_sigmoid=False, p_threshold=p_threshold
    )
    metrics = calculate_model_metrics(outcomes)
    return outcomes, metrics


def add_metrics(ax, metrics, outcomes, start_y=0.5, line_spacing=0.15):
    """
    Display metrics text in a subplot.

    Args:
        ax: The subplot axis.
        metrics: Dictionary of metric names and values.
        start_y: Starting y-coordinate for the first line of text.
        line_spacing: Spacing between lines.
    """
    # Concatenate metrics into a single text block
    metrics_text = "\n".join(
        f"{metric_name}: {outcome_value:.4f}"
        for i, (metric_name, outcome_value) in enumerate(metrics.items())
    )
    outcomes_text = "\n".join(
        f"{outcome_name}: {outcome_value}"
        for i, (outcome_name, outcome_value) in enumerate(outcomes.items())
    )

    ax.text(
        0.5,
        0.5,
        "Metrics: \n" + metrics_text + "\n\nOutcomes: \n" + outcomes_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="lightgray", alpha=0.3),
    )


def plot_confusion_matrix(TP, FP, TN, FN, ax, class_names=["Positive", "Negative"]):
    # Create confusion matrix
    confusion_matrix = torch.tensor([[TP, FN], [FP, TN]])
    threshold = confusion_matrix.max() * 0.5

    cax = ax.matshow(confusion_matrix, cmap="Blues")

    # Display CM values
    for i, j in product(
        range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])
    ):
        val = confusion_matrix[i, j]
        ax.text(
            j,
            i,
            f"{confusion_matrix[i,j].item()}",
            ha="center",
            va="center",
            color="white" if val > threshold else "black",
            fontsize=12,
        )

    # Set axis labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Predicted\n{name}" for name in class_names])
    ax.set_yticklabels([f"Actual\n{name}" for name in class_names])

    # Add color bar
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)


def add_color_legend(ax, position="left"):
    ax.axis("off")

    # Adjust text alignment based on position
    ha_alignment = "right" if position == "left" else "left"
    x_text = 0.60 if position == "left" else 0.05
    x_color = 0.65 if position == "right" else 0.12

    # Add color codes and labels
    for i, (label, color_rgb) in enumerate(color_dict.items()):
        # Normalize RGB to [0,1] for matplotlib
        color_norm = [c / 255 for c in color_rgb]

        # Add colored square
        ax.add_patch(
            mpatches.Rectangle(
                (x_color, 1 - 0.15 * (i + 1)),
                0.05,
                0.05,
                transform=ax.transAxes,
                facecolor=color_norm,
                edgecolor="black",
            )
        )

        # Add label
        ax.text(
            x_text,
            1 - 0.15 * (i + 1) + 0.020,
            color_names[label],
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment=ha_alignment,
        )


def save_figure(
    figure_name,
    image,
    label,
    generated_masks,
    generated_segmentation_images,
    generated_outcomes,
    calculated_metrics,
):
    fig, axes = plt.subplots(4, len(generated_masks) + 2, figsize=(7 * 3.5, 4 * 3))
    image = image.squeeze(0).squeeze(0)
    label = label.squeeze(0)

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Image")
    for i in range(4):
        axes[i, 0].axis("off")

    for i, (
        (epoch, mask),
        (_, outcomes),
        (_, metrics),
        (_, segmentation_image),
    ) in enumerate(
        zip(
            generated_masks.items(),
            generated_outcomes.items(),
            calculated_metrics.items(),
            generated_segmentation_images.items(),
        )
    ):
        axes[0, i + 1].imshow(mask, cmap="gray")
        axes[0, i + 1].set_title(f"Label epoch {epoch}")
        axes[0, i + 1].axis("off")

        # Display outcomes map segmentation
        axes[1, i + 1].imshow(segmentation_image)
        axes[1, i + 1].set_title("Segmentation")

        # Plot confusion matrix
        plot_confusion_matrix(
            outcomes["TP"],
            outcomes["FP"],
            outcomes["TN"],
            outcomes["FN"],
            axes[2, i + 1],
        )

        # Display metrics
        axes[3, i + 1].axis("off")
        add_metrics(axes[3, i + 1], metrics, outcomes)

    axes[0, -1].imshow(label, cmap="gray")
    axes[0, -1].set_title("True label")
    for i in range(4):
        axes[i, -1].axis("off")

    add_color_legend(axes[1, -1], position="right")

    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()


def generate_figures(model, dataitem, weights_path, figures_path):
    image, label, image_name = dataitem
    image, label = image.to(device), label.to(device)
    # image_name = image_name[0]
    print(f"Image name: {image_name}")

    generated_masks = {}
    generated_segmentation_images = {}
    generated_outcomes = {}
    calculated_metrics = {}
    for epoch in epochs:
        weights_fname = os.path.join(weights_path, f"weights_{epoch}_epochs.pth")
        # Load model weights
        model.load_state_dict(
            torch.load(weights_fname, map_location=device, weights_only=True)
        )
        # Generate predictions
        outputs, generated_masks[epoch] = generate_mask(model, image)
        image_outcomes, image_metrics = calculate_metrics(outputs, label)
        generated_segmentation_images[epoch] = generate_segmentation_image(
            outputs, label
        )
        epoch_metrics = {
            "Precision": image_metrics["precision"],
            "Recall": image_metrics["recall"],
            "F1-Score": image_metrics["f1score"],
            "Accuracy": image_metrics["accuracy"],
            "Specificity": image_metrics["specificity"],
            "f-IoU": image_metrics["f_iou"],
            "b-IoU": image_metrics["b_iou"],
            "m-IoU": image_metrics["m_iou"],
        }

        generated_outcomes[epoch] = image_outcomes
        calculated_metrics[epoch] = epoch_metrics

    # Generate plot
    output_filename = os.path.join(figures_path, image_name + ".png")
    label = torch.squeeze(label, 0)
    save_figure(
        output_filename,
        image,
        label,
        generated_masks,
        generated_segmentation_images,
        generated_outcomes,
        calculated_metrics,
    )


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
    figures_path = os.path.join(args.figures_path, "results", encoder_name, model_name)
    os.makedirs(figures_path, exist_ok=True)

    # For each instance of each dataloader we generate one plot
    figures_name_path = os.path.join(figures_path, args.dataset)
    os.makedirs(figures_name_path, exist_ok=True)
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
        generate_figures(
            model, [image, label, image_name], weights_path, figures_name_path
        )

    print(args.done_message)
