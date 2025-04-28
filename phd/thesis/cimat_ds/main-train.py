import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse

from utils import default
from models import build_model
from slurm import slurm_vars
from trainer import Trainer
from dataset import prepare_dataloaders

# Configure command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--random_seed",
    type=int,
    required=True,
    help="random seeder for reproducible results",
)
parser.add_argument(
    "--epochs", type=int, required=True, default=5, help="number of epochs to train"
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
    "--separator",
    type=str,
    required=False,
    default="-",
    help="features channel symbol separator",
)
parser.add_argument(
    "--weights_path",
    type=str,
    required=False,
    default="weights",
    help="path to store training weighs",
)
parser.add_argument(
    "--metrics_path",
    type=str,
    required=False,
    default="metrics",
    help="path to store metrics",
)
parser.add_argument(
    "--logging_path",
    type=str,
    required=False,
    default="outputs",
    help="path to store logging",
)
parser.add_argument(
    "--nodes",
    type=str,
    required=False,
    help="SLURM nodes where the task is run",
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
random_seed = args.random_seed
epochs = args.epochs
base_path = os.path.expanduser("~")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nodes = default(args.nodes, "")
# Model and encoder selection is based on arguments
model_name = default(args.model_name, "unet")
encoder_name = default(args.encoder_name, "base")
feat_channels = default(args.feat_channels, "oov")
separator = default(args.separator, "-")
logging.info(f"Device: {device}")
logging.info(f"Model name: {model_name}")
logging.info(f"Encoder name: {encoder_name}")
logging.info(f"Epochs: {epochs}")
logging.info(f"Features channels: {feat_channels}")
logging.info(f"Features separator: {separator}")
print("Initial seed: ", random_seed)
print("Nodes: ", nodes)
print("Device: ", device)
print("Model: ", model_name)
print("Encoder: ", encoder_name)
print("Epochs: ", epochs)
print("Features channels: ", feat_channels)
print("Features separator: ", separator)

# Set PyTorch random seed for reproducible results
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cuda.deterministic = True

if __name__ == "__main__":
    # Prepare data loaders
    train_dataloader, valid_dataloader, test_dataloader = prepare_dataloaders(
        base_dir=base_path,
        feat_channels=feat_channels,
        feat_separator=separator,
        max_images=args.max_images,
    )
    # Prepare model according to SLURM array task id
    # Output channels according to len of features channels len(oov)=3
    model = build_model(
        model_name=model_name,
        encoder_name=encoder_name,
        in_channels=len(feat_channels.split(separator)),
    ).to(device)
    # Prepare optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Configure paths
    metrics_path = os.path.join(
        args.metrics_path, encoder_name, model_name, feat_channels
    )
    weights_path = os.path.join(
        args.weights_path, encoder_name, model_name, feat_channels
    )
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    # Instance trainer
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        epochs,
        device,
        metrics_path=metrics_path,
        weights_path=weights_path,
    )
    # Start training process
    trainer.fit([train_dataloader, valid_dataloader, test_dataloader])
    # Create flag file to indicate main script that weight models has been generated
    f = open(
        os.path.join(
            "outputs", encoder_name, model_name, feat_channels, "training.txt"
        ),
        "x",
    )
    f.close()
    logging.info(args.done_message)
    print(args.done_message)
