import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd

from utils import default
from models import build_model
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
print("Args: ", args)


# Initial configuration
random_seed = args.random_seed
epochs = args.epochs
base_path = os.path.expanduser("~")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nodes = default(args.nodes, "")
model_name = default(args.model_name, "unet")
encoder_name = default(args.encoder_name, "resnet34")
print("Initial seed: ", random_seed)
print("Nodes: ", nodes)
print("Device: ", device)
print("Model: ", model_name)
print("Encoder: ", encoder_name)
print("Epochs: ", epochs)

# Set PyTorch random seed for reproducible results
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cuda.deterministic = True

# Save run configuration to CSV to reproducible results (add model, encoder, epochs, random_seed)
configuration_data = {
    "model_name": [model_name],
    "encoder_name": [encoder_name],
    "random_seed": [random_seed],
    "epochs": [epochs],
}
configuration_path = os.path.join("outputs", encoder_name, model_name)
os.makedirs(configuration_path, exist_ok=True)
configuration_ds = pd.DataFrame.from_dict(configuration_data)
configuration_ds.to_csv(
    os.path.join(configuration_path, "configuration_data.csv"),
    mode="a",
    index=False,
)

if __name__ == "__main__":
    # Prepare data loaders
    train_dataloader, test_dataloader = prepare_dataloaders(
        base_path,
        sat_dir="sentinel",
        max_train_images=args.max_train_images,
        max_test_images=args.max_test_images,
    )
    # Prepare model according to SLURM array task id
    model = build_model(
        model_name=model_name,
        encoder_name=encoder_name,
        in_channels=3,
    ).to(device)
    # Prepare optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    # Scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()
    # loss_fn = tvsersky_loss
    # loss_fn = JointLoss()

    # Configure paths
    metrics_path = os.path.join(args.metrics_path, encoder_name, model_name)
    weights_path = os.path.join(args.weights_path, encoder_name, model_name)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    # Instance trainer
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        loss_fn,
        epochs,
        device,
        metrics_path=metrics_path,
        weights_path=weights_path,
    )
    # Start training process
    trainer.fit([train_dataloader, test_dataloader])
    # Create flag file to indicate main script that weight models has been generated
    f = open(os.path.join("outputs", encoder_name, model_name, "training.txt"), "x")
    f.close()
    print(args.done_message)
