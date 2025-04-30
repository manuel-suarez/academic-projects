import os
import time
import torch
import argparse
import pandas as pd
import lightning as L
import segmentation_models_pytorch as smp
from models.unet_vgg16 import UnetVgg16
from models.unet_vgg19 import UnetVgg19
from models.unet_resnet18 import UnetResNet18
from models.unet_resnet34 import UnetResNet34

from data import CimatDataset
from model import CimatModel
from torch import nn, optim
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Configure device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Use SLURM array environment variables to determine training and cross validation set number
    slurm_array_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID"))
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    slurm_node_list = os.getenv("SLURM_JOB_NODELIST")
    print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    print(f"SLURM_JOB_NODELIST: {slurm_node_list}")

    train_file = "{:02}".format(slurm_array_task_id)
    cross_file = "{:02}".format(slurm_array_task_id)
    print(f"Train file: {train_file}")
    print(f"Cross file: {cross_file}")

    # Configure directories
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(
        home_dir,
        "data",
        "projects",
        "consorcio-ia",
        "data",
        "oil_spills_17",
        "augmented_dataset",
    )
    feat_dir = os.path.join(data_dir, "features")
    labl_dir = os.path.join(data_dir, "labels")
    train_dir = os.path.join(data_dir, "learningCSV", "trainingFiles")
    cross_dir = os.path.join(data_dir, "learningCSV", "crossFiles")

    # Check if results path exists
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)

    # Features to use
    feat_channels = ["ORIGIN", "ORIGIN", "VAR"]

    # Load CSV key files
    train_set = pd.read_csv(os.path.join(train_dir, f"train{train_file}.csv"))
    valid_set = pd.read_csv(os.path.join(cross_dir, f"cross{cross_file}.csv"))
    print(f"Training CSV file length: {len(train_set)}")
    print(f"Validation CSV file length: {len(valid_set)}")

    # Load generators
    train_keys = train_set["key"]
    valid_keys = valid_set["key"]
    train_dataset = CimatDataset(
        keys=train_keys,
        features_path=feat_dir,
        features_ext=".tiff",
        features_channels=feat_channels,
        labels_path=labl_dir,
        labels_ext=".pgm",
        dimensions=[224, 224, 3],
    )

    valid_dataset = CimatDataset(
        keys=valid_keys,
        features_path=feat_dir,
        features_ext=".tiff",
        features_channels=feat_channels,
        labels_path=labl_dir,
        labels_ext=".pgm",
        dimensions=[224, 224, 3],
    )
    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=12
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=2, shuffle=False, num_workers=4
    )

    # Load and configure model (segmentation_models_pytorch)

    # We must do a comparison between the SMP and custom models implementations
    # Unet+vgg16, Unet+vgg19, Unet+resnet18, Unet+resnet34
    models = [
        UnetVgg16(),
        UnetVgg19(),
        UnetResNet18(),
        UnetResNet34(),
        smp.Unet(
            encoder_name="vgg16",
            encoder_weights=None,
            classes=1,
            activation="sigmoid",
        ),
        smp.Unet(
            encoder_name="vgg19",
            encoder_weights=None,
            classes=1,
            activation="sigmoid",
        ),
        smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            classes=1,
            activation="sigmoid",
        ),
        smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            classes=1,
            activation="sigmoid",
        ),
    ]
    models_names = [
        "Unet+Vgg16",
        "Unet+Vgg19",
        "Unet+ResNet18",
        "Unet+ResNet34",
        "SMP-Unet+Vgg16",
        "SMP-Unet+Vgg19",
        "SMP-Unet+ResNet18",
        "SMP-Unet+ResNet34",
    ]

    model = models[slurm_array_task_id - 1]
    module = CimatModel(model=model)
    trainer = L.Trainer(max_epochs=int(args.num_epochs), devices=2, accelerator="gpu")
    print(f"Using model: {models_names[slurm_array_task_id-1]}")
    print("Lightning, 1 device, gpu")
    # Training
    print("[INFO] training the network...")
    startTime = time.time()
    trainer.fit(
        model=module,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    # display total time
    endTime = time.time()
    print(
        "[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime
        )
    )
