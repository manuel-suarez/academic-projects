import os
import argparse
import itertools
import subprocess

# Configure command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs", type=int, required=True, default=5, help="number of epochs to train"
)
args = parser.parse_args()
print("Args: ", args)

epochs = args.epochs
# Script to run the master script specifically with the required SLURM array indexes
# to run the pending training model

sizes = [18, 34]
models = [
    "unet",
    "linknet",
    "pspnet",
    "fpn",
    "deeplabv3p",
    "manet",
    "unet2p",
    "unet3p",
]
encoders = [
    "resnet",
    "resnetmr",
    "resnetmrv2_",
    "resnetmrv3_",
    "resnetmrv4_",
    "resnetmrv5_",
    "resnetmrv6_",
    "resnetmrv7_",
    "resnetmrv8_",
    # "senet",
    # "senetmr",
    # "senetmrv2_",
    # "cbamnet",
    # "cbamnetmr",
    # "cbamnetmrv2_",
    # "mrnet",
    # "mrnetv2_",
]

# Max num of models to run
total_models = len(sizes) * len(models) * len(encoders)

# Now we need to iterate over the directories to know which models has been trained
outputs_path = "outputs"
# Make training paths
os.makedirs(outputs_path, exist_ok=True)
os.makedirs(os.path.join(outputs_path, "master"), exist_ok=True)
indexes = ""

for index, (size, model_name, encoder) in enumerate(
    itertools.product(sizes, models, encoders)
):
    encoder_name = f"{encoder}{size}"
    encoder_path = os.path.join(outputs_path, encoder_name, model_name)
    if not os.path.exists(os.path.join(encoder_path, "training.txt")):
        # Append index to list of indexes to run
        indexes += f"{index},"
# Remove last comma and append array task limit
indexes = indexes[:-1] + "%1"
print(indexes)
subprocess.run(
    [
        "sbatch",
        f"--array={indexes}",
        "--exclude=g-0-12,g-0-8",
        "run-master.slurm",
        str(epochs),
    ]
)
# print(f"sbatch --array={indexes} run-master.slurm")
