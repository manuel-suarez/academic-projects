import os
import itertools
import subprocess

# Script to run the master script specifically with the required SLURM array indexes
# to run the pending training model

sizes = [34]
models = [
    "unet",
    # "linknet",
    # "pspnet",
    # "fpn",
    # "deeplabv3p",
    # "manet",
    # "unet2p",
    # "unet3p",
]
encoders = [
    "resnet",
    # "resnetmr",
    # "resnetmrv2_",
    "resnetmrv3_",
    # "resnetmdv1_",
    "resnetmdv2_",
    # "senet",
    # "senetmr",
    # "senetmrv2_",
    # "mrnet",
    # "mrnetv2_",
    # "mrnetv3_",
    # "cbamnet",
    # "cbamnetmr",
    # "cbamnetmrv2_",
]
channels = [
    "o",
    # "o-v",
    # "ta",
    # "tc",
    # "td",
    # "tn",
    # "te",
    # "tr",
    # "tm",
    # "tv",
    # "th",
    # "tx",
]

# Max num of models to run
total_models = len(sizes) * len(models) * len(encoders) * len(channels)

# Now we need to iterate over the directories to know which models has been trained
outputs_path = "outputs"
indexes = ""
os.makedirs(os.path.join(outputs_path, "master"), exist_ok=True)

for index, (size, model_name, encoder, feat_channels) in enumerate(
    itertools.product(sizes, models, encoders, channels)
):
    encoder_name = f"{encoder}{size}"
    encoder_path = os.path.join(outputs_path, encoder_name, model_name, feat_channels)
    if not os.path.exists(os.path.join(encoder_path, "training.txt")):
        # Append index to list of indexes to run
        indexes += f"{index},"
# Remove last comma and append array task limit
indexes = indexes[:-1] + "%1"
print(indexes)
subprocess.run(
    ["sbatch", f"--array={indexes}", "--exclude=g-0-12,g-0-8", "run-master.slurm"]
)
# print(f"sbatch --array={indexes} run-master.slurm")
