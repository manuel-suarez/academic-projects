import os
import itertools
import subprocess
import argparse


def list_of_strings(arg):
    return arg.split(",")


# Configure command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--models_names", type=list_of_strings, required=True)
parser.add_argument("--encoders_names", type=list_of_strings, required=True)
parser.add_argument("--encoders_layers", type=list_of_strings, required=True)
parser.add_argument(
    "--encoders_mr_blocks", type=list_of_strings, required=False, default=[]
)
parser.add_argument(
    "--encoders_dal_layers", type=list_of_strings, required=False, default=[]
)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--pretrained", type=bool, required=True)
parser.add_argument("--slurm_job_name", type=str, required=False, default="")
args = parser.parse_args()
print("Args: ", args)

models_names = args.models_names
encoders_names = args.encoders_names
encoders_layers = args.encoders_layers
encoders_mr_blocks = args.encoders_mr_blocks
encoders_dal_layers = args.encoders_dal_layers
epochs = args.epochs
pretrained = args.pretrained
slurm_job_name = args.slurm_job_name
# Script to run the master script specifically with the required SLURM array indexes
# to run the pending training model


# Max num of models to run
total_models = (
    len(encoders_layers)
    * len(models_names)
    * len(encoders_names)
    * (len(encoders_mr_blocks) if len(encoders_mr_blocks) > 1 else 1)
    * (len(encoders_dal_layers) if len(encoders_dal_layers) > 1 else 1)
)

# Now we need to iterate over the directories to know which models has been trained
outputs_path = "outputs"
# Make training paths
os.makedirs(outputs_path, exist_ok=True)
os.makedirs(os.path.join(outputs_path, "master"), exist_ok=True)

indexes = ""
for index, (
    model_name,
    encoder_name,
    encoder_layers,
    encoder_mr_block,
    encoder_dal_layers,
) in enumerate(
    itertools.product(
        models_names,
        encoders_names,
        encoders_layers,
        encoders_mr_blocks if encoders_mr_blocks != [] else [""],
        encoders_dal_layers if encoders_dal_layers != [] else [""],
    )
):
    encoder_path_name = (
        f"{encoder_name}{encoder_layers}"
        + (f"_MR{encoder_mr_block}" if encoder_mr_block != "" else "")
        + (f"_MD{encoder_dal_layers}" if encoder_dal_layers != "" else "")
    )
    encoder_path = os.path.join(outputs_path, encoder_path_name, model_name)
    if not os.path.exists(os.path.join(encoder_path, "training.txt")):
        # Append index to list of indexes to run
        indexes += f"{index},"
# Remove last comma and append array task limit
indexes = indexes[:-1] + "%1"
print(indexes)
# For BASH
models = ",".join(models_names)
encoders = ",".join(
    [
        f"{encoder_name}{encoder_layers}"
        + (f"_MR{encoder_mr_block_version}" if encoder_mr_block_version != "" else "")
        + (
            f"_MD{encoder_dal_layers_version}"
            if encoder_dal_layers_version != ""
            else ""
        )
        for encoder_name, encoder_layers, encoder_mr_block_version, encoder_dal_layers_version in itertools.product(
            encoders_names,
            encoders_layers,
            encoders_mr_blocks if encoders_mr_blocks != [] else [""],
            encoders_dal_layers if encoders_dal_layers != [] else [""],
        )
        if not (encoder_mr_block_version == "" and encoder_dal_layers_version != "")
    ]
)
print("Models: ", models)
print("Encoders: ", encoders)
if slurm_job_name == "":
    subprocess.run(
        [
            "sbatch",
            f"--array={indexes}",
            "--exclude=g-0-12,g-0-8",
            "run-master.slurm",
            models,
            encoders,
            str(epochs),
            str(pretrained),
        ]
    )
else:
    subprocess.run(
        [
            "sbatch",
            f"--job-name={slurm_job_name}",
            f"--array={indexes}",
            "--exclude=g-0-12,g-0-8",
            "run-master.slurm",
            models,
            encoders,
            str(epochs),
            str(pretrained),
        ]
    )
