import os
import argparse
import subprocess

from dataset import KrestenitisDataset


# Define a custom argument type for a list
def list_of_strings(arg):
    return arg.split(",")


# Configure command line arguments
parser = argparse.ArgumentParser()
# We are passing the epochs argument as a list of integers
parser.add_argument(
    "--epochs",
    type=str,
    required=False,
    default="20,40,60,80,100",
    help="comma separated list of epoch weigths to load",
)
parser.add_argument(
    "--model_name", type=str, required=True, default="unet", help="model to use"
)
parser.add_argument(
    "--encoder_name", type=str, required=True, default="base", help="encoder to use"
)
parser.add_argument(
    "--datasets",
    type=list_of_strings,
    required=False,
    default="train,val,test",
    help="comma separated list of dataset figures to generate",
)
parser.add_argument(
    "--ntasks",
    type=int,
    required=False,
    default=500,  # 10 array ids
    help="num of tasks to use",
)
parser.add_argument(
    "--figures_path",
    type=str,
    required=False,
    default="figures",
    help="path to store figures",
)
args = parser.parse_args()
print("Args: ", args)

# We are working directly with the dataset cause we are accesing by index
base_path = os.path.expanduser("~")
figures_path = os.path.join(
    args.figures_path, "results", args.encoder_name, args.model_name
)
os.makedirs(figures_path, exist_ok=True)
for dataset_name in args.datasets:
    data_path = os.path.join(base_path, "data", "krestenitis_v1", dataset_name)
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels_1D")

    # Get dataset length
    images_list = [fname.split(".")[0] for fname in os.listdir(images_path)]
    dataset_len = len(images_list)
    # We are using array_nodes argument to know how many processes need per node, if the values are not
    # completely divided by the other then add 1 to ensure process all values needed
    ntasks = min(args.ntasks, dataset_len)
    # We are using only 1 node (for model selection) and ntasks for image processing (process more than one image on each task)
    array_nodes = 1
    images_per_task = dataset_len // ntasks
    missing_images = dataset_len % ntasks
    print(
        f"Dataset name: {dataset_name}, dataset len: {dataset_len}, array nodes: {array_nodes}, ntasks: {ntasks}, images per tasks: {images_per_task}, missing images: {missing_images}"
    )
    subprocess.run(
        [
            "sbatch",
            f"--job-name=Krestenitis-Figures-{dataset_name}",
            f"--array=1-{array_nodes}",
            f"--ntasks={ntasks}",
            "run-figures.slurm",
            args.model_name,
            args.encoder_name,
            args.epochs,
            dataset_name,
            str(images_per_task),
            str(missing_images),
            args.figures_path,
        ]
    )
