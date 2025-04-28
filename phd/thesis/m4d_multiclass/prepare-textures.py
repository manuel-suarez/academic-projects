import os
import argparse
import itertools
import subprocess


# Define a custom argument type for a list
def list_of_strings(arg):
    return arg.split(",")


# Configure command line arguments
parser = argparse.ArgumentParser()
# We are passing the epochs argument as a list of integers
parser.add_argument(
    "--distances",
    type=str,
    required=False,
    default="1,2,3",
    help="comma separated values for the distances of GLCM",
)
parser.add_argument(
    "--angles",
    type=str,
    required=False,
    default="0,np.pi/4,np.pi/2,3*np.pi/4",
    help="comma separated values for the angles of GLCM",
)
parser.add_argument(
    "--levels",
    type=str,
    required=False,
    default="16",
    help="gray levels for GLCM",
)
parser.add_argument(
    "--patch_size",
    type=str,
    required=False,
    default="9",
    help="patch size",
)
parser.add_argument(
    "--textures",
    type=str,
    required=False,
    default="contrast,dissimilarity,homogeneity,energy,correlation,asm,mean,variance,std,entropy",
    help="features to generate",
)
parser.add_argument(
    "--datasets",
    type=list_of_strings,
    required=False,
    default="train,test",
    help="comma separated list of dataset figures to generate",
)
parser.add_argument(
    "--versions", type=list_of_strings, required=False, default="v1,v2,v3,v4"
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
os.makedirs(os.path.join("outputs", "texture"), exist_ok=True)

# We are working directly with the dataset cause we are accesing by index
base_path = os.path.expanduser("~")
# Create main directory
figures_path = os.path.join(args.figures_path, "textures")
os.makedirs(figures_path, exist_ok=True)
for version, dataset_name in itertools.product(args.versions, args.datasets):
    os.makedirs(os.path.join(figures_path, version, dataset_name), exist_ok=True)
    data_path = os.path.join(base_path, "data", f"krestenitis_{version}", dataset_name)
    images_path = os.path.join(data_path, "images")

    # Get dataset length
    images_list = [fname.split(".")[0] for fname in os.listdir(images_path)]
    dataset_len = len(images_list)
    # We are using array_nodes argument to know how many processes need per node, if the values are not
    # completely divided by the other then add 1 to ensure process all values needed
    ntasks = min(args.ntasks, dataset_len)
    # array_nodes = dataset_len // ntasks + (1 if dataset_len % ntasks != 0 else 0)
    array_nodes = 1
    images_per_task = dataset_len // ntasks
    missing_images = dataset_len % ntasks
    # ntasks = dataset_len // array_nodes + (1 if dataset_len % array_nodes != 0 else 0)
    print(
        f"Dataset name: {dataset_name}, version: {version}, dataset len: {dataset_len}, array nodes: {array_nodes}, ntasks: {ntasks}, images per taks: {images_per_task}, missing images: {missing_images}"
    )
    subprocess.run(
        [
            "sbatch",
            f"--job-name=M4D-Textures-{dataset_name}-{version}",
            f"--array=1-{array_nodes}",
            f"--ntasks={ntasks}",
            "run-textures.slurm",
            args.distances,
            args.angles,
            args.levels,
            args.patch_size,
            dataset_name,
            version,
            str(images_per_task),
            str(missing_images),
        ]
    )
