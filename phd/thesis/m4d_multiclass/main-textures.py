# Generación de filtros de texturas
import os
import argparse

import numpy as np

from skimage.io import imread, imsave
from skimage.util import view_as_windows
from skimage.feature import graycomatrix, graycoprops
from matplotlib import pyplot as plt


# Define a custom argument type for a list
def list_of_strings(arg):
    return arg.split(",")


# Configure command line arguments
parser = argparse.ArgumentParser()
# We are passing the epochs argument as a list of integers
parser.add_argument(
    "--distances",
    type=list_of_strings,
    required=False,
    # default="1,2,3",
    default="1,2,3",
    help="comma separated values for the distances of GLCM",
)
parser.add_argument(
    "--angles",
    type=list_of_strings,
    required=False,
    # default="0,np.pi/4,np.pi/2,3*np.pi/4",
    default="0,np.pi/4,np.pi/2,3*np.pi/4",
    help="comma separated values for the angles of GLCM",
)
parser.add_argument(
    "--levels", type=int, required=False, default=16, help="levels of gray for GLCM"
)
parser.add_argument(
    "--patch_size", type=int, required=False, default=9, help="patch size"
)
parser.add_argument(
    "--version",
    type=str,
    required=False,
    default="v4",
    help="dataset version images to process",
)
parser.add_argument(
    "--textures",
    type=list_of_strings,
    required=False,
    default="contrast,dissimilarity,homogeneity,energy,correlation,asm,mean,variance,std,entropy",
    help="features to generate",
)
textures_titles = {
    "contrast": "Contrast",
    "dissimilarity": "Dissimilarity",
    "homogeneity": "Homogeneity",
    "asm": "ASM",
    "energy": "Energy",
    "correlation": "Correlation",
    "mean": "GLCM Mean",
    "variance": "GLCM Var",
    "std": "GLCM Std",
    "entropy": "Entropy",
}
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
    "--done_message",
    type=str,
    required=False,
    default="Done!",
    help="message to show on training end",
)
args = parser.parse_args()
print("Args: ", args)

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
# dataindex = (int(slurm_array_task_id) - 1) * int(slurm_ntasks) + int(slurm_procid)
images_indexes = [
    int(slurm_procid) * images_per_task + index for index in range(images_per_task)
]
if int(slurm_procid) < args.missing_images:
    additional_index = int(slurm_ntasks) * images_per_task + int(slurm_procid)
    images_indexes.append(additional_index)
print("Images indexes: ", images_indexes)


# Initial configuration
base_path = os.path.expanduser("~")
# Define distance and angles for GLCM computation
distances = [int(d) for d in args.distances]
angles = [eval(expr) for expr in args.angles]
# Parameters
patch_size = args.patch_size
# Textures list
textures = args.textures
# Version data to process
version = args.version


def generate_textures(image):
    # Need to convert image to 255 type int
    image = (image * (args.levels - 1)).astype("uint8")

    # Get patches from the image
    patches = view_as_windows(image, (patch_size, patch_size))

    # Initialize one array for each one of the textures
    textures_maps = {texture: np.zeros(image.shape) for texture in textures}

    # Loop over each patch and calculate GLCM properties
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]

            # Compute the GLCM for the patch
            glcm = graycomatrix(
                patch,
                distances=distances,
                angles=angles,
                levels=args.levels,
                symmetric=True,
                normed=True,
            )

            # Calculate GLCM properties
            for texture in textures:
                texture_value = graycoprops(
                    glcm, "ASM" if texture == "asm" else texture
                )[0, 0]

                # Assign to the corresponding location in the map
                textures_maps[texture][
                    i + patch_size // 2, j + patch_size // 2
                ] = texture_value

    return textures_maps


if __name__ == "__main__":
    # We are working directly with the dataset cause we are accesing by index
    data_path = os.path.join(
        base_path, "data", f"krestenitis_{args.version}", args.dataset
    )
    # Configure paths
    images_path = os.path.join(data_path, "images")
    labels_path = os.path.join(data_path, "labels_1D")
    features_path = os.path.join(data_path, "textures")
    figures_path = os.path.join("figures", "textures", args.version, args.dataset)

    os.makedirs(features_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    # Create features paths
    for texture in textures:
        os.makedirs(os.path.join(features_path, texture), exist_ok=True)
    images_list = [fname.split(".")[0] for fname in os.listdir(images_path)]
    for image_index in images_indexes:
        print(f"Processing dataindex: {image_index}")
        # Get data (based on index)
        if image_index >= len(images_list):
            print(f"Image index {image_index} out of images list bounds")
            exit(0)
        image_name = images_list[image_index]
        image = imread(os.path.join(images_path, image_name + ".jpg"), as_gray=True)
        label = imread(os.path.join(labels_path, image_name + ".png"), as_gray=True)
        label[label > 0] = 1.0

        textures_maps = generate_textures(image)

        # Save results
        for texture in textures:
            imsave(
                os.path.join(features_path, texture, image_name + ".tif"),
                textures_maps[texture],
            )
        # Save figure
        fig, axs = plt.subplots(1, 2 + len(textures), figsize=(4 * len(textures), 4))
        axs[0].imshow(image, cmap="gray")
        axs[0].set_title("Image")
        axs[0].set_axis_off()
        for index, texture in enumerate(textures):
            axs[index + 1].imshow(textures_maps[texture], cmap="gray")
            axs[index + 1].set_title(textures_titles[texture])
            axs[index + 1].set_axis_off()
        axs[-1].imshow(label, cmap="gray")
        axs[-1].set_title("Label")
        axs[-1].set_axis_off()
        plt.savefig(os.path.join(figures_path, image_name + ".png"))
        plt.close()

    print(args.done_message)
