import os
import itertools
import numpy as np
import pandas as pd
import albumentations as A

from glob import glob
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from PIL import Image
from tqdm import tqdm

# Directories configuration
home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data")
src_path = os.path.join(data_path, "cimat", "dataset-cimat")
dst_path = os.path.join(data_path, "cimat", "dataset-cimat", "segmentation")
# Initial configuration
image_path = "image_norm"
label_path = "mask_bin"
patch_size = 224

Image.MAX_IMAGE_PIXELS = None

# Definimos el modelo U-Net con un backbone preentrenado (ResNet)
slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
slurm_ntasks = os.getenv("SLURM_NTASKS")
slurm_procid = os.getenv("SLURM_PROCID")
slurm_task_pid = os.getenv("SLURM_TASK_PID")
print("SLURM_ARRAY_TASK_ID: ", slurm_array_task_id)
print("SLURM_NTASKS: ", slurm_ntasks)
print("SLURM_PROCID: ", slurm_procid)
print("SLURM_TASK_PID: ", slurm_task_pid)

# Define transforms
transform = A.Compose(
    [
        A.RandomCrop(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)


def patchify_image(
    src_path,
    img_dir,
    mask_dir,
    dst_path,
    img_name,
    patch_size,
):
    print(
        src_path,
        img_dir,
        mask_dir,
        dst_path,
        img_name,
        patch_size,
    )
    # In this case we are counting how many patches were generated from this image using GLOB
    images_patches = glob(
        os.path.join(dst_path, "features", "origin", img_name + "_*_train.tif")
    )
    labels_patches = glob(os.path.join(dst_path, "labels", img_name + "_*_train.png"))

    total_patches = len(images_patches)
    patches_per_task = total_patches // int(slurm_ntasks)
    missing_patches_per_task = total_patches % int(slurm_ntasks)

    patches_indexes = [
        int(slurm_procid) * patches_per_task + index
        for index in range(patches_per_task)
    ]
    if int(slurm_procid) < missing_patches_per_task:
        additional_index = int(slurm_ntasks) * patches_per_task + int(slurm_procid)
        patches_indexes.append(additional_index)

    print("Patches indexes: ", patches_indexes)

    # Build patchex indexes to process considering the max patches, ntasks and task process id
    for patch_index in patches_indexes:
        image_patch_name = images_patches[patch_index]
        label_patch_name = labels_patches[patch_index]

        print("Processing patch name: ", image_patch_name)
        print("Label patch name: ", label_patch_name)

        # Original image patch
        image_patch = imread(image_patch_name, as_gray=True)
        label_patch = imread(label_patch_name, as_gray=True)
        print("Image patch shape: ", image_patch.shape)

        # Check if oil pixels are 12% of the patch
        oil_pixels = np.count_nonzero(label_patch == 1)
        total_pixels = label_patch.size
        print("Oil pixels: ", oil_pixels)
        print("Total pixels: ", total_pixels)
        print("Oil pixels proportion: ", (oil_pixels / total_pixels))
        # if (oil_pixels / total_pixels) < 0.10:
        #    continue

        # Fixed num of augmentations
        num_of_patches = 1
        print(f"Generating {num_of_patches} augmented patches...")
        for i in range(num_of_patches):
            transformed = transform(image=image_patch, mask=label_patch)
            transformed_image = transformed["image"]
            transformed_label = transformed["mask"]
            # Save
            dst_name = image_patch_name.split("/")[-1]
            dst_name = dst_name.split(".")[0]
            imsave(
                os.path.join(
                    dst_path, "features", "origin", dst_name + f"_aug{i:03d}.tif"
                ),
                transformed_image,
                check_contrast=False,
            )
            imsave(
                os.path.join(dst_path, "labels", dst_name + f"_aug{i:03d}.png"),
                transformed_label,
                check_contrast=False,
            )


# Create output directories
os.makedirs(dst_path, exist_ok=True)
os.makedirs(os.path.join(dst_path, "features", "origin"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "labels"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "figures"), exist_ok=True)

fname = os.listdir(os.path.join(src_path, "image_norm"))[int(slurm_array_task_id) - 1]
patchify_image(
    src_path,
    "image_norm",
    "mask_bin",
    dst_path,
    fname.split(".")[0],
    patch_size,
)
print("Done!")
