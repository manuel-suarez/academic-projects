import os
import itertools
import rasterio
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from PIL import Image
from tqdm import tqdm

# Directories configuration
home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data")
src_path = os.path.join(data_path, "cimat", "dataset-ipicyt")
dst_path = os.path.join(data_path, "cimat", "dataset-ipicyt", "segmentation")
# Initial configuration
image_path = "Oil"
label_path = "Mask_oil"
patch_size = 224

Image.MAX_IMAGE_PIXELS = None

# Definimos el modelo U-Net con un backbone preentrenado (ResNet)
slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID", 1)
slurm_ntasks = os.getenv("SLURM_NTASKS", 64)
slurm_procid = os.getenv("SLURM_PROCID", 0)
slurm_task_pid = os.getenv("SLURM_TASK_PID", 0)
print("SLURM_ARRAY_TASK_ID: ", slurm_array_task_id)
print("SLURM_NTASKS: ", slurm_ntasks)
print("SLURM_PROCID: ", slurm_procid)
print("SLURM_TASK_PID: ", slurm_task_pid)


def scale_image(image):
    min_image = np.min(image)
    max_image = np.max(image)
    image_scaled = (image - min_image) / (max_image - min_image)
    return image_scaled


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
    # In this case we are opening both image and mask to patchify at the same time
    # considering that we are removing outside regions pixels (SAR image) and separating
    # oil from not oil spill patches
    product = rasterio.open(os.path.join(src_path, img_dir, img_name + ".tif"))
    mask = rasterio.open(os.path.join(src_path, mask_dir, img_name + ".tif"))

    image_vh = product.read(1)
    image_vv = product.read(2)
    mask = mask.read(1)
    # Scale image between 0 and 1
    image_vh_scaled = scale_image(image_vh)
    image_vv_scaled = scale_image(image_vv)

    # Verifying that image and mask have the same shape
    image_height, image_width = image_vh.shape
    mask_height, mask_width = mask.shape
    print("Image shape: ", image_vh.shape)
    print("Mask shape: ", mask.shape)
    if (image_height != mask_height) or (image_width != mask_width):
        print("Error, image and mask must have the same dimensions")
        exit(-1)

    count_x = int(image_width // patch_size) + 1
    count_y = int(image_height // patch_size) + 1

    total_patches = count_x * count_y
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
    patches_positions = list(itertools.product(range(count_y), range(count_x)))
    for patch_index in patches_indexes:
        j, i = patches_positions[patch_index]

        print("Processing patch index: ", patch_index, i, j)
        # Get pixel positions for patch
        y = patch_size * j
        x = patch_size * i

        # Crop whenever patch size is outside image
        if x + patch_size > image_width:
            x = image_width - patch_size - 1
        if y + patch_size > image_height:
            y = image_height - patch_size - 1
        print("Position on image: ", y, x)
        print("Patch size: ", patch_size)

        # Original image patch
        image_vh_patch = image_vh[y : y + patch_size, x : x + patch_size]

        print("Image patch shape: ", image_vh_patch.shape)
        # Scaled image patch
        image_vh_scaled_patch = image_vh_scaled[y : y + patch_size, x : x + patch_size]
        image_vv_scaled_patch = image_vv_scaled[y : y + patch_size, x : x + patch_size]
        total_patches = total_patches + 1
        dst_img_name = img_name + f"_{patch_index:04d}_train"
        mask_patch = mask[y : y + patch_size, x : x + patch_size]
        print("Mask patch shape: ", mask_patch.shape)

        # Check if oil pixels are 12% of the patch
        oil_pixels = np.count_nonzero(mask_patch == 1)
        total_pixels = mask_patch.size
        print("Oil pixels: ", oil_pixels)
        print("Total pixels: ", total_pixels)
        print("Oil pixels proportion: ", (oil_pixels / total_pixels))
        if (oil_pixels / total_pixels) < 0.01:
            continue

        imsave(
            os.path.join(dst_path, "features", "origin_vh", dst_img_name + ".tif"),
            image_vh_scaled_patch,
            check_contrast=False,
        )
        imsave(
            os.path.join(dst_path, "features", "origin_vv", dst_img_name + ".tif"),
            image_vv_scaled_patch,
            check_contrast=False,
        )
        # Save in png for visualization
        image_vh_to_save = Image.fromarray(
            (image_vh_scaled_patch * 255).astype(np.int16)
        )
        image_vh_to_save.save(
            os.path.join(dst_path, "images", dst_img_name + "_vh.png")
        )
        image_vv_to_save = Image.fromarray(
            (image_vv_scaled_patch * 255).astype(np.int16)
        )
        image_vv_to_save.save(
            os.path.join(dst_path, "images", dst_img_name + "_vv.png")
        )
        imsave(
            os.path.join(dst_path, "labels", dst_img_name + ".png"),
            mask_patch,
            check_contrast=False,
        )
        # Save figure with image and mask patches
        fig, ax = plt.subplots(1, 3, figsize=(12, 8))
        ax[0].imshow(image_vh_scaled_patch, cmap="gray")
        ax[0].set_title("Image VH patch")
        ax[1].imshow(image_vv_scaled_patch, cmap="gray")
        ax[1].set_title("Image VV patch")
        ax[2].imshow(mask_patch, cmap="gray")
        ax[2].set_title("Mask patch")
        fig.tight_layout()
        fig.suptitle(dst_img_name)
        plt.savefig(os.path.join(dst_path, "figures", dst_img_name + ".png"))
        plt.close()


# Create output directories
os.makedirs(dst_path, exist_ok=True)
os.makedirs(os.path.join(dst_path, "features", "origin_vh"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "features", "origin_vv"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "labels"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "figures"), exist_ok=True)

# There are 1200 images for IPICYT dataset so we need to divide them between 12 task arrays (100 images each one)
indexes = range(
    (int(slurm_array_task_id) - 1) * 100, int(slurm_array_task_id) * 100 - 1
)

for index in indexes:
    fname = os.listdir(os.path.join(src_path, "Oil"))[index]
    patchify_image(
        src_path,
        "Oil",
        "Mask_oil",
        dst_path,
        fname.split(".")[0],
        patch_size,
    )
print("Done!")
