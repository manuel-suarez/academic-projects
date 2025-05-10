import os
import itertools
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from PIL import Image
from tqdm import tqdm

# Directories configuration
home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data", "cimat")
src_path = os.path.join(data_path, "dataset-cimat")
dst_path = os.path.join(data_path, "dataset-cimat", "segmentation")
# Initial configuration
image_path = "image_norm"
label_path = "mask_bin"
patch_size = 224

Image.MAX_IMAGE_PIXELS = None


def count_patch_pixels(
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
    mask_patches_dict = {
        "patch_name": [],
        "total_pixels": [],
        "oil_pixels": [],
        "sea_pixels": [],
        "invalid_patch": [],
        "full_oil_patch": [],
        "full_sea_patch": [],
    }
    total_pixels = 0
    oil_pixels = 0
    sea_pixels = 0
    # In this case we are opening both image and mask to patchify at the same time
    # considering that we are removing outside regions pixels (SAR image) and separating
    # oil from not oil spill patches
    image = imread(os.path.join(src_path, img_dir, img_name + ".tif"), as_gray=True)
    mask = imread(os.path.join(src_path, mask_dir, img_name + ".png"), as_gray=True)
    # Scale image between 0 and 1
    min_image = np.min(image)
    max_image = np.max(image)
    image_scaled = (image - min_image) / (max_image - min_image)
    # Mark all pixels on the mask above 0 as oil
    mask[mask > 0] = 1

    # Verifying that image and mask have the same shape
    image_height, image_width = image.shape
    mask_height, mask_width = mask.shape
    if (image_height != mask_height) or (image_width != mask_width):
        print("Error, image and mask must have the same dimensions")
        exit(-1)

    count_x = int(image_width // patch_size) + 1
    count_y = int(image_height // patch_size) + 1

    # Build patchex indexes to process considering the max patches, ntasks and task process id
    patches_positions = list(itertools.product(range(count_y), range(count_x)))
    for patch_index, (j, i) in enumerate(patches_positions):
        print("Processing patch position: ", patch_index, i, j)
        # Get pixel positions for patch
        y = patch_size * j
        x = patch_size * i

        # Crop whenever patch size is outside image
        if x + patch_size > image_width:
            x = image_width - patch_size - 1
        if y + patch_size > image_height:
            y = image_height - patch_size - 1

        dst_img_name = img_name + f"_{patch_index:04d}_train"
        mask_patch_df = pd.read_csv(
            os.path.join(dst_path, "counts", "patches", dst_img_name + ".csv")
        )
        total_pixels += mask_patch_df["total_pixels"].iloc[0]
        oil_pixels += mask_patch_df["oil_pixels"].iloc[0]
        sea_pixels += mask_patch_df["sea_pixels"].iloc[0]
        mask_patches_dict["patch_name"].append(mask_patch_df["patch_name"].iloc[0])
        mask_patches_dict["total_pixels"].append(mask_patch_df["total_pixels"].iloc[0])
        mask_patches_dict["oil_pixels"].append(mask_patch_df["oil_pixels"].iloc[0])
        mask_patches_dict["sea_pixels"].append(mask_patch_df["sea_pixels"].iloc[0])
        mask_patches_dict["invalid_patch"].append(
            mask_patch_df["invalid_patch"].iloc[0]
        )
        mask_patches_dict["full_oil_patch"].append(
            mask_patch_df["full_oil_patch"].iloc[0]
        )
        mask_patches_dict["full_sea_patch"].append(
            mask_patch_df["full_sea_patch"].iloc[0]
        )

    return mask_patches_dict, total_pixels, oil_pixels, sea_pixels


# Create output directories
os.makedirs(dst_path, exist_ok=True)
os.makedirs(os.path.join(dst_path, "features", "origin"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "labels"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "figures"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "counts", "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "counts", "totals"), exist_ok=True)

mask_images_dict = {
    "image_name": [],
    "total_pixels": [],
    "oil_pixels": [],
    "sea_pixels": [],
}
total_pixels = 0
total_oil_pixels = 0
total_sea_pixels = 0
for fname in os.listdir(os.path.join(src_path, "image_norm")):
    image_patches_dict, image_total_pixels, image_oil_pixels, image_sea_pixels = (
        count_patch_pixels(
            src_path,
            "image_norm",
            "mask_bin",
            dst_path,
            fname.split(".")[0],
            patch_size,
        )
    )
    # Save image patches CSV
    image_patches_df = pd.DataFrame.from_dict(image_patches_dict)
    image_patches_df.to_csv(
        os.path.join(dst_path, "counts", "images", fname.split(".")[0] + ".csv")
    )
    # Accumulate images CSV
    mask_images_dict["image_name"].append(fname)
    mask_images_dict["total_pixels"].append(image_total_pixels)
    mask_images_dict["oil_pixels"].append(image_oil_pixels)
    mask_images_dict["sea_pixels"].append(image_sea_pixels)
    # Total CSV
    total_oil_pixels += image_oil_pixels
    total_sea_pixels += image_sea_pixels

# Save totals
mask_totals_dict = {
    "oil_pixels": [total_oil_pixels],
    "sea_pixels": [total_sea_pixels],
}
mask_totals_df = pd.DataFrame.from_dict(mask_totals_dict)
mask_totals_df.to_csv(os.path.join(dst_path, "counts", "totals", "total_count.csv"))
print("Done!")
