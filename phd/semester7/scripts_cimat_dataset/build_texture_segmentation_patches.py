import os
import rasterio
import argparse
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
patch_size = 224

Image.MAX_IMAGE_PIXELS = None


def patchify_image(
    src_path,
    img_dir,
    mask_dir,
    dst_path,
    txt_path,
    img_name,
    patch_size,
):
    print(
        src_path,
        img_dir,
        mask_dir,
        dst_path,
        txt_path,
        img_name,
        patch_size,
    )
    # In this case we are opening both image and mask to patchify at the same time
    # considering that we are removing outside regions pixels (SAR image) and separating
    # oil from not oil spill patches
    image = rasterio.open(os.path.join(src_path, img_dir, img_name + ".tif")).read(1)
    mask = imread(
        os.path.join(src_path, mask_dir, img_name + ".png"), as_gray=True
    ).astype(np.uint8)
    print("Mask: ", mask.min(), mask.max(), mask.shape, mask.dtype)
    mask[mask > 0] = 1
    print("Mask: ", mask.min(), mask.max(), mask.shape, mask.dtype)
    # Scale image between 0 and 1
    min_image = np.min(image)
    max_image = np.max(image)
    image_scaled = (image - min_image) / (max_image - min_image)
    # Mark all pixels on the mask above 0 as oil

    # Verifying that image and mask have the same shape
    image_height, image_width = image.shape
    mask_height, mask_width = mask.shape
    if (image_height != mask_height) or (image_width != mask_width):
        print("Error, image and mask must have the same dimensions")
        print("Image dimensions: ", image_height, image_width)
        print("Mask dimensions: ", mask_height, mask_width)
        exit(-1)

    count_x = int(image_width // patch_size) + 1
    count_y = int(image_height // patch_size) + 1
    invalid_patches = 0
    total_patches = 0
    oil_patches = 0
    full_oil_patches = 0
    empty_oil_patches = 0
    pixels_oil = 0

    for index, (j, i) in tqdm(
        enumerate(itertools.product(range(count_y), range(count_x)))
    ):
        # Get pixel positions for patch
        y = patch_size * j
        x = patch_size * i

        # Crop whenever patch size is outside image
        if x + patch_size > image_width:
            x = image_width - patch_size - 1
        if y + patch_size > image_height:
            y = image_height - patch_size - 1

        # Original image patch
        image_patch = image[y : y + patch_size, x : x + patch_size]
        # Scaled image patch
        image_scaled_patch = image_scaled[y : y + patch_size, x : x + patch_size]
        total_patches = total_patches + 1
        min_image_patch = np.min(image_patch)
        max_image_patch = np.max(image_patch)
        # We are checking if patch image values are 0, if so then continue next patch
        # (we are in an invalid SAR image patch)
        if min_image_patch == 0 and max_image_patch == 0:
            invalid_patches = invalid_patches + 1
            continue
        dst_img_name = img_name + f"_{index:04d}"
        mask_patch = mask[y : y + patch_size, x : x + patch_size]
        min_mask_patch = np.min(mask_patch)
        max_mask_patch = np.max(mask_patch)
        if min_mask_patch == 1 and max_mask_patch == 1:
            # Full oil patch
            full_oil_patches = full_oil_patches + 1
        if min_mask_patch == 0 and max_mask_patch == 0:
            # Empty oil patch
            empty_oil_patches = empty_oil_patches + 1
            continue

        # Count how many pixels in the mask are equal to 1
        pixels_oil += np.sum(mask_patch)
        # We are only saving patches with at least some content of oil
        oil_patches = oil_patches + 1
        imsave(
            os.path.join(dst_path, "features", "origin", dst_img_name + ".tif"),
            image_scaled_patch,
            check_contrast=False,
        )
        # Save in png for visualization
        image_to_save = Image.fromarray((image_scaled_patch * 255).astype(np.int16))
        image_to_save.save(os.path.join(dst_path, "images", dst_img_name + ".png"))
        imsave(
            os.path.join(dst_path, "labels", dst_img_name + ".png"),
            mask_patch,
            check_contrast=False,
        )
        # Save figure with image and mask patches
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].imshow(image_scaled_patch, cmap="gray")
        ax[0].set_title("Image patch")
        ax[1].imshow(mask_patch, cmap="gray")
        ax[1].set_title("Mask patch")
        fig.tight_layout()
        fig.suptitle(dst_img_name)
        plt.savefig(os.path.join(dst_path, "figures", "images", img_name + ".png"))
        plt.close()
    # Calculate percentage of pixel oils
    total_pixels = oil_patches * patch_size * patch_size
    percentage_pixels_oil = round(pixels_oil / total_pixels * 100, 2)
    print(
        f"{img_name}, width, height: ({image_width}, {image_height}), total patches: {total_patches}, invalid_patches: {invalid_patches}, oil patches: {oil_patches}, full oil patches: {full_oil_patches}, empty oil patches: {empty_oil_patches}, total pixels: {total_pixels}, pixels oil: {pixels_oil}, percentage pixels_oil: {percentage_pixels_oil}"
    )
    # Now traverse texture directories
    for texture_dir in os.listdir(os.path.join(src_path, txt_path)):
        print(f"Texture: {texture_dir}")
        # Open texture image
        texture_image = rasterio.open(
            os.path.join(src_path, txt_path, texture_dir, img_name + ".tif")
        ).read(1)
        # Scale texture_image between 0 and 1
        texture_image_scaled = (texture_image - texture_image.min()) / (
            texture_image.max() - texture_image.min()
        )

        # Verifying that image and mask have the same shape
        texture_image_height, texture_image_width = texture_image.shape
        # if (texture_image_height != mask_height) or (texture_image_width != mask_width):
        #    print(
        #        f"Error, texture image {texture_dir} and mask must have the same dimensions"
        #    )
        #    print("Texture dimensions: ", texture_image_height, texture_image_width)
        #    print("Mask dimensions: ", mask_height, mask_width)
        #    exit(-1)

        for index, (j, i) in tqdm(
            enumerate(itertools.product(range(count_y), range(count_x)))
        ):
            # Get pixel positions for patch
            y = patch_size * j
            x = patch_size * i

            # Crop whenever patch size is outside image
            if x + patch_size > image_width:
                x = image_width - patch_size - 1
            if y + patch_size > image_height:
                y = image_height - patch_size - 1

            # Original image patch
            image_patch = image[y : y + patch_size, x : x + patch_size]
            # Scaled image patch
            texture_image_scaled_patch = texture_image_scaled[
                y : y + patch_size, x : x + patch_size
            ]
            min_image_patch = np.min(image_patch)
            max_image_patch = np.max(image_patch)
            # We are checking if patch image values are 0, if so then continue next patch
            # (we are in an invalid SAR image patch)
            if min_image_patch == 0 and max_image_patch == 0:
                continue
            dst_img_name = img_name + f"_{index:04d}"
            mask_patch = mask[y : y + patch_size, x : x + patch_size]
            min_mask_patch = np.min(mask_patch)
            max_mask_patch = np.max(mask_patch)
            if min_mask_patch == 1 and max_mask_patch == 1:
                # Full oil patch
                pass
            if min_mask_patch == 0 and max_mask_patch == 0:
                # Empty oil patch
                continue

            # Count how many pixels in the mask are equal to 1
            # We are only saving patches with at least some content of oil
            os.makedirs(
                os.path.join(dst_path, "features", "texture", texture_dir),
                exist_ok=True,
            )
            imsave(
                os.path.join(
                    dst_path, "features", "texture", texture_dir, dst_img_name + ".tif"
                ),
                texture_image_scaled_patch,
                check_contrast=False,
            )

    return (
        image_width,
        image_height,
        total_patches,
        invalid_patches,
        oil_patches,
        full_oil_patches,
        empty_oil_patches,
        total_pixels,
        pixels_oil,
        percentage_pixels_oil,
    )


parser = argparse.ArgumentParser(
    prog="GenTexturePatches", description="Generate texture patches"
)
parser.add_argument("--filename")
args = parser.parse_args()
print(args)

# Create output directories
os.makedirs(dst_path, exist_ok=True)
os.makedirs(os.path.join(dst_path, "features", "origin"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "labels"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "figures"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "figures", "images"), exist_ok=True)
os.makedirs(os.path.join(dst_path, "figures", "texture"), exist_ok=True)

fname = args.filename
patchify_image(
    src_path,
    "tiff",
    "mask_bin",
    dst_path,
    "textures",
    fname.split(".")[0],
    patch_size,
)
print("Done!")
