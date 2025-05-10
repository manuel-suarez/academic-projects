import os
import itertools
import numpy as np
import pandas as pd

from skimage.io import imread, imsave
from PIL import Image
from tqdm import tqdm

# Directories configuration
home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data", "cimat")
src_path = os.path.join(data_path, "dataset-cimat")
dst_path = os.path.join(data_path, "dataset-cimat", "classification")
not_oil_dir = "not_oil"
oil_dir = "oil"
# Initial configuration
patch_size = 224

Image.MAX_IMAGE_PIXELS = None


def patchify_image(
    src_path,
    img_dir,
    mask_dir,
    dst_path,
    oil_dir,
    not_oil_dir,
    img_name,
    patch_size,
    max_not_oil_patches=None,
):
    print(
        src_path,
        img_dir,
        mask_dir,
        dst_path,
        oil_dir,
        not_oil_dir,
        img_name,
        patch_size,
    )
    # In this case we are opening both image and mask to patchify at the same time
    # considering that we are removing outside regions pixels (SAR image) and separating
    # oil from not oil spill patches
    image = imread(os.path.join(src_path, img_dir, img_name + ".tif"), as_gray=True)
    mask = imread(os.path.join(src_path, mask_dir, img_name + ".png"), as_gray=True)
    # Scale image between 0 and 1
    min_image = np.min(image)
    max_image = np.max(image)
    image = (image - min_image) / (max_image - min_image)

    # Verifying that image and mask have the same shape
    image_height, image_width = image.shape
    mask_height, mask_width = mask.shape
    if (image_height != mask_height) or (image_width != mask_width):
        print("Error, image and mask must have the same dimensions")
        exit(-1)

    count_x = int(image_width // patch_size) + 1
    count_y = int(image_height // patch_size) + 1
    invalid_patches = 0
    oil_mask_patch = 0
    not_oil_mask_patch = 0
    total_patches = 0

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

        image_patch = image[y : y + patch_size, x : x + patch_size]
        total_patches = total_patches + 1
        min_image_patch = np.min(image_patch)
        max_image_patch = np.max(image_patch)
        # We are checking if patch image values are 0, if so then continue next patch (we are in an invalid SAR image patch)
        if min_image_patch == 0 and max_image_patch == 0:
            invalid_patches = invalid_patches + 1
            continue
        dst_img_name = img_name + f"_{index:04d}_train.tif"
        mask_patch = mask[y : y + patch_size, x : x + patch_size]
        min_mask_patch = np.min(mask_patch)
        max_mask_patch = np.max(mask_patch)
        if min_mask_patch == 0 and max_mask_patch == 0:
            if max_not_oil_patches is not None:
                if max_not_oil_patches == 0:
                    continue
                max_not_oil_patches = max_not_oil_patches - 1
            # Not oil patch
            not_oil_mask_patch = not_oil_mask_patch + 1
            # Output on not oil patches directory (for classification problem)
            imsave(
                os.path.join(dst_path, not_oil_dir, dst_img_name),
                image_patch,
                check_contrast=False,
            )
        else:
            # Oil patch
            oil_mask_patch = oil_mask_patch + 1
            # Output on oil patches directory (for classification problem)
            imsave(
                os.path.join(dst_path, oil_dir, dst_img_name),
                image_patch,
                check_contrast=False,
            )
    print(
        f"{img_name}, width, height: ({image_width}, {image_height}), total patches: {total_patches}, invalid_patches: {invalid_patches}, oil patches: {oil_mask_patch}, not oil patches: {not_oil_mask_patch}"
    )
    return (
        image_width,
        image_height,
        total_patches,
        invalid_patches,
        oil_mask_patch,
        not_oil_mask_patch,
    )


# Create output directories
os.makedirs(dst_path, exist_ok=True)
os.makedirs(os.path.join(dst_path, oil_dir), exist_ok=True)
os.makedirs(os.path.join(dst_path, not_oil_dir), exist_ok=True)

results = {
    "image": [],
    "width": [],
    "height": [],
    "total_patches": [],
    "invalid_patches": [],
    "oil_patches": [],
    "not_oil_patches": [],
}
for fname in os.listdir(os.path.join(src_path, "image_tiff")):
    width, height, total_patches, invalid_patches, oil_patches, not_oil_patches = (
        patchify_image(
            src_path,
            "image_tiff",
            "mask_bin",
            dst_path,
            oil_dir,
            not_oil_dir,
            fname.split(".")[0],
            patch_size,
            120,
        )
    )
    results["image"].append(fname)
    results["width"].append(width)
    results["height"].append(height)
    results["total_patches"].append(total_patches)
    results["invalid_patches"].append(invalid_patches)
    results["oil_patches"].append(oil_patches)
    results["not_oil_patches"].append(not_oil_patches)
results_df = pd.DataFrame.from_dict(results)
results_df.to_csv("results_classification.csv")
print("Done!")
