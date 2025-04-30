import os
import itertools
import numpy as np
import pandas as pd
import albumentations as A

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

# Get total oil and sea pixels
mask_totals_df = pd.read_csv(
    os.path.join(dst_path, "counts", "totals", "total_count.csv")
)
total_oil_pixels = mask_totals_df["oil_pixels"].iloc[0]
total_sea_pixels = mask_totals_df["sea_pixels"].iloc[0]
total_pixels = total_oil_pixels + total_sea_pixels
# Open list of patches counts per image
image_patches_dfs = []
for fname in os.listdir(os.path.join(src_path, "image_norm")):
    image_patches_df = pd.read_csv(
        os.path.join(dst_path, "counts", "images", fname.split(".")[0] + ".csv")
    )
    image_patches_dfs.append(image_patches_df)
# Join dataframe
mask_images_patches = pd.concat(image_patches_dfs)
print("Mask images patches: ", len(mask_images_patches))
total_mask_patches = len(mask_images_patches)
# Sort by oil pixels (descending)
mask_images_patches = mask_images_patches.sort_values("oil_pixels", ascending=False)

print(
    f"Initial total pixel counts, oil: {total_oil_pixels}, sea: {total_sea_pixels}, total: {total_pixels}"
)
print(
    f"Percentage of pixel counts, oil: {round(total_oil_pixels/total_pixels,2)}, sea: {round(total_sea_pixels/total_pixels,2)}"
)
# Define transforms
transform = A.Compose(
    [
        A.RandomCrop(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)
# Iterate over list of patches augmenting those with more than 10% of oil pixels until we have approximate the equal
augmented_oil_pixels = 0
augmented_sea_pixels = 0
for index, row in tqdm(mask_images_patches.iterrows()):
    patch_oil_pixels = row["oil_pixels"]
    patch_sea_pixels = row["sea_pixels"]
    patch_total_pixels = row["total_pixels"]
    patch_percentage_oil_pixels = round(patch_oil_pixels / patch_total_pixels * 100, 2)
    if patch_percentage_oil_pixels >= 10.0:
        num_of_patches = int(round(patch_percentage_oil_pixels, 0))
        patch_name = row["patch_name"]
        patch_image = imread(
            os.path.join(dst_path, "features", "origin", patch_name + ".tif")
        )
        patch_label = imread(os.path.join(dst_path, "labels", patch_name + ".png"))
        for i in range(num_of_patches):
            # Apply augmentation and save
            transformed = transform(image=patch_image, mask=patch_label)
            transformed_image = transformed["image"]
            transformed_mask = transformed["mask"]
            # Save
            imsave(
                os.path.join(
                    dst_path, "features", "origin", patch_name + f"_aug{i:03d}.tif"
                ),
                transformed_image,
                check_contrast=False,
            )
            imsave(
                os.path.join(dst_path, "labels", patch_name + f"_aug{i:03d}.png"),
                transformed_mask,
                check_contrast=False,
            )

        augmented_oil_pixels += patch_oil_pixels * num_of_patches
        augmented_sea_pixels += patch_sea_pixels * num_of_patches
        total_mask_patches += num_of_patches

# total of oil and sea pixels
augmented_total_pixels = augmented_oil_pixels + augmented_sea_pixels
print(
    f"Augmented total pixel counts, oil: {augmented_oil_pixels}, sea: {augmented_sea_pixels}, total: {augmented_total_pixels}"
)
percentage_oil_pixels = round(augmented_oil_pixels / augmented_total_pixels, 2)
percentage_sea_pixels = round(augmented_sea_pixels / augmented_total_pixels, 2)
print(
    f"Percentage of augmented pixel counts, oil: {percentage_oil_pixels}, sea: {percentage_sea_pixels}"
)
total_oil_pixels += augmented_oil_pixels
total_sea_pixels += augmented_sea_pixels
total_pixels = total_oil_pixels + total_sea_pixels
print(
    f"Final total pixel counts, oil: {total_oil_pixels}, sea: {total_sea_pixels}, total: {total_pixels}"
)
print(
    f"Percentage of pixel counts, oil: {round(total_oil_pixels/total_pixels,2)}, sea: {round(total_sea_pixels/total_pixels,2)}"
)
print(f"Total patches count: {total_mask_patches}")
print("Done!")
