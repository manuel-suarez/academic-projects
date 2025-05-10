import os
import itertools
import numpy as np
import pandas as pd

from skimage.io import imread, imsave
from PIL import Image
from tqdm import tqdm

# Directories configuration
home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "data", "cimat", "dataset-cimat")
src_dir = os.path.join(data_dir, "image_tiff")
dst_dir = os.path.join(data_dir, "image_patches")
os.makedirs(dst_dir, exist_ok=True)
# Initial configuration
patch_size = 224

Image.MAX_IMAGE_PIXELS = None


def patchify_image(src_path, dst_path, img_name, patch_size):
    image = imread(os.path.join(src_path, img_name), as_gray=True)

    image_height, image_width = image.shape
    count_x = int(image_width // patch_size) + 1
    count_y = int(image_height // patch_size) + 1
    count_0 = 0
    count_patches = 0

    for index, (j, i) in tqdm(
        enumerate(itertools.product(range(count_y), range(count_x)))
    ):
        y = patch_size * j
        x = patch_size * i

        if x + patch_size > image_width:
            x = image_width - patch_size - 1
        if y + patch_size > image_height:
            y = image_height - patch_size - 1

        dst_name = img_name.split(".")[0] + f"_{index:04d}_train.tif"
        # print(f"Image patch: {index}, i,j = ({i}, {j}), x,y = ({x}, {y})")
        image_patch = image[y : y + patch_size, x : x + patch_size]
        # print(f"Image patch: {image_patch.shape}")
        min_patch = np.min(image_patch)
        max_patch = np.max(image_patch)
        if min_patch == 0 and max_patch == 0:
            count_0 = count_0 + 1
        count_patches = count_patches + 1
        # print(f"Image patch: min: {min_patch}, max: {max_patch}")
        # Save patch
        imsave(os.path.join(dst_path, dst_name), image_patch, check_contrast=False)
    print(
        f"{img_name}, width, height: ({image_width, image_height}), total patches: {count_patches}, patches with zeros: {count_0}"
    )
    return image_width, image_height, count_patches, count_0


results = {"image": [], "width": [], "height": [], "patches": [], "zeros": []}
for fname in os.listdir(src_dir):
    width, height, patches, zeros = patchify_image(src_dir, dst_dir, fname, patch_size)
    results["image"].append(fname)
    results["width"].append(width)
    results["height"].append(height)
    results["patches"].append(patches)
    results["zeros"].append(zeros)
results_df = pd.DataFrame.from_dict(results)
results_df.to_csv("results_images.csv")
print("Done!")
