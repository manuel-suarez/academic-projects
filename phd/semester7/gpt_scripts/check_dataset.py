import os
import rasterio
import numpy as np
import pandas as pd

from skimage.io import imread
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data", "cimat", "dataset-cimat")
images_path = os.path.join(data_path, "tiff")
labels_path = os.path.join(data_path, "mask_bin")
textures_path = os.path.join(data_path, "textures")

# Load products filenames
fnames = [fname.split(".")[0] for fname in os.listdir(images_path)]
fnames.sort()
# Load textures directories names
textures = os.listdir(textures_path)
textures.sort()

# Load tiff images and check scales
images = [
    rasterio.open(os.path.join(images_path, fname + ".tif")).read(1) for fname in fnames
]
# Mask invalid pixels
masked_images = [np.ma.masked_inside(image, 0.0, 1e-5) for image in images]
# Scaled to 0-1
scaled_images = [
    (masked_image - masked_image.min()) / (masked_image.max() - masked_image.min())
    for masked_image in masked_images
]
# Print scales
results_dict = {
    "product": [],
    "image_min": [],
    "image_max": [],
    **{f"{texture}_min": [] for texture in textures},
    **{f"{texture}_max": [] for texture in textures},
}
for fname, image, masked_image, scaled_image in zip(
    fnames, images, masked_images, scaled_images
):
    print(fname)
    print("\tImage: ", image.min(), image.max())
    print("\tMasked image: ", masked_image.min(), masked_image.max())
    print("\tScaled image: ", scaled_image.min(), scaled_image.max())
    results_dict["product"].append(fname)
    results_dict["image_min"].append(image.min())
    results_dict["image_max"].append(image.max())
    # Check textures
    print("\tTextures:")
    for texture_path in textures:
        # Load texture
        texture_image = rasterio.open(
            os.path.join(textures_path, texture_path, fname + ".tif")
        ).read(1)
        masked_texture = np.ma.masked_equal(texture_image, -9999)
        scaled_texture = (masked_texture - masked_texture.min()) / (
            masked_texture.max() - masked_texture.min()
        )
        print(f"\t{texture_path}")
        print("\t\tTexture: ", texture_image.min(), texture_image.max())
        print("\t\tMasked texture: ", masked_texture.min(), masked_texture.max())
        print("\t\tScaled texture: ", scaled_texture.min(), scaled_texture.max())
        results_dict[f"{texture_path}_min"].append(masked_texture.min())
        results_dict[f"{texture_path}_max"].append(masked_texture.max())

results_df = pd.DataFrame(results_dict)
results_df.to_latex()
print("Done!")
