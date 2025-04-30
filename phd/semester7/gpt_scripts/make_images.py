import os
import rasterio
import numpy as np

from matplotlib import pyplot as plt
from skimage.io import imread
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data", "cimat", "dataset-cimat")
images_path = os.path.join(data_path, "tiff")
labels_path = os.path.join(data_path, "mask_bin")
textures_path = os.path.join(data_path, "textures")
figures_path = os.path.join(data_path, "figures")
os.makedirs(figures_path, exist_ok=True)

fnames = os.listdir(images_path)
fnames.sort()
textures = os.listdir(textures_path)
textures.sort()

images = [rasterio.open(os.path.join(images_path, fname)).read(1) for fname in fnames]
# Mask invalid pixels
masked_images = [np.ma.masked_inside(image, 0.0, 1e-5) for image in images]
# Scale to 0-1
scaled_images = [
    (masked_image - masked_image.min()) / (masked_image.max() - masked_image.min())
    for masked_image in masked_images
]
# Histograms
images_path = os.path.join(figures_path, "images")
os.makedirs(images_path, exist_ok=True)

cmap = plt.cm.viridis
cmap.set_bad(color="red")


def save_image(index):
    name = fnames[index].split(".")[0]
    image = scaled_images[index]
    print(name, image.min(), image.max())
    # Load label
    label = imread(os.path.join(labels_path, name + ".png"), as_gray=True)

    # Plot figures
    fig, axs = plt.subplots(6, 2, figsize=(38, 12))
    # Image and label
    axs[0, 0].imshow(image, cmap=cmap)
    axs[0, 0].set_title("SAR")
    axs[0, 1].imshow(label, cmap=cmap)
    axs[0, 1].set_title("Label")
    # Textures
    for index, texture_path in enumerate(textures):
        # Load texture
        texture = rasterio.open(
            os.path.join(textures_path, texture_path, name + ".tif")
        ).read(1)
        masked_texture = np.ma.masked_equal(texture, -9999)
        scaled_texture = (masked_texture - masked_texture.min()) / (
            masked_texture.max() - masked_texture.min()
        )
        axs[index % 5 + 1, index % 2].imshow(scaled_texture, cmap=cmap)
        axs[index % 5 + 1, index % 2].set_title(texture_path)

        del texture
        del masked_texture
        del scaled_texture
    plt.savefig(os.path.join(images_path, name + ".png"))
    plt.close()


for index in range(len(images)):
    save_image(index)
