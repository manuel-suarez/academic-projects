import os
import rasterio
import numpy as np

from matplotlib import pyplot as plt

home_path = os.path.expanduser("~")
data_path = os.path.join(home_path, "data", "cimat", "dataset-cimat")
images_path = os.path.join(data_path, "tiff")
figures_path = os.path.join(data_path, "figures")
os.makedirs(figures_path, exist_ok=True)

fnames = os.listdir(images_path)
fnames.sort()

images = [rasterio.open(os.path.join(images_path, fname)).read(1) for fname in fnames]
# Mask invalid pixels
masked_images = [np.ma.masked_inside(image, 0.0, 1e-5) for image in images]
# Scale to 0-1
scaled_images = [
    (masked_image - masked_image.min()) / (masked_image.max() - masked_image.min())
    for masked_image in masked_images
]
# Histograms
histograms_path = os.path.join(figures_path, "histograms")
os.makedirs(histograms_path, exist_ok=True)


def save_histogram(index):
    print(fnames[index].split(".")[0], images[index].min(), images[index].max())
    fig, axs = plt.subplots(1, 3)
    axs[0].hist(images[index])
    axs[0].set_title("SAR")
    axs[1].hist(masked_images[index].compressed())
    axs[1].set_title("Masked")
    axs[2].hist(scaled_images[index].compressed())
    axs[2].set_title("Scaled")
    plt.savefig(os.path.join(histograms_path, fnames[index].split(".")[0] + ".png"))
    plt.close()


for index in range(len(images)):
    save_histogram(index)
