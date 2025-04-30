import os
import cv2
import rasterio
import numpy as np
from rasterio.plot import show
from matplotlib import pyplot as plt
from PIL import Image

work_dir = os.path.expanduser("~")
data_dir = os.path.join(work_dir, "data", "dataset-cimat")
tiff_dir = os.path.join(data_dir, "tiff")
norm_dir = os.path.join(data_dir, "tiff_norm")
var_dir = os.path.join(data_dir, "var_norm_jpg")
wind_dir = os.path.join(data_dir, "windfield_tiff")
mask_dir = os.path.join(data_dir, "binary_mask_gimp_png")

slurm_array_job_id = int(os.getenv('SLURM_ARRAY_JOB_ID'))
slurm_array_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")

def load_files(fname):
    img_tiff = rasterio.open(os.path.join(tiff_dir, fname)).read(1)
    img_norm = rasterio.open(os.path.join(norm_dir, fname)).read(1)
    img_var = np.array(Image.open(os.path.join(var_dir, fname.split(".")[0] + ".jpg")))
    img_wind = np.array(Image.open(os.path.join(wind_dir, fname)))
    img_mask = cv2.imread(
        os.path.join(mask_dir, fname.split(".")[0] + ".png"), cv2.IMREAD_GRAYSCALE
    )
    # Thresholding PNG binary mask
    _, img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)

    mask = (img_mask == 255).astype(np.int8)
    img_thrs = img_tiff * mask

    return img_tiff, img_norm, img_var, img_wind, img_mask, img_thrs


def plot_image_and_mask(fname, img_tiff, img_mask, img_thrs):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))
    titles = ["tiff", "mask", "segmentation"]
    # Display of original image, mask and segmentation
    for i, image in enumerate([img_tiff, img_mask, img_thrs]):
        im = axes[i].imshow(image, cmap="gray")
        axes[i].set_axis_off()
        axes[i].title.set_text(titles[i])
        fig.colorbar(im, ax=axes[i], fraction=0.070, pad=0.04)
    plt.tight_layout()
    fig.suptitle(fname.split(".")[0], fontsize=16)
    # plt.show()
    plt.savefig(f"{fname.split('.')[0]}_image_maks.png")


def plot_network_channels(fname, img_norm, img_var, img_wind):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))
    titles = ["norm", "var", "wind"]
    for i, image in enumerate([img_norm, img_var, img_wind]):
        im = axes[i].imshow(image, cmap="gray") if i < 2 else axes[i].imshow(image)
        axes[i].set_axis_off()
        axes[i].title.set_text(titles[i])
        fig.colorbar(im, ax=axes[i], fraction=0.070, pad=0.04)
    plt.tight_layout()
    fig.suptitle(fname.split(".")[0], fontsize=16)
    # plt.show()
    plt.savefig(f"{fname.split('.')[0]}_network_channels.png")


def plot_histograms(fname, img_tiff, img_norm, img_thrs):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))
    titles = ["tiff", "norm", "segmentation"]
    # Remove invalid values for histogram
    # img_tiff[(np.abs(img_tiff) < 1e-25).astype(bool)] = np.nan
    for i, image in enumerate([img_tiff, img_norm, img_thrs]):
        im = axes[i].hist(image)
        axes[i].title.set_text(titles[i])
    plt.tight_layout()
    fig.suptitle(fname.split(".")[0], fontsize=16)
    # plt.show()
    plt.savefig(f"{fname.split('.')[0]}_histograms.png")


def plot_thr_histograms(fname, img_tiff, img_thrs):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    titles = ["tiff", "segmentation"]

    im = axes[0].hist(img_tiff[np.abs(img_tiff) > 1e-25], bins=100000)
    axes[0].title.set_text("tiff")

    im = axes[1].hist(img_thrs[img_thrs < -5], bins=10000)
    axes[1].title.set_text("segmentation")

    plt.tight_layout()
    fig.suptitle(fname.split(".")[0], fontsize=16)
    # plt.show()
    plt.savefig(f"{fname.split('.')[0]}_thr_histograms.png")


def plot_image(fname):
    img_tiff, img_norm, img_var, img_wind, img_mask, img_thrs = load_files(fname)
    plot_image_and_mask(fname, img_tiff, img_mask, img_thrs)
    plot_network_channels(fname, img_norm, img_var, img_wind)
    plot_histograms(fname, img_tiff, img_norm, img_thrs)
    plot_thr_histograms(fname, img_tiff, img_thrs)

# Process file according to array task ID
fname = os.listdir(tiff_dir)[int(os.getenv('SLURM_ARRAY_TASK_ID'))]
print(fname)
plot_image(fname)
